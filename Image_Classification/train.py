from WaveMix_Lite import WaveMix
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ExponentialLR
import time
import numpy as np
from dataset import NtutEMNIST
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import loadNtutEMNIST, saveModel, makeDirectory, setupLogger
import logging
import sys

def main():
    start_time = time.time()
    modeDir, ckptDir = makeDirectory('train', name=PROJECT_NAME)
    setupLogger(modeDir, mode='train')
    logging.info("Start training WaveMix model")
    logging.info(f'PROJECT_NAME = {PROJECT_NAME}')
    logging.info(f'GPUS = {GPUS}')
    logging.info(f'BATCH_SIZE = {BATCH_SIZE}')
    logging.info(f'VAL_BATCH_SIZE = {VAL_BATCH_SIZE}')
    logging.info(f'NUM_EPOCHS = {NUM_EPOCHS}')
    logging.info(f'TRAIN_DATASET_PATH = {TRAIN_DATASET_PATH}')
    torch.backends.cudnn.enabled=True
    
    # Define model
    model = WaveMix(
        num_classes=62,
        depth=7,
        mult=2,
        ff_channel=256,
        final_dim=256,
        dropout=0.5
    )
    model = nn.DataParallel(model, device_ids=GPUS).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.90)

    # Define dataloader
    x_train, y_train, x_val, y_val = loadNtutEMNIST(TRAIN_DATASET_PATH, val_factor=0.1)
    dataset = NtutEMNIST(x_train, y_train, data_aug=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True
        )
    n_train_batches = len(dataloader)

    val_dataset = NtutEMNIST(x_val, y_val)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
        )
    n_val_batches = len(val_dataloader)

    #training phase
    max_accuracy = max_epoch = 0
    log = {
        'train_loss_list':[],
        'train_acc_list':[],
        'val_loss_list':[],
        'val_acc_list':[],
    }
    scaler = GradScaler()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = epoch_acc = 0
        pbar = tqdm(dataloader)
        for i, (img, label) in enumerate(pbar):
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True).flatten().long()
            with autocast():
                # forward propagation
                predict = model(img)
                # calculate loss
                loss = F.cross_entropy(predict, label)
            epoch_loss += loss.item()
            # calculate accuracy
            pred_one_hot = torch.argmax(predict, dim=1)
            accuracy = float((pred_one_hot == label).float().mean())
            epoch_acc += accuracy
            # back propagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # Update progress bar
            pbar_msg = f'Epoch: [{epoch}/{NUM_EPOCHS}] Iter:[{i + 1}/{n_train_batches}], ' +\
                    f'lr: {["{:.6f}".format(x["lr"]) for x in optimizer.param_groups]}, ' +\
                    f'Loss: {loss.item():.2f}, Acc:{accuracy:.2f}'
            pbar.set_description(pbar_msg)
        lr_scheduler.step()
        log['train_acc_list'].append(epoch_acc / n_train_batches)
        log['train_loss_list'].append(epoch_loss / n_train_batches)
        
        # Save model checkpoint
        if epoch % 10 == 0:
            saveModel(model, log, os.path.join(ckptDir, f'epoch_{epoch}.pt'))

        # Validation
        model.eval()
        accuracy = epoch_loss = 0
        for i,(img, label) in enumerate(tqdm(val_dataloader, desc='Validation')):
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True).flatten().long()
            with autocast():
                # forward propagation
                predict = model(img)
                # calculate loss
                loss = F.cross_entropy(predict, label)
            epoch_loss += loss.item()
            # calculate accuracy
            pred_one_hot = torch.argmax(predict, dim=1)
            accuracy += float((pred_one_hot == label).float().mean())

        # Save best model
        if (val_acc := accuracy / n_val_batches) > max_accuracy:
            max_accuracy = val_acc
            max_epoch = epoch
            saveModel(model, log, os.path.join(ckptDir, 'best.pt'))

        log['val_acc_list'].append(accuracy / n_val_batches)
        log['val_loss_list'].append(epoch_loss / n_val_batches)
        logging.info(f"Epoch:{epoch}, " + \
            f"Train_acc:{log['train_acc_list'][-1]:.4f}, " + \
            f"Val_acc:{log['val_acc_list'][-1]:.4f}, " + \
            f"Train_loss:{log['train_loss_list'][-1]:.3f}, " + \
            f"Val_loss:{log['val_loss_list'][-1]:.3f}, " + \
            f"Max_accuracy:{max_accuracy:.4f} in epoch:{max_epoch}"
        )
        
    logging.info(time.strftime('Start: %Y.%m.%d %H:%M:%S',time.localtime(start_time)))
    logging.info(time.strftime('End: %Y.%m.%d %H:%M:%S',time.localtime(time.time())))
    
    # Visualize accuracy and loss
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(log['val_loss_list'], label="val")
    plt.plot(log['train_loss_list'], label="train")
    plt.legend()
    plt.subplot(122)
    plt.title("Accuracy")
    plt.plot(log['val_acc_list'], label="val")
    plt.plot(log['train_acc_list'], label="train")
    plt.legend()
    plt.savefig(os.path.join(modeDir, 'plot.png'))

if __name__=="__main__":
    GPUS = [0, 1]
    BATCH_SIZE = 2048 * len(GPUS)
    VAL_BATCH_SIZE = int(BATCH_SIZE * 0.2)
    NUM_EPOCHS = 60
    TRAIN_DATASET_PATH = "emnist-byclass-train.npz"
    PROJECT_NAME = 'WaveMix_Origin_Drop_Lr'
    main()