from WaveMix_Lite import WaveMix
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
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
    modeDir = makeDirectory('validate')
    setupLogger(modeDir, 'validate')
    logging.info("Start validating WaveMix model")
    logging.info(f'GPUS = {GPUS}')
    logging.info(f'VAL_BATCH_SIZE = {VAL_BATCH_SIZE}')
    logging.info(f'TRAIN_DATASET_PATH = {TRAIN_DATASET_PATH}')
    logging.info(f'MODEL_PARAM_PATH = {MODEL_PARAM_PATH}')
    torch.backends.cudnn.enabled=True
    
    # Define model
    model = WaveMix(
        num_classes=62,
        depth=7,
        mult=2,
        ff_channel=256,
        final_dim=256,
        dropout=0.1
    )
    model = nn.DataParallel(model, device_ids=GPUS).cuda()
    model.load_state_dict(torch.load(MODEL_PARAM_PATH)['model_dict'])

    # Define dataloader
    _, _, x_val, y_val = loadNtutEMNIST(TRAIN_DATASET_PATH, val_factor=0.1)

    val_dataset = NtutEMNIST(x_val, y_val)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
        )
    n_val_batches = len(val_dataloader)

    # Validation
    model.eval()
    accuracy = total_loss = 0
    for i,(img, label) in enumerate(tqdm(val_dataloader, desc='Validation')):
        img = img.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True).flatten().long()
        with autocast():
            # forward propagation
            predict = model(img)
            # calculate loss
            loss = F.cross_entropy(predict, label)
        total_loss += loss.item()
        # calculate accuracy
        pred_one_hot = torch.argmax(predict, dim=1)
        accuracy += (pred_one_hot == label).float().mean()

    logging.info(
        f"Accuracy:{(accuracy / n_val_batches):.4f}, " + \
        f"Val_loss:{(total_loss / n_val_batches):.3f}"
    )
        
    logging.info(time.strftime('Start: %Y.%m.%d %H:%M:%S',time.localtime(start_time)))
    logging.info(time.strftime('End: %Y.%m.%d %H:%M:%S',time.localtime(time.time())))

if __name__=="__main__":
    GPUS = [0, 1]
    VAL_BATCH_SIZE = 16 * len(GPUS)
    TRAIN_DATASET_PATH = "emnist-byclass-train.npz"
    MODEL_PARAM_PATH = 'runs/train/checkpoints/best.pt'
    main()