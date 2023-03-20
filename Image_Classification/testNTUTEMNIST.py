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
import logging
from utils import makeDirectory, setupLogger, loadNtutEMNIST

def main():
    start_time = time.time()
    modeDir = makeDirectory('test')
    setupLogger(modeDir, mode='test')
    logging.info("Start testing WaveMix model")
    logging.info(f'GPUS = {GPUS}')
    logging.info(f'BATCH_SIZE = {BATCH_SIZE}')
    logging.info(f'MODEL_PARAM_PATH = {MODEL_PARAM_PATH}')
    logging.info(f'TEST_DATASET_PATH = {TEST_DATASET_PATH}')
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
    test_images = loadNtutEMNIST(TEST_DATASET_PATH, test=True)
    dataset = NtutEMNIST(test_images)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE
        )

    # Inference
    model.eval()
    pbar = tqdm(dataloader, desc='Inference')
    with open(os.path.join(modeDir, 'pred_results.csv'), 'w') as f:
        f.write('Id,Category\n')
        current_id = 0
        for i, img in enumerate(pbar):
            img = img.cuda(non_blocking=True)
            with autocast():
                # predict
                predict = model(img)
            pred_cls = torch.argmax(predict, dim=1)
            for single_pred in pred_cls:
                if current_id > 116321:
                    break
                f.write(f'{current_id},{single_pred}\n')
                current_id += 1

    logging.info(time.strftime('Start: %Y.%m.%d %H:%M:%S',time.localtime(start_time)))
    logging.info(time.strftime('End: %Y.%m.%d %H:%M:%S',time.localtime(time.time())))

if __name__=="__main__":
    GPUS = [0, 1]
    BATCH_SIZE = 16 * len(GPUS)
    MODEL_PARAM_PATH = 'runs/train/WaveMix_Origin_Drop_Lr/checkpoints/best.pt'
    TEST_DATASET_PATH = "emnist-byclass-test.npz"
    main()