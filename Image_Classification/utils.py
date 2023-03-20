import numpy as np
import torch
import logging
import os
import time

def loadNtutEMNIST(img_root, val_factor=0, test=False, mean=0.1736, std=0.3246):
    data = np.load(img_root)
    if test:
        images = data['testing_images']
    else:
        images = data['training_images']
        labels = data['training_labels']

    images = images.reshape((images.shape[0], 28, 28, 1))
    images = (images - mean) / std

    if test:
        return images

    val_size = int(images.shape[0] * val_factor)
    x_val = images[:val_size]
    y_val = labels[:val_size]
    x_train = images[val_size:]
    y_train = labels[val_size:]

    return x_train, y_train, x_val, y_val

def saveModel(model, log:dict, save_path:str='checkpoint.pt') -> None:
    '''
    Save pytorch model as state_dict
    input:
        model: pytorch model
        log: a dictionary with training information
            {
                'loss_train_log': list[],
                'loss_val_log': list[],
                ...
            }
        save_name: a path with filename of saved model
    '''
    log['model_dict'] = model.state_dict()
    torch.save(log, save_path)
    logging.info(f'Model saved to: {save_path}')

def getProjectDir(modeDir:str, name, n):
    dirName = name + str(n) if n else name
    if not os.path.exists(projectDir := os.path.join(modeDir, dirName)):
        return projectDir
    else:
        return getProjectDir(modeDir, name, n + 1)

def makeDirectory(mode:str, name:str=None):
    if not os.path.exists(outputDir := 'runs'):
        os.mkdir(outputDir)
    if not os.path.exists(modeDir := os.path.join(outputDir, mode)):
        os.mkdir(modeDir)
    name = mode if name is None else name
    projectDir = getProjectDir(modeDir, name, 0)
    os.mkdir(projectDir)
    
    if mode == 'train':
        if not os.path.exists(ckptDir := os.path.join(projectDir, 'checkpoints')):
            os.mkdir(ckptDir)
        return projectDir, ckptDir
    else:
        return projectDir

def setupLogger(savePath:str, mode:str):
    datetime = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime(time.time()))
    logging.basicConfig(
        filename=os.path.join(savePath, f'{mode}_log_{datetime}.log'),
        level=logging.INFO, 
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
