

import os
import numpy as np 
import shutil
import yaml
import datetime
import time
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


from data.get_dataLoader import get_testLoader
from models.model import Resnet152
from utils import accuracy




def write_result(total_results, experiment_Dir):
    filename = os.path.join(experiment_Dir, 'preduct_result_ep44.csv')
    df = pd.DataFrame(total_results, columns =['filename', 'category']) 
    df.to_csv(filename, index=False)
    print('result save to:', filename)


def predict(images, paths, model):

    outputs = model(images)
    
    results = []
    for i in range(len(paths)):
        output = list(outputs[i])
        path = paths[i].split('/')[-1]
        result = str(output.index(max(output))).zfill(2)
        results.append([path, result])
    return results



def main(preds_dir=None, save_results=True):

    # Load project config data
    cfg = yaml.full_load(open("config.yml", 'r'))
    

    # Restore the model, LIME explainer, and model class indices from their respective serializations
    
    num_classes = cfg['DATA']['NUM_CLASSES']
    dataDir = cfg['PATHS']['BATCH_PRED_IMGS']
    experiment_Dir = cfg['PATHS']['MODEL_TO_LOAD']
    os.environ["CUDA_VISIBLE_DEVICES"]= cfg['TRAIN']['GPU']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emb_dim = cfg['NN']['RESNET152']['EMB_DIM']
    pretrained_model = os.path.join(experiment_Dir, 'model_best.pth.tar')
    #pretrained_model = os.path.join(experiment_Dir, 'ep_49.pth.tar')
    print('Model used: ', pretrained_model)

    test_loader = get_testLoader(dataDir)

    model = Resnet152(emb_dim, num_classes, LOAD_ORIGIN=False).to(device)  
    checkpoint = torch.load(pretrained_model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    total_results = []
    with torch.no_grad():
        for i, (images, _, paths) in tqdm(enumerate(test_loader)):
            results = predict(images.to(device), paths, model)
            total_results += results


    if save_results:
        write_result(total_results, experiment_Dir)

if __name__ == '__main__':
    
    results = main(preds_dir=None, save_results=True)