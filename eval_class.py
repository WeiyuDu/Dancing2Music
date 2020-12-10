import os
import argparse
import functools
import librosa
import shutil
import sys 
sys.path.insert(0, 'preprocess')
import preprocess as p
import subprocess as sp
from shutil import copyfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model_comp2 import *
from networks import *
from options import TestOptions
import modulate
import utils
from data import get_loader
import tqdm

#parser = TestOptions()
#args = parser.parse()
#args.train = False
cuda_device = 'cuda:0'
neta_snapshot = './data/stats/aud_3cls.ckpt'
data_dir = './data'
checkpoint2 = torch.load(neta_snapshot)
neta_cls = AudioClassifier_rnn(10,30,28,cls=3)
neta_cls.load_state_dict(checkpoint2)

neta_cls.eval()
neta_cls.to(cuda_device)
todo = ['ballet', 'zumba', 'hiphop']
for j in range(3):
    data_loader = get_loader(batch_size=256, shuffle=False, 
        num_workers=4, dataset=2, 
        data_dir=data_dir, eval=todo[j])
    cls1 = 0
    cls2 = 0
    cls3 = 0
    total = 0
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(data_loader)):
            (stdpSeq, aud) = batch
            aud = aud.to(cuda_device)
            curr_class = neta_cls(aud)
            label = torch.argmax(curr_class, dim=1)
            cls1 += torch.sum(label == 0).item()
            cls2 += torch.sum(label == 1).item()
            cls3 += torch.sum(label == 2).item()
            total += aud.shape[0]
    print(todo[j])
    print('total', total)
    print('1',cls1)
    print('2',cls2)
    print('3',cls3)