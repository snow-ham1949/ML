serial = 12
from torchensemble import VotingClassifier
from torchensemble.utils.io import load
# Main link
# !wget -O libriphone.zip "https://github.com/xraychen/shiny-robot/releases/download/v1.0/libriphone.zip"
# !unzip -oq libriphone.zip
# !ls libriphone

import os
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n) 
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41 # NOTE: pre-computed, should not need change
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
      phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

      for line in phone_file:
          line = line.strip('\n').split(' ')
          label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
      y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != 'test':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
      print(y.shape)
      return X, y
    else:
      return X

class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(p=0.25),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, lstm_hidden_dim = 256, lstm_hidden_layers = 5):
        super(Classifier, self).__init__()

        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, lstm_hidden_layers, batch_first=True, bidirectional=True)
       
        self.fc = nn.Sequential(
            BasicBlock(lstm_hidden_dim*2, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        lstm_output, (hidden, cell) = self.lstm(x)
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        # print("LSTM output", lstm_output.shape)
        x=self.fc(lstm_output[:,-1,:])
        return x

n_estimators= 8

concat_nframes = 53              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.95               # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 881105                        # random seed
batch_size = 512               
num_epoch = 35                 
learning_rate = 0.001         
model_path = './model'

# model parameters
# input_dim = concat_nframes
input_dim = 39
hidden_layers = 10               # the number of hidden layers
hidden_dim = 1000               # the hidden dim
lstm_hidden_dim = 256
lstm_hidden_layers = 3


import gc

# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)
train_X = torch.reshape(train_X,(train_X.shape[0], concat_nframes, 39))
# print(train_X.shape)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)
val_X = torch.reshape(val_X,(val_X.shape[0], concat_nframes, 39))

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')


import numpy as np

#fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(seed)

# create model, define a loss function, and optimizer
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim,lstm_hidden_dim = lstm_hidden_dim, lstm_hidden_layers = lstm_hidden_layers).to(device)
criterion = nn.CrossEntropyLoss() 
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = 0.05)
ensemble = VotingClassifier(
    estimator=model,               # here is your deep learning model
    n_estimators=n_estimators,                        # number of base estimators
    cuda=True
)
ensemble.set_criterion(criterion)
ensemble.set_optimizer(
    "Adam",                                 # type of parameter optimizer
    lr=learning_rate,                       # learning rate of parameter optimizer
    weight_decay=0.05,              # weight decay of parameter optimizer
)
ensemble.set_scheduler(
    "CosineAnnealingLR",                    # type of learning rate scheduler
    T_max=10,                           # additional arguments on the scheduler
)
ensemble.fit(
    train_loader = train_loader,
    epochs=num_epoch,                          # number of training epochs
    save_model = True,
    save_dir = "./model/",
    test_loader = val_loader
)
# accuracy = model.predict(test_loader)

# %%
del train_loader, val_loader
gc.collect()

test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)
test_X = torch.reshape(test_X,(test_X.shape[0], concat_nframes, 39))
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
del model
del ensemble
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim,lstm_hidden_dim = lstm_hidden_dim, lstm_hidden_layers = lstm_hidden_layers).to(device)
# load model
ensemble = VotingClassifier(
    estimator=model,               # here is your deep learning model
    n_estimators=n_estimators,                        # number of base estimators
    cuda=True
)
load(ensemble, save_dir = model_path, logger = None)
# model.load_state_dict(torch.load(model_path))

# Make prediction.

test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)

ensemble.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = ensemble(features)

        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

with open('prediction'+ str(serial) + '.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))


