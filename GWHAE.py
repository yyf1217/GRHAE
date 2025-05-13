import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import random
from GWHAE_hnn import HNN_gram

import random
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, dense_to_sparse
import networkx as nx
import matplotlib
import matplotlib.cm as cm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import os
import pickle
import pandas as pd

np.random.seed(1234)
np.set_printoptions(threshold=np.inf)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(12393)
torch.manual_seed(123)


file_name1 = "./data/青花瓷色度统计LAB.xlsx"
sheet_names = ["元代", "明代"]

data = pd.read_excel(file_name1, sheet_name=sheet_names)
yuan = data["元代"]
ming = data["明代"]

le = LabelEncoder()
yuan['青花呈色编码'] = le.fit_transform(yuan['青花呈色'])
yuan = pd.get_dummies(yuan, columns=['器型'], prefix='器型')

ming['青花呈色编码'] = le.fit_transform(ming['青花呈色'])
ming = pd.get_dummies(ming, columns=['器型'], prefix='器型')

for column in yuan.columns:
    if column not in ming.columns:
        ming[column] = 0  
ming = ming[yuan.columns]

yuan_features = np.array(yuan)[1:,2:]
yuan_feature = np.array(yuan_features, dtype=np.float32)
ming_features = np.array(ming)[1:,2:]
ming_feature = np.array(ming_features, dtype=np.float32)


num_yuan = len(yuan_feature)
in_feats = yuan_feature.shape[1]
dataset = []

for i in range (num_yuan):
    x = torch.from_numpy(yuan_feature[i,:]).to(torch.float).unsqueeze(0)
    y = torch.tensor([0], dtype=torch.long)
    z = torch.argmax(x[0, 7:])
    data = Data(x=x, y=y, z=z).to(device)
    dataset.append(data)

random.shuffle(dataset)
num_ming = len(ming_feature)

for i in range (num_ming):
    x = torch.from_numpy(ming_feature[i,:]).to(torch.float).unsqueeze(0)
    y = torch.tensor([1], dtype=torch.long)
    z = torch.argmax(x[0, 7:])
    data = Data(x=x, y=y, z=z).to(device)
    dataset.append(data)

train_dataset = dataset[:285]
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataset = dataset[285:]
in_feats = train_dataset[0].x.shape[1]
max_count_label = train_dataset[0].y
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100)

method = 'hnn_gram'
model = HNN_gram()
model.fit(train_loader, in_feats)


t = 0
score_pred = np.zeros(len(test_dataset))
test_label = np.zeros(len(test_dataset))
output_h = torch.zeros((len(train_dataset)+len(test_dataset), 8))
colors_label = np.zeros(len(train_dataset)+len(test_dataset), int)

for data in test_loader: 
    batch_len = data.x.shape[0]
    y_label = (data.y != max_count_label).int().detach().cpu().numpy()
    scores, output = model.decision_function(data)
    print('scores')
    print(scores)
    score_pred[t:t + batch_len] = scores
    test_label[t:t + batch_len] = y_label
    output_h[t:t + batch_len, :] = output
    colors_label[t:t + batch_len] = y_label

    t += batch_len


print(test_label)
print(score_pred)

test_roc_auc = roc_auc_score(test_label, score_pred)
test_ap = average_precision_score(test_label, score_pred)
fpr, tpr, thresholds = roc_curve(test_label, score_pred)
distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
best_threshold = thresholds[np.argmin(distances)]
roc_auc = auc(fpr, tpr)

print(fpr)
print(tpr)
print(thresholds)
print(best_threshold)

print(f'AUC graph: {test_roc_auc:.6f}, AP graph: {test_ap:.6f}')