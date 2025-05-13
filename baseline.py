import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import random
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
from sklearn.metrics import roc_curve, auc
from util import load_data
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
import matplotlib.cm as cm

from pyod.models.lof import LOF
from pyod.models.deep_svdd  import DeepSVDD
from pyod.models.abod  import ABOD
from pyod.models.mo_gaal  import MO_GAAL
from pyod.models.so_gaal  import SO_GAAL
from pyod.models.auto_encoder  import AutoEncoder
from pyod.models.dif  import DIF
from pyod.models.ecod  import ECOD
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


import time
from sklearn.preprocessing import LabelEncoder

import umap


random.seed(1234) 

np.set_printoptions(threshold=np.inf)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(123)




import pandas as pd

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
    data = Data(x=x, y=y).to(device)
    dataset.append(data)

random.shuffle(dataset)

num_ming = len(ming_feature)
# print(num_ming)

for i in range (num_ming):
    x = torch.from_numpy(ming_feature[i,:]).to(torch.float).unsqueeze(0)
    y = torch.tensor([1], dtype=torch.long)
    data = Data(x=x, y=y).to(device)
    dataset.append(data)

train_dataset = dataset[:285]
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataset = dataset[285:]
test_loader = DataLoader(test_dataset, batch_size=1)





method = 'if'
# method = 'lof'
# method = 'ocsvm'
# method = 'ochnn'
# method = 'deepsvdd'
# method = 'abod'
# method = 'mo_gaal'
# method = 'so_gaal'
# method = 'auto_encoder'
# method = 'dif'
# method = 'ecod'









if method == 'ochnn':
    model = OCHNN()
elif method == 'if':
    train_epoch = 500
    model = IsolationForest()
elif method == 'lof':
    train_epoch = 500
    model = LOF()
elif method == 'ocsvm':
    train_epoch = 500
    model = OneClassSVM(kernel='rbf', gamma='auto')
elif method == 'deepsvdd':
    train_epoch = 900
    model = DeepSVDD(in_feats, use_ae=True, hidden_neurons=[32, 32, 32, 8], epochs=train_epoch, batch_size=100, dropout_rate=0.)
elif method == 'ae1svm':
    train_epoch = 1500
    model = AE1SVM(hidden_neurons=[32, 32, 32, 8], batch_norm=False, learning_rate=8e-4, epochs=train_epoch, batch_size=100, dropout_rate=0.0, weight_decay=0)
elif method == 'abod':
    model = ABOD(contamination=0.5)
elif method == 'mo_gaal':
    train_epoch = 100
    model = MO_GAAL(k=3, stop_epochs=train_epoch, lr_d=0.0005, lr_g=0.00001, momentum=0.9, contamination=0.5)
elif method == 'so_gaal':
    train_epoch = 100
    model = SO_GAAL(stop_epochs=train_epoch, lr_d=0.0001, lr_g=0.00001, momentum=0.9, contamination=0.5)
elif method == 'auto_encoder':
    train_epoch = 400
    model = AutoEncoder(contamination=0.5, preprocessing=True, learning_rate=0.0001, epochs=train_epoch, batch_size=100, hidden_neurons=[32, 32, 32, 8], batch_norm=False, dropout_rate=0.)
elif method == 'dif':
    model = DIF(device=device)
elif method == 'ecod':
    model = ECOD(contamination=0.5)
        


if method == 'ochnn':
    # for data in train_loader:
    model.fit(train_loader, in_feats)
elif method == 'if':
    for cur_epoch in range(train_epoch):
        for data in train_loader:
            model.fit(data.x.detach().cpu().numpy())
elif method == 'lof':
    for cur_epoch in range(train_epoch):
        for data in train_loader:
            model.fit(data.x.detach().cpu().numpy())
elif method == 'ocsvm':
    for cur_epoch in range(train_epoch):
        for data in train_loader:
            model.fit(data.x.detach().cpu().numpy())
elif method == 'deepsvdd':
    arr_x = []
    for data in train_loader:
        arr_x_i = data.x.detach()
        arr_x.append(arr_x_i)
    arr_x = torch.cat(arr_x, dim=0)
    model.fit(arr_x.detach().cpu().numpy())
elif method == 'abod':
    arr_x = []
    for data in train_loader:
        arr_x_i = data.x.detach()
        arr_x.append(arr_x_i)
    arr_x = torch.cat(arr_x, dim=0)
    model.fit(arr_x.detach().cpu().numpy())
elif method == 'mo_gaal':
    arr_x = []
    for data in train_loader:
        arr_x_i = data.x.detach()
        arr_x.append(arr_x_i)
    arr_x = torch.cat(arr_x, dim=0)
    model.fit(arr_x.detach().cpu().numpy())
elif method == 'so_gaal':
    arr_x = []
    for data in train_loader:
        arr_x_i = data.x.detach()
        arr_x.append(arr_x_i)
    arr_x = torch.cat(arr_x, dim=0)
    model.fit(arr_x.detach().cpu().numpy())
elif method == 'auto_encoder':
    arr_x = []
    for data in train_loader:
        arr_x_i = data.x.detach()
        arr_x.append(arr_x_i)
    arr_x = torch.cat(arr_x, dim=0)
    model.fit(arr_x.detach().cpu().numpy())
elif method == 'ecod':
    arr_x = []
    for data in train_loader:
        arr_x_i = data.x.detach()
        arr_x.append(arr_x_i)
    arr_x = torch.cat(arr_x, dim=0)
    model.fit(arr_x.detach().cpu().numpy())
elif method == 'dif':
    arr_x = []
    arr_y = []
    for data in train_loader:
        arr_x_i = data.x.detach()
        arr_x.append(arr_x_i)
        arr_y_i = data.y.detach()
        arr_y.append(arr_y_i)
    arr_x = torch.cat(arr_x, dim=0)
    arr_y = torch.cat(arr_y, dim=0)
    model.fit(arr_x.detach().cpu().numpy(), arr_y.detach().cpu().numpy())
    

t = 0
score_pred = np.zeros(len(test_dataset))
test_label = np.zeros(len(test_dataset))
output_h = torch.zeros((len(test_dataset), 8))
colors_label = np.zeros(len(test_dataset), int)
execution_time = 0

for data in test_loader:  # Iterate in batches over the training/test dataset.
    if data.y == 1:
        y_label = 1
    else:
        y_label = 0

    start_time = time.time()

    
    if method == 'ochnn':
        scores, output = model.decision_function(data)
        print('scores')
        print(scores)
        score_pred[t] = scores
        test_label[t] = y_label
        output_h[t, :] = output
        colors_label[t] = y_label
    elif method == 'if':
        scores = model.decision_function(data.x.detach().cpu().numpy())
        print('scores')
        print(scores)
        score_pred[t] = scores
        test_label[t] = y_label
    elif method == 'lof':
        scores = model.decision_function(data.x.detach().cpu().numpy())
        print('scores')
        print(scores)
        score_pred[t] = scores
        test_label[t] = y_label
    elif method == 'ocsvm':
        scores = model.predict(data.x.detach().cpu().numpy())
        print('scores')
        print(scores)
        score_pred[t] = scores
        test_label[t] = y_label
    elif method == 'deepsvdd':
        scores = model.decision_function(data.x.detach().cpu().numpy())
        print('scores')
        print(scores)
        score_pred[t] = scores
        test_label[t] = y_label
    elif method == 'ae1svm':
        scores = model.decision_function(data.x.detach().cpu().numpy())
        print('scores')
        print(scores)
        score_pred[t] = scores
        test_label[t] = y_label
    elif method == 'abod':
        scores = model.decision_function(data.x.detach().cpu().numpy())
        print('scores')
        print(scores)
        score_pred[t] = scores
        test_label[t] = y_label
    elif method == 'mo_gaal':
        scores = model.decision_function(data.x.detach().cpu().numpy())
        print('scores')
        print(scores)
        score_pred[t] = scores
        test_label[t] = y_label
    elif method == 'so_gaal':
        scores = model.decision_function(data.x.detach().cpu().numpy())
        print('scores')
        print(scores)
        score_pred[t] = scores
        test_label[t] = y_label
    elif method == 'auto_encoder':
        scores = model.decision_function(data.x.detach().cpu().numpy())
        print('scores')
        print(scores)
        score_pred[t] = scores
        test_label[t] = y_label
    elif method == 'dif':
        scores = model.decision_function(data.x.detach().cpu().numpy())
        print('scores')
        print(scores)
        score_pred[t] = scores
        test_label[t] = y_label
    elif method == 'ecod':
        scores = model.decision_function(data.x.detach().cpu().numpy())
        print('scores')
        print(scores)
        score_pred[t] = scores
        test_label[t] = y_label

    end_time = time.time()
    execution_time = execution_time + (end_time - start_time)

    t += 1



avg_execution_time = execution_time/t
print(f'AVG run time: {avg_execution_time:.8f}')


roc_auc = roc_auc_score(test_label, score_pred)
ap = average_precision_score(test_label, score_pred)

fpr, tpr, thresholds = roc_curve(test_label, score_pred)
distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
best_threshold = thresholds[np.argmin(distances)]
roc_auc = auc(fpr, tpr)

print(fpr)
print(tpr)
print(thresholds)
print(best_threshold)

print(f'AUC: {roc_auc:.6f}, AP: {ap:.6f}')




