import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import random
# from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
# from util import load_data
from gcvgae_toydata_no_ring_test2 import GCVGAE
from dominant import DOMINANT
from conad import CONAD
from guide import GUIDE
from gcnae import GCNAE
from gaan import GAAN
import random
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
from sklearn.metrics import roc_curve, auc
from util import load_data
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, dense_to_sparse
import networkx as nx
import matplotlib
import matplotlib.cm as cm


random.seed(12345)  # 设置随机种子为123


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(12345)


# import numpy as np
# import networkx as nx
# from torch_geometric.utils import to_networkx


import pandas as pd

# # np.random.seed(123456)

# # 节点特征维度
# feat_dim = 20
# # 最小节点数
# min_nodes = 4
# # 最大节点数
# max_nodes = 20
# # 图数量
# num_graphs = 4000
# t = 0

# graphs = []



# # for i in range(200):
# while True:
#     # 随机生成节点数量
#     num_nodes = np.random.randint(min_nodes, max_nodes + 1)
#     # 生成随机图
#     graph = nx.gnm_random_graph(num_nodes, num_nodes + np.random.randint(0, num_nodes))
    
#     while True:
#         # 确保图包含双环
#         cycles = list(nx.cycle_basis(graph))
#         # 获取所有孤立的节点
#         isolated_nodes = list(nx.isolates(graph))
#         if len(cycles) == 2 and len(isolated_nodes) == 0:
#             # print(cycles)
#             break
#         else:
#             graph = nx.gnm_random_graph(num_nodes, num_nodes + np.random.randint(0, num_nodes))
#     # if len(cycles) ==1 :
#     #         # 
#     #         continue
#     print(cycles)

    

#     # 将边缘索引保存到数组中
#     edge_index = np.array(list(graph.edges)).T
#     # # 生成随机特征
#     # node_features = np.zeros((num_nodes, feat_dim))
#     # for j in range(num_nodes):
#     #     node_features[j, np.random.randint(feat_dim)] = 1
#     node_features = np.zeros((num_nodes, max_nodes))
#     for i in range(num_nodes):
#         neighbors = list(graph.neighbors(i))
#         node_features[i, neighbors] = 1
#     # 将图数据保存到字典中
#     graph_dict = {'num_nodes': num_nodes, 'edge_mat': edge_index, 'node_features': node_features, 'y': 1}
#     graphs.append(graph_dict)
#     t = t+1
#     if t==200:
#         break
#     print(t)


# # 最小节点数
# min_nodes = 3

# for i in range(num_graphs):
#     # 随机生成节点数量
#     num_nodes = np.random.randint(min_nodes, max_nodes + 1)
#     # 生成随机图
#     graph = nx.gnm_random_graph(num_nodes, num_nodes + np.random.randint(0, num_nodes))
#     # 确保图包含单环
#     while True:
#         cycles = list(nx.cycle_basis(graph))
#         # 获取所有孤立的节点
#         isolated_nodes = list(nx.isolates(graph))
#         if len(cycles) == 1 and len(isolated_nodes) == 0:
#             print(cycles)
#             break
#         else:
#             graph = nx.gnm_random_graph(num_nodes, num_nodes + np.random.randint(0, num_nodes))


#     # 将边缘索引保存到数组中
#     edge_index = np.array(list(graph.edges)).T
#     # # 生成随机特征
#     # node_features = np.zeros((num_nodes, feat_dim))
#     # for j in range(num_nodes):
#     #     node_features[j, np.random.randint(feat_dim)] = 1
#     node_features = np.zeros((num_nodes, max_nodes))
#     for i in range(num_nodes):
#         neighbors = list(graph.neighbors(i))
#         node_features[i, neighbors] = 1
#     # 将图数据保存到字典中
#     graph_dict = {'num_nodes': num_nodes, 'edge_mat': edge_index, 'node_features': node_features, 'y': 0}
#     graphs.append(graph_dict)
#     t = t+1
#     print(t)



# # # plot graph
# # data = graphs[0]
# # print(data)
# # nx_graph = to_networkx(data)
# # pos = nx.kamada_kawai_layout(nx_graph)
# # node_labels = 0

# # # node_colors = np.array(['red' if label == 0 else 'green' for label in node_labels])
# # norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# # cmap = cm.get_cmap('Accent')
# # plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
# # node_colors = np.array([plt_colors.to_rgba(node_labels[i]) for i in range(len(node_labels))])
# # print(node_colors)

# # nx.draw(nx_graph, pos=pos, node_color=node_colors)
# # plt.axis('off')
# # # plt.savefig('proteins_graph.png', dpi=300)
# # plt.savefig(f'./toy_data_plot/{t}.png', dpi=300)
# # plt.close('all')

# # 保存数据
# np.savez('toy_data.npz', graphs=graphs)
# print('end')


# 指定Excel文件名和工作表名称列表
file_name1 = "./data/青花瓷色度统计LAB.xlsx"
sheet_names = ["元代", "明代"]

num_feature = 17

# 读取Excel文件中的所有工作表，并存储在一个字典中
yuan = pd.read_excel(file_name1, sheet_name=sheet_names)
tai = pd.read_excel(file_name2)
you = pd.read_excel(file_name3)
test = pd.read_excel(file_name4, sheet_name=sheet_names)
train = pd.read_excel(file_name5, sheet_name=sheet_names2)
# print(qinghua)
# print(tai)
# print(you)
# print(test)

df_sorted = qinghua.sort_values(by=['瓷片编号'])
unique_items, unique_index = np.unique(df_sorted['瓷片编号'], return_index=True)
# print(df_sorted)
df_sorted = np.array(df_sorted)
num_qinghua = len(unique_items)
qinghua_features = np.zeros((num_qinghua, num_feature))
for i in range (num_qinghua):
    qinghua_features[i] = df_sorted[unique_index[i], 3:]
# duplicates = df_sorted[df_sorted.duplicated(subset=['瓷片编号'], keep=False)]
# train_qinghua = df_sorted[unique_items]
# print(qinghua_features)


# print(df_sorted.duplicated(subset=['瓷片编号'], keep=False))
df_sorted2 = tai.sort_values(by=['瓷片编号'])
unique_items2, unique_index2 = np.unique(df_sorted2['瓷片编号'], return_index=True)
print(df_sorted2)
df_sorted2 = np.array(df_sorted2)
num_tai = len(unique_items2)
tai_features = np.zeros((num_tai, num_feature))
for i in range (num_tai):
    tai_features[i, :10] = df_sorted2[unique_index2[i], 3:]
# duplicates2 = df_sorted2[df_sorted2.duplicated(subset=['瓷片编号'], keep=False)]
# print(df_sorted2)
print(num_tai)


# common_items = np.intersect1d(unique_items, unique_items2)
# common_items_positions_in_vector1 = np.where(np.in1d(unique_items, common_items))[0]
# common_items_positions_in_vector2 = np.where(np.in1d(unique_items2, common_items))[0]
# print(common_items)
# print(common_items_positions_in_vector1)
# print(common_items_positions_in_vector2)


df_sorted3 = you.sort_values(by=['瓷片编号'])
# duplicates3 = df_sorted3[df_sorted3.duplicated(subset=['瓷片编号'], keep=False)]
unique_items3, unique_index3 = np.unique(df_sorted3['瓷片编号'], return_index=True)
# print(df_sorted3)
df_sorted3 = np.array(df_sorted3)
num_you = len(unique_items3)
you_features = np.zeros((num_you, num_feature))
for i in range (num_you):
    you_features[i, :10] = df_sorted3[unique_index3[i], 3:]
# print(df_sorted3)
# print(you_features)



# common_items2 = np.intersect1d(common_items, unique_items3)
# common_items2_positions_in_vector1 = np.where(np.in1d(common_items, common_items2))[0]
# common_items2_positions_in_vector2 = np.where(np.in1d(unique_items3, common_items2))[0]
# print(common_items2)
# print(common_items2_positions_in_vector1)
# print(common_items2_positions_in_vector2)


common_items = np.intersect1d(unique_items, unique_items2)
common_items = np.intersect1d(common_items, unique_items3)
common_items_positions_in_vector1 = np.where(np.in1d(unique_items, common_items))[0]
common_items_positions_in_vector2 = np.where(np.in1d(unique_items2, common_items))[0]
common_items_positions_in_vector3 = np.where(np.in1d(unique_items3, common_items))[0]
# print(common_items)
# print(common_items_positions_in_vector1)
# print(common_items_positions_in_vector2)
# print(common_items_positions_in_vector3)



num_data = len(common_items)
dataset = []

for i in range (num_data):
    x1 = torch.from_numpy(qinghua_features[common_items_positions_in_vector1[i]]).to(torch.float)
    x2 = torch.from_numpy(tai_features[common_items_positions_in_vector2[i]]).to(torch.float)
    x3 = torch.from_numpy(you_features[common_items_positions_in_vector3[i]]).to(torch.float)
    # x = torch.cat((x1, x2, x3), dim=1)
    x = torch.cat((x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0)), dim=0)
    in_feats = x.shape[1]
    # edge_index = torch.triu(torch.ones(3, 3)).to(torch.long)
    adj = torch.ones((3, 3)) - torch.eye(3)
    # 将邻接矩阵转换为edge_index表示的形式
    edge_index, _ = dense_to_sparse(adj)

    y = torch.tensor([0], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y).to(device)
    dataset.append(data)


train1 = np.array(train["胎"])
num_train = len(train1)
tai_features_train = np.zeros((num_train, num_feature))
tai_features_train[:, :10] = train1[:, 1:]
tai_features_train = [x.astype(float) for x in tai_features_train]
train2 = np.array(train["釉"])
you_features_train = np.zeros((num_train, num_feature))
you_features_train[:, :10] = train2[:, 1:]
you_features_train = [x.astype(float) for x in you_features_train]
train3 = np.array(train["青花"])
qinghua_features_train = np.zeros((num_train, num_feature))
qinghua_features_train = train3[:, 1:]
qinghua_features_train = [x.astype(float) for x in qinghua_features_train]

for i in range (num_train):
    x1 = torch.from_numpy(qinghua_features_train[i]).to(torch.float)
    x2 = torch.from_numpy(tai_features_train[i]).to(torch.float)
    x3 = torch.from_numpy(you_features_train[i]).to(torch.float)
    # x = torch.cat((x1, x2, x3), dim=1)
    x = torch.cat((x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0)), dim=0)
    in_feats = x.shape[1]
    adj = torch.ones((3, 3)) - torch.eye(3)
    # 将邻接矩阵转换为edge_index表示的形式
    edge_index, _ = dense_to_sparse(adj)
    y = torch.tensor([0], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y).to(device)
    dataset.append(data)


test1 = np.array(test["胎"])
num_test = len(test1)
tai_features_test = np.zeros((num_test, num_feature))
tai_features_test[:, :10] = test1[:, 2:]
tai_features_test = [x.astype(float) for x in tai_features_test]
test2 = np.array(test["釉"])
you_features_test = np.zeros((num_test, num_feature))
you_features_test[:, :10] = test2[:, 2:]
you_features_test = [x.astype(float) for x in you_features_test]
test3 = np.array(test["青花彩"])
qinghua_features_test = np.zeros((num_test, num_feature))
qinghua_features_test = test3[:, 2:]
qinghua_features_test = [x.astype(float) for x in qinghua_features_test]
# print(qinghua_features_test)
# print(tai_features_test)
# print(you_features_test)


for i in range (num_test):
    x1 = torch.from_numpy(qinghua_features_test[i]).to(torch.float)
    x2 = torch.from_numpy(tai_features_test[i]).to(torch.float)
    x3 = torch.from_numpy(you_features_test[i]).to(torch.float)
    # x = torch.cat((x1, x2, x3), dim=1)
    x = torch.cat((x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0)), dim=0)
    in_feats = x.shape[1]
    adj = torch.ones((3, 3)) - torch.eye(3)
    # 将邻接矩阵转换为edge_index表示的形式
    edge_index, _ = dense_to_sparse(adj)
    y = torch.tensor([1], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y).to(device)
    dataset.append(data)



# # Observe the distribution of the data
# for i in range(len(dataset)):
#     print(i)
#     print(dataset[i]['y'])
# print(len(dataset))
# print(debug)
# print(len(dataset))
# if isinstance(dataset, list):
#     print("dataset is a list")
# else:
#     print("dataset is not a list")


train_dataset = dataset[:208]


train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# test_dataset = dataset[70:77] + dataset[81:88]
test_dataset = dataset[208:]


test_loader = DataLoader(test_dataset, batch_size=1)


def map_to_minus_one_to_one(x):
    return (2 * ((x - np.min(x)) / (np.max(x) - np.min(x))) - 1)



method = 'grad_cam_vgae'
# method = 'dominant'
# method = 'conad'
# method = 'guide'
# method = 'gcnae'
# method = 'gaan'


if method == 'grad_cam_vgae':
    model = GCVGAE()
elif method == 'dominant':
    model = DOMINANT()
elif method == 'conad':
    model = CONAD()
elif method == 'guide':
    model = GUIDE()
elif method == 'gcnae':
    model = GCNAE()
elif method == 'gaan':
    model = GAAN()



if method == 'grad_cam_vgae':
    model.fit(train_loader, in_feats)
elif method == 'dominant':
    model.fit(train_loader, in_feats)
elif method == 'conad':
    model.fit(train_loader, in_feats)
elif method == 'guide':
    model.fit(train_loader, in_feats, adj_feats)
elif method == 'gcnae':
    model.fit(train_loader, in_feats)
elif method == 'gaan':
    model.fit(train_loader, in_feats)

t = 0
score_pred = np.zeros(len(test_dataset))
graph_label = np.zeros(len(test_dataset))

for data in test_loader:  # Iterate in batches over the training/test dataset.
    # x = data.x.detach().cpu().numpy()
    # x = data.x.detach().cpu().numpy()
    # x = np.ones(data.num_nodes,1)
    # print(data.y)
    if data.y == 0:
        y_label = 1
    else:
        y_label = 0

    
    if method == 'grad_cam_vgae':
        # data = data.to(device)
        scores = model.gradcam(data)
        print('scores')
        print(scores)
        score_pred[t] = sum(scores)
        graph_label[t] = y_label
    elif method == 'dominant':
        # data = data.to(device)
        scores = model.decision_function(data)
        print('scores')
        print(scores)
        score_pred[t] = sum(scores)
        graph_label[t] = y_label
    elif method == 'conad':
        # data = data.to(device)
        scores = model.decision_function(data)
        print('scores')
        print(scores)
        score_pred[t] = sum(scores)
        graph_label[t] = y_label
    elif method == 'guide':
        # data = data.to(device)
        scores = model.decision_function(data)
        print('scores')
        print(scores)
        score_pred[t] = sum(scores)
        graph_label[t] = y_label
    elif method == 'gcnae':
        # data = data.to(device)
        scores = model.decision_function(data)
        print('scores')
        print(scores)
        score_pred[t] = sum(scores)
        graph_label[t] = y_label
    elif method == 'gaan':
        # data = data.to(device)
        scores = model.decision_function(data)
        print('scores')
        print(scores)
        score_pred[t] = sum(scores)
        graph_label[t] = y_label
    t += 1



graph_roc_auc = roc_auc_score(graph_label, score_pred)
graph_ap = average_precision_score(graph_label, score_pred)

fpr, tpr, thresholds = roc_curve(graph_label, score_pred)

# 计算欧几里得距离
distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)

# 找到最小距离对应的阈值
best_threshold = thresholds[np.argmin(distances)]

roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(f'Roc.png', dpi=300)
plt.close('all')
# plt.show()

print(fpr)
print(tpr)
print(thresholds)
print(best_threshold)

print(f'AUC graph: {graph_roc_auc:.6f}, AP graph: {graph_ap:.6f}')


t = 0

for data in test_loader:  # Iterate in batches over the training/test dataset.
    # x = data.x.detach().cpu().numpy()
    # x = data.x.detach().cpu().numpy()
    # x = np.ones(data.num_nodes,1)
    # print(data.y)
    if data.y == 0:
        y_label = 1
    else:
        y_label = 0

    
    if method == 'grad_cam_vgae':
        # data = data.to(device)
        scores = model.gradcam(data)
        if sum(scores) >= best_threshold:
            pre = 0
        else:
            pre = 1
        if pre == y_label:
            print(t)
            # # plot graph
            # nx_graph = to_networkx(data).to_undirected()
            # # print(nx_graph)
            # # 获取图中的所有环
            # cycles = nx.cycle_basis(nx_graph)
            # # print(cycles)
            # # pos = nx.planar_layout(nx_graph)
            # # pos = nx.spring_layout(nx_graph)
            # pos = nx.circular_layout(nx_graph)

            # node_labels = map_to_minus_one_to_one(np.array(scores))
            # # node_labels = np.where(scores >= (best_threshold/data.x.shape[0]), 1, 0)
            
            # # node_colors = np.array(['red' if label == 0 else 'green' for label in node_labels])
            # norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
            # cmap = cm.get_cmap('Accent')
            # plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
            # node_colors = np.array([plt_colors.to_rgba(node_labels[i]) for i in range(len(node_labels))])
            # print(node_colors)

            # nx.draw(nx_graph, pos=pos, node_color=node_colors, arrows=False)

            # # 绘制环中的边
            # c = 0
            # color_c = ['red', 'blue']
            # for cycle in cycles:
            #     cycle_edges = [(cycle[i], cycle[i+1]) for i in range(len(cycle)-1)] + [(cycle[-1], cycle[0])]
            #     nx.draw_networkx_edges(nx_graph, pos, edgelist=cycle_edges, edge_color=color_c[c], width=2)
            #     c = c+1

            # plt.axis('off')
            # # plt.savefig('proteins_graph.png', dpi=300)
            # plt.savefig(f'./test_data/toy_data_no_ring_plot3/{t}.png', dpi=300)
            # plt.close('all')
        
    t += 1





