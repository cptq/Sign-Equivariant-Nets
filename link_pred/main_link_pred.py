import os.path as osp
import time
import argparse

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import networkx as nx
import numpy as np
import scipy.sparse.linalg

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx, negative_sampling, to_undirected, to_networkx, get_laplacian, to_scipy_sparse_matrix
from torch_geometric.nn.conv import MessagePassing

from models import GCN, GCNSignNet, DecodeOnly, LearnDecode, SignDSS

#MODEL_NAME = 'gcn'
#MODEL_NAME = 'signnet'
#MODEL_NAME = 'decode_only'
#MODEL_NAME = 'learn_decode'
MODEL_NAME = 'sign_equiv'
USE_EIGS = True
NUM_EIGS = 16
NUM_CLUSTERS = 2
NUM_PARAMS = -1
LR = .01
NUM_EPOCHS = 100
NUM_TRIALS = 10
GRAPH_TYPE = 'pa'

in_dim = NUM_EIGS if USE_EIGS else 1

def main():

    def get_synthetic_data(USE_EIGS, NUM_EIGS, NUM_CLUSTERS, graph_type='er'):
        n = 1000
        num_nodes = n*NUM_CLUSTERS
        added_edges = 1000

        if graph_type == 'er':
            H = nx.erdos_renyi_graph(n, .05)
        elif graph_type == 'pa':
            H = nx.barabasi_albert_graph(n, 20)
        else:
            raise ValueError('Invalid graph type')

        G = nx.disjoint_union_all([H for _ in range(NUM_CLUSTERS)])
        for _ in range(added_edges):
            i = np.random.randint(0, num_nodes)
            j = np.random.randint(0, num_nodes)
            G.add_edge(i,j)
        full_data = from_networkx(G)
        full_data.edge_index = to_undirected(full_data.edge_index)
        full_data.x = torch.ones(num_nodes, 1)

        train_prop = .8
        val_prop = .1
        test_prop = .1

        linksplitter = T.RandomLinkSplit(num_val=val_prop,
                                         num_test=test_prop,
                                         is_undirected=True,
                                         add_negative_train_samples=False)
        train_data, val_data, test_data = linksplitter(full_data)

        # adding Laplacian eigenvectors
        if USE_EIGS:
            edge_index, edge_weight = get_laplacian(train_data.edge_index, train_data.edge_weight, normalization='sym', num_nodes=train_data.num_nodes)
            L = to_scipy_sparse_matrix(edge_index, edge_weight, train_data.num_nodes)
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(L, k=NUM_EIGS+1, sigma=1e-8, which='LM', return_eigenvectors=True)
            eigvecs = torch.from_numpy(eigvecs[:, 1:])
            train_data.x = eigvecs

            val_data.x = train_data.x
            test_data.x = train_data.x

        return train_data, val_data, test_data

    train_data, val_data, test_data = get_synthetic_data(USE_EIGS, NUM_EIGS, NUM_CLUSTERS, graph_type=GRAPH_TYPE)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if MODEL_NAME == 'gcn':
        model = GCN(1, 128, 64, num_layers=3).to(device)
    elif MODEL_NAME == 'signnet':
        model = GCNSignNet(NUM_EIGS, NUM_EIGS, 128, 64).to(device)
    elif MODEL_NAME == 'decode_only':
        model = DecodeOnly()

        start_time = time.time()
        out = model.decode(test_data.x, test_data.edge_label_index).view(-1).sigmoid()
        test_auc = roc_auc_score(test_data.edge_label.cpu().numpy(), out.cpu().numpy())
        print(f'Decode Only Test: {test_auc:.4f}')
        elapsed = time.time() - start_time
        return test_auc, elapsed
    elif MODEL_NAME == 'learn_decode':
        model = LearnDecode(NUM_EIGS, 148).to(device)
    elif MODEL_NAME == 'sign_equiv':
        model = SignDSS(NUM_EIGS, num_layers=4).to(device)
    else:
        raise ValueError('Invalid model name')

    print(model)

    global NUM_PARAMS
    NUM_PARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Num Parameters:', NUM_PARAMS)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    def train(data):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            data.edge_label,
            data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        return loss


    @torch.no_grad()
    def test(data):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


    best_val_auc = final_test_auc = 0
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS+1):
        loss_lst = []
        val_auc_lst = []
        test_auc_lst = []
        
        loss = train(train_data.to(device))
        val_auc = test(val_data.to(device))
        test_auc = test(test_data.to(device))

        loss_lst.append(loss.item())
        val_auc_lst.append(val_auc)
        test_auc_lst.append(test_auc)

        epoch_loss = np.mean(loss_lst)
        epoch_val_auc = np.mean(val_auc_lst)
        epoch_test_auc = np.mean(test_auc_lst)

        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            final_test_auc = epoch_test_auc
        print(f'Epoch: {epoch:03d}, Loss: {epoch_loss:.4f}, Val: {epoch_val_auc:.4f}, '
              f'Test: {epoch_test_auc:.4f}')
    elapsed = time.time() - start_time
    elapsed_per_epoch = elapsed / NUM_EPOCHS

    return final_test_auc, elapsed_per_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sign_equiv')
    parser.add_argument('--use_eigs', type=int, default=0)
    parser.add_argument('--num_eigs', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--graph_type', type=str, default='pa')
    parser.add_argument('--num_clusters', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.01)


    test_auc_lst = []
    time_lst = []
    for trial in range(args.num_trials):
        print()
        print('Trial:', trial)
        test_auc, elapsed = main()
        test_auc_lst.append(test_auc)
        time_lst.append(elapsed)
    final_test_auc = np.mean(test_auc_lst)
    test_auc_std = np.std(test_auc_lst)
    time_mean = np.mean(time_lst)
    time_std = np.std(time_lst)

    print('Graph type:', args.graph_type)
    print('Model:', args.model)
    print('Use eigs:', args.use_eigs)
    print('Num Eigs:', args.num_eigs)
    print('Params:', NUM_PARAMS)
    print('lr:', args.lr)
    print('Num clusters:', args.num_clusters)
    print(f'Final Test: {final_test_auc:.4f} +- {test_auc_std:.4f}')
    print(f'Runtime: {time_mean:.4f} +- {time_std:.4f}')

