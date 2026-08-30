import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.utils.validation import check_is_fitted
from torch_sparse import SparseTensor
import optimizers

from base import BaseDetector
from utility import validate_device
from metrics import eval_roc_auc
import os

from torch.nn.modules.module import Module
import torch.nn.init as init
import math
from optimizers.radam import RiemannianAdam
import manifolds
from geoopt.manifolds.stereographic.math import dist0, project, weighted_midpoint



class HNN_layer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNN_layer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class HNN_base(nn.Module):
    """
    Hyperbolic neural networks.
    """

    def __init__(self, manifold, in_features, n_hidden, n_layers, out_features, c, dropout, act, use_bias):
        super(HNN_base, self).__init__()
        self.manifold = manifold
        self.c = c
        self.encoder = nn.ModuleList()

        ## encoder
        if n_hidden*(2**(n_layers)) > 1024:
            input_hidden = 1024
        else:
            input_hidden = n_hidden*(2**(n_layers))
        # input layer
        self.encoder.append(HNN_layer(self.manifold, in_features, input_hidden, self.c, dropout, act, use_bias))
        # hidden layers
        for i in range(n_layers):
            if n_hidden*(2**(n_layers-i)) > 1024:
                input_hidden = 1024
            else:
                input_hidden = n_hidden*(2**(n_layers-i))
            if n_hidden*(2**(n_layers-i-1)) > 1024:
                output_hidden = 1024
            else:
                output_hidden = n_hidden*(2**(n_layers-i-1))
            self.encoder.append(HNN_layer(self.manifold, input_hidden, output_hidden, self.c, dropout, act, use_bias))
        # output layer
        self.encoder.append(HypLinear(self.manifold, n_hidden, out_features, self.c, dropout, use_bias))


        ## decoder
        self.decoder = nn.ModuleList()
        self.decoder.append(HNN_layer(self.manifold, out_features, n_hidden, self.c, dropout, act, use_bias))

        # hidden layers
        for i in range(n_layers):
            if n_hidden*(2**i) > 1024:
                input_hidden = 1024
            else:
                input_hidden = n_hidden*(2**i)
            if n_hidden*(2**(i+1)) > 1024:
                output_hidden = 1024
            else:
                output_hidden = n_hidden*(2**(i+1))
            self.decoder.append(HNN_layer(self.manifold, input_hidden, output_hidden, self.c, dropout, act, use_bias))
        self.decoder.append(HypLinear(self.manifold, output_hidden, in_features, self.c, dropout, use_bias))

    def activations_hook(self, grad):
        self.h_grads = grad

    def forward(self, x):
        with torch.enable_grad():
            self.x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
            self.x_hyp.requires_grad_(True)
        self.x_hyp.register_hook(self.activations_hook)
        h = self.x_hyp
        for i, layer in enumerate(self.encoder[0:-1]):
            h = layer(h)
        h = self.encoder[-1](h)
        z_hyp = h
        for layer in self.decoder:
            z_hyp = layer(z_hyp)
        z = self.manifold.proj_tan0(self.manifold.logmap0(self.manifold.proj(z_hyp, c=self.c), c=self.c), c=self.c)
        

        return h, z


class HNN_gram(BaseDetector):
    def __init__(self,
                 n_hidden=32,
                 n_layers=4,
                 out_features=8,
                 contamination=0.5,
                 dropout=0.2,
                 lr=8e-4,
                 weight_decay=0.0000,
                 eps=0.001,
                 nu=0.5,
                 gpu=0,
                 epoch=300,
                 warmup_epoch=2,
                 train_gw_epoch=5,
                 manifold='PoincareBall',
                 c=0.2,
                 use_bias=1,
                 beta=0.8,
                 verbose=True,
                 act=F.relu,
                 checkpoint_path='./train_model/hnn_gram_git/model.pth'):
        super(HNN_gram, self).__init__(contamination=contamination)
        self.dropout = dropout
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.out_feats = out_features
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.nu = nu
        self.data_center = 0
        self.radius = 0.0
        self.epoch = epoch
        self.warmup_epoch = warmup_epoch
        self.train_gw_epoch = train_gw_epoch
        self.act = act
        self.device = validate_device(gpu)
        self.manifold = getattr(manifolds, manifold)()
        self.c = c
        self.use_bias = use_bias
        self.beta = beta
        self.checkpoint_path = checkpoint_path
        

        print("Hidden Units: {}, Layers: {}, Output Features: {}, Learning Rate: {:.6f}, c=: {:.2f}, Beta: {:.6f}, Warmup Epoch: {}, Train GW Epoch: {}, Dropout: {:.2f}, Weight Decay: {:.4f}".format(
                n_hidden, n_layers, out_features, lr, c, beta, warmup_epoch, train_gw_epoch, dropout, weight_decay), end='')



        # other param
        self.verbose = verbose
        self.model = None
        self.hyp_act = HypAct(self.manifold, self.c, self.c, self.act)


    def loss_function(self, x, x_, update=False):
        diff_attribute = torch.pow(x - x_, 2)
        loss = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

        return loss


    def fit(self, loader, in_feats, y_true=None):
        self.in_feats = in_feats

        # initialize the model and optimizer
        self.model = HNN_base(self.manifold,
                              self.in_feats,
                              self.n_hidden,
                              self.n_layers,
                              self.out_feats,
                              self.c,
                              self.dropout,
                              self.act,
                              self.use_bias)


        self.optimizer = RiemannianAdam(params=self.model.parameters(), lr=self.lr,
                                                    weight_decay=self.weight_decay)



        self.model = self.model.to(self.device)
        # training the model
        self.model.train()
        
        for cur_epoch in range(self.epoch):
            epoch_loss = 0
            t = 0

            for data in loader:
                data = data.to(self.device)
                h, outputs = self.model(data.x)
                rec_loss = self.loss_function(outputs, data.x)
                x = data.x.detach()

                x_distance_squared = torch.cdist(x, x, p=2).pow(2)
                h_distance_squared = self.manifold.pairwise_distance(h, c=self.c).pow(2)
                # Axy = x_distance_squared
                # ATxy = h_distance_squared
                Axy = 1.0 / (1.0 + x_distance_squared)
                ATxy = 1.0 / (1.0 + h_distance_squared)
                # Axy = torch.log(1.0 + x_distance_squared)
                # ATxy = torch.log(1.0 + h_distance_squared)
                if x.size(0) > 1:
                    off_diagonal = ~torch.eye(
                        x.size(0), dtype=torch.bool, device=x.device
                    )
                    GW_distance = ((Axy - ATxy) ** 2)[off_diagonal].mean()
                else:
                    GW_distance = x.new_tensor(0.0)
                loss = rec_loss + self.beta * GW_distance

                epoch_loss += loss.item()
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                t = t + 1


            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(cur_epoch,  epoch_loss / t), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_scores)
                    print(" | AUC {:.4f}".format(auc), end='')
                print()
        if self.checkpoint_path is not None:
            save_checkpoint({
                    'epoch': self.epoch - 1,
                    'state_dict': self.model.state_dict(),
                    'curvature': self.c,
                    'beta': self.beta,
                    }, self.checkpoint_path)

        return self


    def decision_function(self, data):

        self.model.eval()

        data = data.to(self.device)
        data.x.requires_grad_(True)
        z_latent, output = self.model(data.x)

        diff_attribute = torch.pow(data.x - output, 2)
        rec_score = torch.sqrt(torch.sum(diff_attribute, 1)).detach().cpu().numpy()
        z_latent_sum = torch.sum(z_latent, dim=0)
        scores = torch.zeros(
            (data.x.shape[0], z_latent.shape[1], data.x.shape[1]),
            dtype=self.model.x_hyp.dtype,
            device=self.device,
        )
        outlier_scores = np.zeros(data.x.shape[0])



        for i in range(z_latent.shape[1]):
            self.model.zero_grad(set_to_none=True)
            if data.x.grad is not None:
                data.x.grad.zero_()
            z_latent_sum[i].backward(retain_graph=True)
            a = self.model.h_grads.detach()
            scores[:, i, :] = self.model.x_hyp * a

        k = scores.new_tensor(-self.c)
        projected_scores = project(scores, k=k, dim=-1)
        self.feature_contributions_ = projected_scores.detach().cpu()
        for j in range(data.x.shape[0]):
            mid_point = weighted_midpoint(projected_scores[j, :, :], k=k)
            outlier_scores[j] = (
                dist0(mid_point, k=k).detach().squeeze().cpu().item()
            )
        return outlier_scores, z_latent.detach()

def save_checkpoint(state, path):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    torch.save(state, path)
