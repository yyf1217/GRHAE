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
from geoopt.manifolds.stereographic.math import weighted_midpoint, dist, dist0 



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
        self.encoder.append(HNN_layer(self.manifold, n_hidden, out_features, self.c, dropout, act, use_bias))


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
        self.decoder.append(HNN_layer(self.manifold, output_hidden, in_features, self.c, dropout, act, use_bias))

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
    """
    OCGNN (One-Class Graph Neural Networks for Anomaly Detection in
    Attributed Networks) is an anomaly detector that measures the
    distance of anomaly to the centroid, in a similar fashion to the
    support vector machine, but in the embedding space after feeding
    towards several layers of GCN.

    See :cite:`wang2021one` for details.

    Parameters
    ----------
    n_hidden :  int, optional
        Hidden dimension of model. Defaults: `256``.
    n_layers : int, optional
        Dimensions of underlying GCN. Defaults: ``4``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    dropout : float, optional
        Dropout rate. Defaults: ``0.3``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Defaults: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Defaults: ``torch.nn.functional.relu``.
    eps : float, optional
        A small valid number for determining the center and make
        sure it does not collapse to 0. Defaults: ``0.001``.
    nu: float, optional
        Regularization parameter. Defaults: ``0.5`` 
    lr : float, optional
        Learning rate. Defaults: ``0.005``.
    epoch : int, optional
        Maximum number of training epoch. Defaults: ``5``.
    warmup_epoch : int, optional
        Number of epochs to update radius and center in the beginning 
        of training. Defaults: ``2``.
    gpu : int
        GPU Index, -1 for using CPU. Defaults: ``0``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Defaults: ``False``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.

    Examples
    --------
    >>> from pygod.models import AnomalyDAE
    >>> model = OCGNN()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

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
                 beta=0.000, 
                 verbose=True,
                 act=F.relu):
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
        

        print("Hidden Units: {}, Layers: {}, Output Features: {}, Learning Rate: {:.6f}, c=: {:.2f}, Beta: {:.6f}, Warmup Epoch: {}, Train GW Epoch: {}, Dropout: {:.2f}, Weight Decay: {:.4f}".format(
                n_hidden, n_layers, out_features, lr, c, beta, warmup_epoch, train_gw_epoch, dropout, weight_decay), end='')



        # other param
        self.verbose = verbose
        self.model = None
        self.hyp_act = HypAct(self.manifold, self.c, self.c, self.act)


    def loss_function(self, x, x_, update=False):
        """
        Calculate the loss in paper Equation (4)
        
        Parameters
        ----------
        outputs : torch.Tensor
            The output in the reduced space by GCN.
        update : bool, optional (default=False)
            If you need to update the radius, set update=True.

        Returns
        ----------
        dist : torch.Tensor
            Average distance.
        scores : torch.Tensor
            Anomaly scores.
        loss : torch.Tensor
            A combined loss of radius and average scores.
        """

        diff_attribute = torch.pow(x - x_, 2)
        loss = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

        return loss


    def fit(self, loader, in_feats, y_true=None):
        """
        Fit detector with input data.

        Parameters
        ----------
        G : torch_geometric.data.Data
            The input data.
        y_true : numpy.ndarray, optional
            The optional outlier ground truth labels used to monitor
            the training progress. They are not used to optimize the
            unsupervised model. Default: ``None``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
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
                h, outputs = self.model(data.x)
                rec_loss = self.loss_function(outputs, data.x)
                x = data.x.detach()
                diff_matrix = x.unsqueeze(1) - x.unsqueeze(0)
                xy = torch.norm(diff_matrix**2, p=2, dim=2)
                Tx = h
                Txy = self.manifold.pairwise_distance(Tx)
                Axy = 1.0/(1.0+xy)
                ATxy = 1.0/(1.0+Txy)
                GW_distance = ((Axy - ATxy)**2).mean()
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
            save_checkpoint({
                    'epoch': cur_epoch,
                    'state_dict': self.model.state_dict(),
                    }, os.path.join('./train_model/hnn_gram_re/'))

        return self


    def decision_function(self, data):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on distance 
        to the centroid and measurement within the radius
        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        
        Returns
        -------
        anomaly_scores : numpy.array
            The anomaly score of the input samples of shape (n_samples,).
        """

        self.model.eval()

        data.x.requires_grad_(True)
        z_latent, output = self.model(data.x)

        diff_attribute = torch.pow(data.x - output, 2)
        rec_score = torch.sqrt(torch.sum(diff_attribute, 1)).detach().cpu().numpy()
        z_latent_sum = torch.sum(z_latent, dim=0)
        scores = torch.zeros((data.x.shape[0], z_latent.shape[1], data.x.shape[1])).to(self.device)
        mid_point = torch.zeros((data.x.shape[0])).to(self.device)
        outlier_scores = np.zeros(data.x.shape[0])



        for i in range(z_latent.shape[1]):
            z_latent_sum[i].backward(retain_graph=True)
            a = self.model.h_grads.detach()
            scores[:,i,:] = (self.model.x_hyp * a).squeeze(0)


        for j in range(data.x.shape[0]):
            mid_point = weighted_midpoint(scores[j,:,:], k=torch.tensor(self.c, device=self.device))
            outlier_scores[j] = dist0(mid_point, k=torch.tensor(self.c, device=self.device)).detach().squeeze(0).cpu().numpy()
        return outlier_scores, z_latent

def save_checkpoint(state, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    best_file = os.path.join(outdir, 'model.pth')
    torch.save(state, best_file)
