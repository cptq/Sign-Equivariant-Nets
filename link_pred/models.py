import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import MessagePassing

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))


    def forward(self, x, edge_index):
        return self.encode(x, edge_index)

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

class GCNSignNet(nn.Module):
    def __init__(self, num_eigs, in_channels, hidden_channels, out_channels):
        super().__init__()
        phi_out_dim = hidden_channels // num_eigs
        self.phi = GCN(1, hidden_channels, phi_out_dim)
        self.rho = GCN(num_eigs * phi_out_dim, hidden_channels, out_channels)

    def forward(self, x, edge_index):
        return self.encode(x, edge_index)

    def encode(self, x, edge_index):
        x = x.unsqueeze(-1) # n x k x 1
        x = x.transpose(0,1) # k x n x 1
        x = self.phi(x, edge_index) + self.phi(-x, edge_index)
        x = x.transpose(0,1) # n x k x d
        x = x.reshape(x.shape[0], -1) # n x kd
        x = self.rho(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)


class DecodeOnly(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x, edge_index):
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

class LearnDecode(torch.nn.Module):
    def __init__(self, num_eigs, hidden_channels, num_layers=2):
        super().__init__()
        modules = []
        modules.append(nn.Linear(num_eigs, hidden_channels))
        modules.append(nn.ReLU())
        for _ in range(num_layers-1):
            modules.append(nn.Linear(hidden_channels, hidden_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_channels,  1))
        self.mlp = nn.Sequential(*modules)

    def encode(self, x, edge_index):
        return x

    def decode(self, z, edge_label_index):
        return self.mlp(z[edge_label_index[0]] * z[edge_label_index[1]]).squeeze()



activation_dict = {'tanh': nn.Tanh(), 'none': nn.Identity(), 'softshrink': nn.Softshrink(lambd=.5), 'sigmoid': nn.Sigmoid(), 'relu': nn.ReLU()}

class BasicMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super(BasicMLP, self).__init__()
        self.lins = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(nn.Linear(in_dim, out_dim))
        else:
            self.lins.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers-2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.lins.append(nn.Linear(hidden_dim, out_dim))
        self.activation = nn.ReLU()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = self.activation(x)
        x = self.lins[-1](x)
        return x


class SignEqGateLayer(nn.Module):
    # R^k to R^k
    def __init__(self, num_eigs, residual=True, identity_init=False, sigmoid=True, num_layers=2):
        super(SignEqGateLayer, self).__init__()
        self.diag = nn.parameter.Parameter(data=torch.ones(1, num_eigs))
        self.mlp = BasicMLP(num_eigs, 92, num_eigs, num_layers)
        self.residual = residual
        self.sigmoid = nn.Sigmoid() if sigmoid else None
        if identity_init:
            with torch.no_grad():
                self.lin.weight.div_(1000)
                self.lin.bias.fill_(1)

    def forward(self, x):
        # x: * x d
        orig_x = x

        x1 = self.diag * x
        x2 = self.mlp(x.abs())
        if self.sigmoid is not None:
            x2 = self.sigmoid(x2)
        x = x1 * x2
        if self.residual:
            x = x + orig_x
        return x

class NoParamConv(MessagePassing):
    ''' Graph aggregation with no parameters '''
    def __init__(self, aggr="mean"):

        super().__init__(aggr)
        self.reset_parameters()

    def reset_parameters(self):
        self.aggr_module.reset_parameters()

    def forward(self, x, edge_index):
        if isinstance(x, torch.Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x)

        return out

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return f'{self.__class__.__name__}(aggr={self.aggr})'


class SignDSSLayer(nn.Module):
    # x_i <-- x_i * MLP(|x_i|) +  aggr_j x_j * MLP(|aggr_j x_j|)
    def __init__(self, num_eigs, activation='none', residual=True, use_conv=True):
        super(SignDSSLayer, self).__init__()
        self.activation = activation_dict[activation]
        self.net1 = SignEqGateLayer(num_eigs, residual=residual)
        self.net2 = SignEqGateLayer(num_eigs, residual=residual)
        self.residual = residual
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = NoParamConv('mean')

    def forward(self, x, edge_index):
        # x is b x n x d
        # batch x points x dimension
        
        # b x n x d -> b x n x d
        orig_x = x
        x1 = self.net1(x)
        if self.use_conv:
            # b x n x d -> b x n x d
            aggr_x = self.conv(x, edge_index)
        else:
            # b x n x d -> b x 1 x d
            aggr_x = x.mean(dim=-2, keepdim=True)
        x2 = self.net2(aggr_x)
        x = x1 + x2
        x = self.activation(x)
        if self.residual:
            x = orig_x + x

        return x

class SignDSS(torch.nn.Module):
    def __init__(self, num_eigs, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SignDSSLayer(num_eigs))
        for _ in range(num_layers-2):
            self.convs.append(SignDSSLayer(num_eigs))
        self.convs.append(SignDSSLayer(num_eigs))


    def forward(self, x, edge_index):
        return self.encode(x, edge_index)

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index) 
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

