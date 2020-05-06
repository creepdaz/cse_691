import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from torch.autograd import Variable
import torch
import scipy
import numpy as np
import scipy.sparse
import sklearn.metrics
import sys
import os

from src.coarsening import *
from src.graphCNN import *


grid_side = 52
number_edges = 8
metric = 'euclidean'
A = grid_graph(grid_side,number_edges,metric) # create graph of Euclidean grid

# Compute coarsened graphs
coarsening_levels = 1
L, perm = coarsen(A, coarsening_levels)
print(L)

class my_sparse_mm(torch.autograd.Function):
    """
    Implementation of a new autograd function for sparse variables, 
    called "my_sparse_mm", by subclassing torch.autograd.Function 
    and implementing the forward and backward passes.
    """
    @staticmethod
    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y
    
    @staticmethod
    def backward(self, grad_output):
        W, x = self.saved_tensors 
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t()) 
        grad_input_dL_dx = torch.mm(W.t(), grad_input )
        return grad_input_dL_dW, grad_input_dL_dx





# define the GCN layer using to extract the features in the graph nerual network
#
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)



'''
input shape is [12, 32, 44, 44, 44]
'''
class gcn_v(nn.Module):
    def __init__(self, net_paras):
        super(gcn_v,self).__init__()
        D, CL1_k, CL1_f,out_feature = net_paras
        # first still need conv3d 
        self.conv1 = nn.Conv3d(1,1,7)
        self.pool1 = nn.MaxUnpool3d(2)
        #input feature, output feature 
        self.cl1 = nn.Linear(CL1_k,CL1_f)
        Fin =CL1_k
        Fout= CL1_f
        # define the init weight of self.cl1
        # define the init bias of self.cl1
        self.cl1.weight.data.uniform_(-scale, scale)
        self.cl1.bias.data.fill_(0.0)
        self.CL1_k = CL1_k
        self.CL1_f = CL1_f
        self.bn= nn.BatchNorm1d(out_feature)


    # cl is the linear combination of the fan out 
    def graph_cheby(self,x,cl,bn,L,Fout,K):
        #B: batch
        #V: number of vertices
        #Fin: number of input features
        # Fout:number of output features
        # K: Chebyshev orders and support  size

        B, V, Fin = x.size()
        B, V, Fin = int(B), int(V), int(Fin)

        # transform to Chebyshev basis
        x0 = x.permute(1, 2, 0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin * B])  # V x Fin*B
        x = x0.unsqueeze(0)  

        if torch.cuda.is_available():
            L = L.cuda()
        # concatenate the x and x_
        def concat(x, x_):
            x_ = x_.unsqueeze(0)            
            return torch.cat((x, x_), 0)    
        if K > 1: 
            x1 = Calulate_grad()(L,x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)),0) 
        
        for k in range(2, K):
            x2 = 2 * Calulate_grad()(L,x1) - x0  
            x = torch.cat((x, x2.unsqueeze(0)),0)  # M x Fin*B
            x0, x1 = x1, x2 
        
        x = x.view([K, V, Fin, B])  # K x V x Fin x B
        x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
        x = x.view([B * V, Fin * K])  # B*V x Fin

        x = cl(x)

        if bn is not None:
            x = bn(x)  # B*V x Fout
        x = x.view([B, V, Fout])  # B x V x Fout

        return x  


    def init_weights(self, W, Fin, Fout):

        scale = np.sqrt( 2.0/ (Fin+Fout) )
        W.uniform_(-scale, scale)

        return W

        # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p): 
        if p > 1: 
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p          
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x  
        else:
            return x
    # forward function of the graph network
    def forward(self,x,d,L,lmax):
        return x


    



class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )
    
    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)

    
class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)
    

class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
    

class EncoderDecorder(nn.Module):

    def __init__(self):
        super(EncoderDecorder, self).__init__()

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)

        self.mid_res = Res3DBlock(128, 128)

        self.decoder_res2 = Res3DBlock(128, 128)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlock(64, 64)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2)

        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(64, 64)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)

        x = self.mid_res(x)

        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x


class V2VModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(V2VModel, self).__init__()
       
        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 16, 7),
            Pool3DBlock(2),
            Res3DBlock(16, 32),
            Res3DBlock(32, 32),
            Res3DBlock(32, 32)
        )

        self.encoder_decoder = EncoderDecorder()

        self.back_layers = nn.Sequential(
            Res3DBlock(32, 32),
            Basic3DBlock(32, 32, 1),
            Basic3DBlock(32, 32, 1),
        )

        self.conv1= nn.Conv3d(32,32,1,1)
        self.conv_graph = nn.Conv3d(32,1,1,1)
        self.grap_pool = nn.MaxPool3d(3)
        self.ratio = 0.9
        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)
        self.opt =False
        self._initialize_weights()


    def _merge_x(self, a, b,ratio):
        a= Variable(a, requires_grad=True)
        b= Variable(b, requires_grad=True)
        res = Variable(a*ratio +(1-ratio)*b, requires_grad=True)

        return res

    def forward(self, x):
        net_para = [1,2,3,4,5]
        # initialize the graph output of the current features
        net_para = []
        x_=[]
        
        if self.opt:
            #input size is [12,32,44,44,44]
            x_ = x.clone()
            # [12,1,44,44,44]
            x_ = self.conv_graph(x) 
            x_= self.grap_pool(x_)
            #[12 ,1 ,14,14,14]
            x_ = x_.view(12,14*14*14)




        else:
            x_ =Variable(self.conv1(x) , requires_grad=True)
        
        
        x = self.back_layers(x)
        #x = x*self.ratio + x_*(1-self.ratio)
        x = self.output_layer(x)
        #x = merge_x(x,x,0.9)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
