import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, LeakyReLU, ELU
from torch.nn import ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Set2Set
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import softmax
# import selfies as sf
from tqdm import tqdm
from torch_geometric.nn import GATConv,GCNConv,GINConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch.nn.parameter import Parameter
import math
import numpy as np
import utils.hypergraph_util as hgut
# from .GNN import GNN_graphpred, MLP 
from json.tool import main
from webbrowser import get
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from dataloaders.pocket_utils import pocket_hypergraph
# def protein_sequence()
from tape import ProteinBertModel, TAPETokenizer

# protein_model=ProteinBertModel.from_pretrained('bert-base')
# tokenizer = TAPETokenizer(vocab='iupac')

def graph_poolings(graph_pool,node_representation,batch):
    emb_dim=""

    num_tasks=1  
    #Different kind of graph pooling
    if graph_pool == "sum":
        pool = global_add_pool
    elif graph_pool == "mean":
        pool = global_mean_pool
    elif graph_pool == "max":
        pool = global_max_pool
    elif graph_pool == "attention":
        pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
    elif graph_pool[:-1] == "set2set":
        # set2set_iter = int(graph_pool[-1])
        # pool = Set2Set(emb_dim, set2set_iter)

        pool=Set2Set(
            in_channels=emb_dim, processing_steps=5, num_layers=2)
    else:
        raise ValueError("Invalid graph pooling type.")

    #For graph-level binary classification
    if graph_pool[:-1] == "set2set":
        mult = 2
    else:
        mult = 1

    graph_pred_linear = torch.nn.Linear(mult * emb_dim, num_tasks)

    return graph_pred_linear(pool(node_representation, batch)) 


class Protein_GAT(torch.nn.Module):
    def __init__(self, GAT_config):
        super(Protein_GAT, self).__init__()
        self.embedding_net = JKMCNWMEmbeddingNet(
            num_features=GAT_config['num_features'],
            dim=128,
            train_eps=True,
            num_edge_attr=1,
            num_layers=6,
            num_channels=3
        )

    def forward(self, x, edge_index, edge_attr, batch):
        print("GNNN")
        graph_embedding, _, _ = self.embedding_net(
            x,
            edge_index,
            edge_attr,
            batch
        )

        return graph_embedding



class GNN_encoder(torch.nn.Module):
    # def __init__(self,emb_dim="",drop_ratio = 0, graph_pooling = "mean", gnn_type = "gat", encoder_config=""):
    def __init__(self,encoder_config, device):
        super(GNN_encoder,self).__init__()

        self.device=device
        # self.freeze_encoder()
        
        self.HGNN_train=encoder_config['encoder_Train']['HGNN']

        self.GNN_train=encoder_config['encoder_Train']['GNN']

        self.encoder_config_sequence = encoder_config['pocket_sequence']
        self.use_cross_attention=encoder_config['cross_attention']
        self.first_hyperedge=encoder_config['encoder_HGNN']['first_hyperedge']
        self.space_edge=encoder_config['HGNN_first_hyperedge']['space_edge']
        self.sequence_edge=encoder_config['HGNN_first_hyperedge']['sequence_edge']


        # combined layers
        self.fc1=nn.Linear(512,256)
        self.fc2=nn.Linear(768,256)
        self.fc3=nn.Linear(512,256)

        self.W_matrix=encoder_config['encoder_HGNN']['W_matrix']

        if self.GNN_train:
            self.gnn=Protein_GAT(encoder_config['encoder_GAT'])

        if self.HGNN_train:

            if encoder_config['encoder_HGNN']['first_hyperedge']:
                self.emb_hgnn=HGNN(encoder_config['HGNN_first_hyperedge'])

            else:
                raise ValueError("error")




    def protein_sequence(self, sequence):
        # if self.encoder_config_sequence:  
        #     token_ids = torch.tensor([self.tokenizer.encode(sequence)]).to(self.device)  
        #     with torch.no_grad():  
        #         output = self.protein_model(token_ids)
        #     sequence_out = output[0]
        #     sequence_mean = torch.mean(sequence_out, dim=1)
        #     embedding = self.fc2(sequence_mean)
        #     return embedding
        # else:
            return None  
        
        
    def forward(self, gnn_data, hgnn_data):
        if self.HGNN_train:
            if self.first_hyperedge:
                if (self.space_edge and self.sequence_edge):
                    node_hgnn,_,_=self.emb_hgnn(hgnn_data.x,
                                            hgnn_data.edge_index,
                                            hgnn_data.batch,
                                            )
                elif self.sequence_edge:
                    node_hgnn,_,_=self.emb_hgnn(hgnn_data.x,
                                            hgnn_data.edge_index,
                                            hgnn_data.batch,
                                            )
                elif self.space_edge:
                    node_hgnn,_,_=self.emb_hgnn(hgnn_data.x,
                                            hgnn_data.edge_index,
                                            hgnn_data.batch,
                                            )
                else:
                    raise ValueError("")
            else:
                raise ValueError("")
        
        if self.GNN_train:
            node_gnn=self.gnn(gnn_data.x[:, :8],
                                gnn_data.edge_index,
                                gnn_data.edge_attr,
                                gnn_data.batch )
            # node_represent = node_gnn  [16,256]
        
        
        if (self.HGNN_train and self.GNN_train):
            graph_embedding=torch.cat([node_hgnn,node_gnn],dim=1)
            graph_embedding=self.fc1(graph_embedding)
        elif self.HGNN_train:
            graph_embedding=node_hgnn
        elif self.GNN_train:
            graph_embedding=node_gnn
        else:
            graph_embedding=None
        if self.encoder_config_sequence:
            sequence_embeddings = []
            for seq in gnn_data.sequence:
                embedding = self.protein_sequence(seq)
                sequence_embeddings.append(embedding)
            sequence_embedding = torch.stack(sequence_embeddings)  # [batch_size,1,256]
            sequence_embedding = sequence_embedding.to(self.device)
            # print(sequence_embedding.shape)
            # print(graph_embedding.shape)
            

            if graph_embedding==None:
                graph_embedding=sequence_embedding
            else:
                graph_embedding = graph_embedding.unsqueeze(1)   

                if self.use_cross_attention:
                    Cross_Attention=CrossAttention(256).to(self.device)
                    graph_embedding=Cross_Attention(graph_embedding,sequence_embedding)
                else:
                    graph_embedding=torch.cat([graph_embedding,sequence_embedding],dim=2)
                    graph_embedding=self.fc3(graph_embedding)


        return graph_embedding


class HGNN(nn.Module):
    def __init__(self, HGNN_config):
        super(HGNN, self).__init__()
        self.W_matrix=HGNN_config['W_matrix']
        num_features = HGNN_config['num_features']
        n_hid=HGNN_config['hidden_size']
        out_size=HGNN_config['out_size']
        self.dropout=0.3
        use_attention=False
        self.hgc1 = HypergraphConv(in_channels=num_features, out_channels=n_hid, use_attention=use_attention)
        self.hgc2 = HypergraphConv(in_channels=n_hid, out_channels=out_size, use_attention=use_attention)
        self.set2set = Set2Set(in_channels=out_size, processing_steps=5, num_layers=2)

    def forward(self, x, hyperedge_index, batch):
        print("hgnn")
        x = F.relu(self.hgc1(x, hyperedge_index))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, hyperedge_index)
        x = F.elu(x)
        hyperedge_features=None
        return self.set2set(x, batch), x, batch

class JKMCNWMEmbeddingNet(torch.nn.Module):
    """
    Jumping knowledge embedding net inspired by the paper "Representation 
    Learning on Graphs with Jumping Knowledge Networks".
    The GNN layers are now MCNWMConv layer.
    """

    def __init__(self, num_features,
                 dim, train_eps, num_edge_attr,
                 num_layers, num_channels=1,
                 layer_aggregate='max'):
        super(JKMCNWMEmbeddingNet, self).__init__()
        self.num_layers = num_layers
        self.layer_aggregate = layer_aggregate

        self.conv0 = MCNWMConv(
            in_dim=num_features,
            out_dim=dim,
            num_channels=num_channels,
            num_edge_attr=num_edge_attr,
            train_eps=train_eps
        )
        self.bn0 = torch.nn.BatchNorm1d(dim)

        for i in range(1, self.num_layers):
            exec('self.conv{} = MCNWMConv(in_dim=dim, out_dim=dim, num_channels={}, num_edge_attr=num_edge_attr, train_eps=train_eps)'.format(
                i, num_channels))
            exec('self.bn{} = torch.nn.BatchNorm1d(dim)'.format(i))

        self.set2set = Set2Set(
            in_channels=dim, processing_steps=5, num_layers=2)

    def forward(self, x, edge_index, edge_attr, batch):
        # GNN layers
        # 
        layer_x = []  # jumping knowledge
        for i in range(0, self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))

            bn = getattr(self, 'bn{}'.format(i))
            x = F.leaky_relu(conv(x, edge_index, edge_attr))
            x = bn(x)

            layer_x.append(x)

        # layer aggregation
        if self.layer_aggregate == 'max':
            x = torch.stack(layer_x, dim=0)
            x = torch.max(x, dim=0)[0]
        elif self.layer_aggregate == 'mean':
            x = torch.stack(layer_x, dim=0)
            x = torch.mean(x, dim=0)[0]

        return self.set2set(x, batch), x, batch

class MCNWMConv(torch.nn.Module):
    """
    Multi-channel neural weighted message module.
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 num_channels,
                 num_edge_attr=1,
                 train_eps=True,
                 eps=0):
        super(MCNWMConv, self).__init__()
        self.nn = Sequential(
            Linear(in_dim * num_channels, out_dim),
            LeakyReLU(),
            Linear(out_dim, out_dim)
        )
        self.NMMs = ModuleList()

        # add the message passing modules
        for _ in range(num_channels):
            self.NMMs.append(NWMConv(num_edge_attr, train_eps, eps))

    def forward(self, x, edge_index, edge_attr):
        # compute the aggregated information for each channel
        channels = []
        for nmm in self.NMMs:
            channels.append(
                nmm(x=x, edge_index=edge_index, edge_attr=edge_attr))

        # concatenate output of each channel
        x = torch.cat(channels, dim=1)

        # use the neural network to shrink dimension back
        x = self.nn(x)

        return x

class NWMConv(MessagePassing):
    """
    The neural weighted message (NWM) layer. output of 
    multiple instances of this will produce multi-channel 
    output.
    """

    def __init__(self, num_edge_attr=1, train_eps=True, eps=0):
        super(NWMConv, self).__init__(aggr='add')
        self.edge_nn = Sequential(
            Linear(num_edge_attr, 8),
            LeakyReLU(),
            Linear(8, 1),
            ELU()
        )
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        # self.reset_parameters()

    def forward(self, x, edge_index, edge_attr, size=None):
        if isinstance(x, Tensor):
            x = (x, x)  # x: OptPairTensor

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            size=size
        )

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return out

    def message(self, x_j, edge_attr):
        weight = self.edge_nn(edge_attr)

        # message size: num_features or dim
        # weight size: 1
        # all the dimensions in a node masked by one weight
        # generated from edge attribute
        return x_j * weight

    def __repr__(self):
        return '{}(edge_nn={})'.format(
            self.__class__.__name__, self.edge_nn
        )


class CrossAttention(nn.Module):
    def __init__(self, emb_dim, att_dropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        self.Wq = nn.Linear(emb_dim, emb_dim) 
        self.Wk = nn.Linear(emb_dim, emb_dim) 
        self.Wv = nn.Linear(emb_dim, emb_dim) 

        

    def forward(self, x, context, pad_mask=None):
        ''' 
        :param x: [batch_size, seq_len_x, emb_dim]
        :param context: [batch_size, seq_len_context, emb_dim]
        :param pad_mask: [batch_size, seq_len_context]
        :return:
        '''
        b, seq_len_x, _ = x.shape
        seq_len_context = context.shape[1]

        Q = self.Wq(x)  # [batch_size, seq_len_x, emb_dim]
        K = self.Wk(context)  # [batch_size, seq_len_context, emb_dim]
        V = self.Wv(context)  # [batch_size, seq_len_context, emb_dim]

        att_weights = torch.einsum('bid,bjd -> bij', Q, K) * self.scale

        if pad_mask is not None:
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bij, bjd -> bid', att_weights, V)  # [batch_size, seq_len_x, emb_dim]
        
        return out, att_weights