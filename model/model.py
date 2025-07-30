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
import selfies as sf
from tqdm import tqdm
from torch_geometric.nn import GATConv
from torch.nn.parameter import Parameter
import math
import numpy as np
from  utils import hypergraph_util as hgut
# from .GNN import GNN_graphpred, MLP 
from json.tool import main
from webbrowser import get
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F
from model.encoder_protein_atom import GNN_encoder
from model.decoder import GPTDecoder


class Pocket_GNN(torch.nn.Module):
    def __init__(self, train_config, encoder_config, decoder_config, device):
        super(Pocket_GNN, self).__init__()
        self.train_config=train_config
        self.encoder_config=encoder_config
        self.decoder_config=decoder_config

        self.encoder=GNN_encoder(encoder_config,device)

        self.decoder=GPTDecoder(decoder_config)

    def forward(self, gnn_data, hgnn_data, smiles, lengths=None):
        Pocket_pre=self.encoder(gnn_data, hgnn_data)    # [16,256]   [batch_size，embedingg_len]
        
        logits, loss, attn_maps = self.decoder(smiles, Pocket_pre,lengths)

        return logits
    
    def sample_from_pocket(self, gnn_data, hgnn_data, smiles, lengths=None):
   
        Pocket_pre=self.encoder(gnn_data, hgnn_data)    # [16,256]   [batch_size，embedingg_len]
        
        logits, loss, attn_maps = self.decoder.conditioned_sample(smiles, Pocket_pre, lengths)

        return logits

 


