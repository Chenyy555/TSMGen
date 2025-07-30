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
from torch_geometric.nn import GATConv
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
import math
import logging


import torch
import torch.nn as nn
from torch.nn import functional as F


# ipdb.set_trace() 

logger = logging.getLogger(__name__)
class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):      
        self.vocab_size = vocab_size                       # 94
        self.block_size = block_size                       # 54
        for k,v in kwargs.items():                         
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        num = int(bool(config.num_props))   #int(config.lstm_layers)    #  int(config.scaffold) 
        # num = 1
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                                     .view(1, 1, config.block_size + num, config.block_size + num))

        self.n_head = config.n_head


    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # q, k = apply_rotary_pos_emb(q, k, T, self.head_dim)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()

        self.use_gate=config.use_gate
        self.att_num=config.att_num
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        if config.use_gate:
            config.use_encoder_norm=True
            config.use_alpha=True
        self.use_encoder_norm=config.use_encoder_norm
        self.use_alpha=config.use_alpha

        if config.use_encoder_norm:
            self.ln4 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        if self.att_num == 1:
            self.encoder_attn = EncoderDecoderAttention(config.n_embd,config.use_gate)
        elif self.att_num == 2:
            self.encoder_attn = EncoderDecoderAttention(config.n_embd,config.use_gate)
            self.encoder_attn2 = EncoderDecoderAttention(config.n_embd,config.use_gate)
        else:
            print("error")
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.alpha2 = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, device, encoder_output):
        y_self, attn_self = self.attn(self.ln1(x))

        x = x + y_self

        if self.use_encoder_norm:
            encoder_output=self.ln4(encoder_output)
        
        if self.att_num == 1:
            # y_enc_dec, attn_enc_dec = self.encoder_attn(encoder_output, self.ln3(x))
            y_enc_dec, attn_enc_dec = self.encoder_attn(self.ln3(x),encoder_output)
            x = x + y_enc_dec    # x = x + y_enc_dec * self.alpha
        else:
            y_enc_dec, attn_enc_dec = self.encoder_attn(self.ln3(x),encoder_output)
            y_enc_dec2, attn_enc_dec = self.encoder_attn2(encoder_output, self.ln3(x))
            x = x + y_enc_dec* self.alpha + y_enc_dec2

        x = x + self.mlp(self.ln2(x))

        return x, attn_self


class GPTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)  
        self.type_emb = nn.Embedding(2, config.n_embd)                 
        if config.num_props:                                           
            self.prop_nn = nn.Linear(config.num_props, config.n_embd)  
     
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)         # embd_pdrop=0.1       
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])  
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 

        self.block_size = config.block_size

        if config.pretain is False:
            self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, prop = None, length=None):
        self.targets=None
        b, t = idx.size()   
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        token_embeddings = self.tok_emb(idx)          
        position_embeddings = self.pos_emb[:, :t, :]  
        type_embeddings = self.type_emb(torch.ones((b,t), dtype = torch.long, device = idx.device)) 
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)  

        if isinstance(prop, tuple):
            prop = prop[0]  

        if prop.dim() == 2:  
            prop = prop.unsqueeze(1)  

        protein_emb = prop

        if self.config.num_props:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device)) 
            prop += type_embd             
            x = torch.cat([prop, x], 1)   



        attn_maps = []
        for layer in self.blocks:    
            x, attn = layer(x, idx.device, protein_emb) 
            attn_maps.append(attn)    

        x = self.ln_f(x)    
        logits = self.head(x) 

        
        if self.config.num_props:
            num = int(bool(self.config.num_props)) 
        else:
            num = 0


        logits = logits[:, num:, :]    
        logits=logits.reshape(-1, logits.size(-1))  
        loss = None


        return logits, loss, attn_maps # (num_layers, batch_size, num_heads, max_seq_len, max_seq_len)


    def conditioned_sample(self, idx, prop=None, length=None):
        self.targets=None
        b, t = idx.size()    
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        token_embeddings = self.tok_emb(idx)          
        position_embeddings = self.pos_emb[:, :t, :]  
        type_embeddings = self.type_emb(torch.ones((b,t), dtype = torch.long, device = idx.device)) 
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)  

        if isinstance(prop, tuple):
            prop = prop[0]  

        if prop.dim() == 2: 
            prop = prop.unsqueeze(1) 

        protein_emb = prop

        if self.config.num_props:
            prop = prop.repeat(b, 1, 1)  
            type_embd = self.type_emb(torch.zeros((b, 1), dtype=torch.long, device=idx.device))  
            prop += type_embd             

            x = torch.cat([prop, x], 1)   # [16,87,256]


        attn_maps = []
        for layer in self.blocks:
            x, attn = layer(x, idx.device,protein_emb)
            attn_maps.append(attn)


        x = self.ln_f(x)  
        logits = self.head(x)  


        # Remove condition part if necessary
        if self.config.num_props:
            num = int(bool(self.config.num_props))
        else:
            num = 0

        logits = logits[:, num:, :]   # Remove condition part

        loss = None  

        return logits, loss, attn_maps
    
class EncoderDecoderAttention(nn.Module):
    def __init__(self, emb_dim, use_gate=False, att_dropout=0.0, alpha=0.1):
        super(EncoderDecoderAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5
        self.use_gate = use_gate
        self.alpha = alpha 
        self.Wq = nn.Linear(emb_dim, emb_dim)  
        self.Wk = nn.Linear(emb_dim, emb_dim)  
        self.Wv = nn.Linear(emb_dim, emb_dim)  

        if use_gate:
            self.Wg = nn.Linear(emb_dim, emb_dim)  
            self.Wout = nn.Linear(emb_dim, emb_dim)
        self.att_dropout = nn.Dropout(att_dropout)

    def forward(self, x, context, pad_mask=None):
        b, seq_len_x, _ = x.shape
        seq_len_context = context.shape[1]

        Q = self.Wq(x)  # [batch_size, seq_len_x, emb_dim]
        K = self.Wk(context)  # [batch_size, seq_len_context, emb_dim]
        V = self.Wv(context)  # [batch_size, seq_len_context, emb_dim]

        att_weights = torch.einsum('bid,bjd -> bij', Q, K) * self.scale  # [batch_size, seq_len_x, seq_len_context]

        att_weights = F.softmax(att_weights, dim=-1)

        att_weights = self.att_dropout(att_weights)

        out = torch.einsum('bij, bjd -> bid', att_weights, V)  # [batch_size, seq_len_x, emb_dim]
 
        if self.use_gate:
            gate = torch.sigmoid(self.Wg(x))  
            gate_output = torch.tanh(self.Wout(x))  
            out = gate * gate_output + (1 - gate) * x  

        return out, att_weights
