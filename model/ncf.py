import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gmf import GMF
from models.mlp import MLP
from utils import resume_checkpoint
import math


class NeuMF(nn.Module):
  def __init__(self , config , num_users , num_items):
    super().__init__()
    self.config = config
    self.gmf = GMF(self.config,num_users , num_items)
    self.mlp = MLP(self.config,num_users , num_items)
    inp =self.config['gmf_out_dim'] + self.config['mlp_out_dim']
    self.hidden = nn.Linear(inp , inp//2)
    self.nmf = nn.Linear(inp//2 , 1)
    self.pretrained = self.config['pretrained']
    self.dropout = nn.Dropout(.7)
    self.weight_init()

  def weight_init(self):
    if self.pretrained == True:
      gmf_model = resume_checkpoint(self.gmf, model_dir=self.config['pretrained_gmf_dir'])
      self.gmf.user_embed.weight.data = gmf_model.user_embed.weight.data
      self.gmf.item_embed.weight.data = gmf_model.item_embed.weight.data
      print('Pretrained_gmf_model_loaded')

      mlp_model = resume_checkpoint(self.mlp, model_dir=self.config['pretrained_mlp_dir'])
      self.mlp.user_embed.weight.data = mlp_model.user_embed.weight.data
      self.mlp.item_embed.weight.data = mlp_model.item_embed.weight.data

      for m1 , m2 in zip(self.mlp.hidden_layers , mlp_model.hidden_layers):
        m1.weight.data = m2.weight.data
        m1.bias.data = m2.bias.data
      
      print('Pretrained_mlp_model_loaded')

    else:
      bound = 1 / math.sqrt(self.config['gmf_out_dim'] + self.config['mlp_out_dim'])
      nn.init.uniform_(self.nmf.weight.data, a =  -bound , b= bound )
      #nn.init.kaiming_uniform_(self.nmf.weight.data, a = 1)
      self.nmf.bias.data.zero_()



  def forward(self , user , item) :
    _ , gmf_out = self.gmf(user , item)
    gmf_out = .5 * gmf_out
    #gmf_out = self.dropout(gmf_out)
    _ , mlp_out = self.mlp(user , item)
    mlp_out = .5 * mlp_out
    #mlp_out = self.dropout(mlp_out)
    fused_out = torch.cat([gmf_out , mlp_out],1)
    fused_out = self.hidden(fused_out)
    fused_out = self.dropout(fused_out)
    score = self.nmf(fused_out)
    #score = torch.sigmoid(score)
    return score, fused_out