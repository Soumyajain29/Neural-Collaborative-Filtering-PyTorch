import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import resume_checkpoint
import math

class GMF(nn.Module):
  def __init__(self , config ,num_users , num_items):
    super().__init__()
    self.num_users = num_users
    self.num_items = num_items
    self.latent_dim = config['latent_dim_gmf']
    self.user_embed = nn.Embedding(self.num_users  , self.latent_dim)
    self.item_embed = nn.Embedding(self.num_items  , self.latent_dim)
    self.hidden     = nn.Linear(self.latent_dim , self.latent_dim//2)
    self.gmf_layer  = nn.Linear(self.latent_dim//2 , 1)
    self.pretrained = config['pretrained']
    self.dropout  = nn.Dropout(.7)

    self.weight_init()

  def weight_init(self):
    nn.init.normal_(self.user_embed.weight.data, mean=0.0, std= .1)
    nn.init.normal_(self.item_embed.weight.data, mean=0.0, std= .1)
    bound = 1 / math.sqrt(self.latent_dim)
    nn.init.uniform_(self.gmf_layer.weight.data, -bound, bound)
    #nn.init.kaiming_uniform_(self.gmf_layer.weight.data, a = 1)
    self.gmf_layer.bias.data.zero_()
      

  def forward(self , user , item) :
    user_embedding = self.user_embed(user)
    item_embedding = self.item_embed(item)
    #user_embedding = self.dropout(user_embedding)
    #item_embedding = self.dropout(item_embedding)
    user_item_sim  = torch.mul(user_embedding  , item_embedding)
    #user_item_sim =  self.dropout(user_item_sim)
    user_item_sim  = self.hidden(user_item_sim)
    user_item_sim =  self.dropout(user_item_sim)
    score          = self.gmf_layer(user_item_sim)
    #score          = torch.sigmoid(score)
    return score  , user_item_sim