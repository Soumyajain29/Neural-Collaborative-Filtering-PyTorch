import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import resume_checkpoint
import math

class MLP(nn.Module):
  def __init__(self , config , num_users , num_items):
    super().__init__()
    self.config = config
    self.num_users = num_users
    self.num_items = num_items
    self.num_layers = self.config['num_layers_mlp']
    self.layer_0_dim = self.config['latent_dim_mlp']
    self.user_embed = nn.Embedding(self.num_users  , self.layer_0_dim // 2)
    self.item_embed = nn.Embedding(self.num_items  , self.layer_0_dim // 2)
    self.pretrained = self.config['pretrained']
    self.dropout   = nn.Dropout(.7)

    self.hidden_layers  = nn.ModuleList()
    input_size =self.layer_0_dim
    for i in range(self.num_layers):
      output_size = input_size//2 
      self.hidden_layers.append(nn.Linear(input_size , output_size))
      input_size = output_size
    
    self.mlp_layer = nn.Linear(output_size , 1)

    self.weight_init()

  def weight_init(self):
    nn.init.normal_(self.user_embed.weight.data, mean=0.0, std=.01)
    nn.init.normal_(self.item_embed.weight.data, mean=0.0, std=.01)

    fan_in =self.layer_0_dim
    for layer in self.hidden_layers:
      fan_out = fan_in //2
      nn.init.xavier_uniform_(layer.weight.data)
      layer.bias.data.zero_()
      # limit = math.sqrt(6/(fan_in + fan_out))
      # nn.init.uniform_(layer.weight.data , a = -limit , b= limit)
      # fan_in = fan_out
      
    #bound = 1 / math.sqrt(fan_out)
    nn.init.kaiming_uniform_(self.mlp_layer.weight.data, a = 1)
    self.mlp_layer.bias.data.zero_()
    #nn.init.uniform_(self.mlp_layer.weight.data, a =  -bound , b= bound )


  def forward(self , user , item) :
    user_embedding = self.user_embed(user)
    item_embedding = self.item_embed(item)
    latent_vector  = torch.cat([user_embedding , item_embedding] , 1)
    for i , layer in enumerate(self.hidden_layers):
      latent_vector = layer(latent_vector)
      latent_vector = F.relu(latent_vector)
      latent_vector = self.dropout(latent_vector)
    score = self.mlp_layer(latent_vector)
    score = torch.sigmoid(score)
    return score , latent_vector