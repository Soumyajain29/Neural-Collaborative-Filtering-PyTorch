from config import gmf_config , mlp_config , neumf_config
from data import PrepareDataset
import numpy as np
from evaluate import Eval
from models.gmf import GMF
from models.mlp import MLP
from models.ncf import NeuMF
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import use_optimizer , save_checkpoint
from copy import deepcopy

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
CUDA_LAUNCH_BLOCKING=1

class train():
  def __init__(self , config):
    self.config = config
    self.evaluation = Eval(config)
    self.num_epoch = config['num_epoch']
    self.dataset = PrepareDataset(config)
    self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.problem = config['problem']
    if self.problem == 'prediction':
      self.train_iterator , self.val_iterator , self.test_iterator , self.num_users , self.num_items = self.get_iterators()
    else:
      self.train_iterator , self.val_iterator , self.test_users, self.test_items , self.num_users , self.num_items = self.get_iterators()
 
  def get_iterators(self):
    if self.problem == 'prediction':
      users , items , labels ,  num_users , num_items = self.dataset.get_instances(path = self.config['train_path'] ,problem = self.problem)
      train_iterator = self.dataset.generator(users , items , labels,  self.config['batch_size'] )
      val_iterator = self.evaluation.get_val_iterator(path = self.config['val_path'] , problem = self.problem )    
      test_iterator = self.evaluation.get_val_iterator(path = self.config['test_path'] , problem = self.problem )
      return train_iterator , val_iterator , test_iterator , num_users , num_items

    else :
      users , items , labels ,val_users , val_items , val_labels ,   num_users , num_items = self.dataset.get_instances(
                                                       path = self.config['train_path'] ,problem = self.problem)
      train_iterator = self.dataset.generator(users , items , labels , self.config['batch_size'])
      val_iterator = self.dataset.generator(val_users , val_items , val_labels , self.config['val_batch_size'])
      test_users , test_items = self.dataset.read_negatives_file(self.config['test_path'])
      return train_iterator , val_iterator , test_users, test_items , num_users , num_items

  def save(self,model, epoch_id, hit_ratio, ndcg):
    #assert hasattr(self, 'model'), 'Please specify the exact model !'
    model_dir = self.config['model_dir']
    save_checkpoint(model, model_dir)

  def training(self):
    problem = self.config['problem']
    best_epoch_loss = float('inf')
    best_epoch_reg_loss = float('inf')
    train_losses = []
    val_losses = []
    test_losses = []
    HR = []
    NDCG = []
    best_ranks = None
    best_adf = None

    model = MLP(self.config , self.num_users , self.num_items)
    model = model.to(self.device)
    
    for name, param in model.named_parameters():
      if param.requires_grad:
          print(name)
    
    optimizer , model = use_optimizer(model , self.config)
    if self.problem is not 'prediction':
      criterian = nn.BCELoss(reduction = 'sum')
      print(criterian)
    else:
      criterian =nn.MSELoss(reduction = 'sum' )
    model.train()
     
    for epoch in range(self.num_epoch):
      epoch_loss = 0.0
      num_int = 0
      epoch_reg_loss = 0.0
      epoch_sum_loss = 0.0
      #reg_loss = None
      for i , batch in enumerate(self.train_iterator):
        user_batch , item_batch ,batch_labels = batch[0] , batch[1] , batch[2]
        user_batch = user_batch.to(self.device)
        item_batch = item_batch.to(self.device)
        batch_labels = batch_labels.to(self.device)
        num_int+= len(user_batch)
        #print(batch_labels)
        optimizer.zero_grad()
        score , similarity_vec = model(user_batch , item_batch)
        #print(pred)
        loss = criterian(score , batch_labels.view(-1,1))
        reg_loss = 0
        #for param in model.parameters():
         # reg_loss = reg_loss + 0.5 * param.norm(2)**2

        factor = 0  #self.config['l2_regularization']
        loss1 = loss + factor * reg_loss
        #loss = loss/len(batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_reg_loss += reg_loss
        epoch_sum_loss += loss1
    
      val_loss , _  = self.evaluation.evaluate(model , criterian , self.val_iterator)
      if self.problem == 'prediction' :
        hr , ndcg = 0,0
        test_loss , analysis_df  = self.evaluation.evaluate(model , criterian , self.test_iterator)
      else:
        test_loss = 0
        hr , ndcg , _ , _ , ranks = self.evaluation.evaluate_hr_ndcg(model , self.test_users , self.test_items)
      
      if val_loss < best_epoch_loss:
        best_epoch_loss = val_loss
        best_epoch = epoch
        model_copy = deepcopy(model)
        best_epoch_hr = hr
        best_epoch_ndcg = ndcg
        #best_adf = analysis_df
        best_epoch_reg_loss = epoch_reg_loss
        best_ranks = ranks
      
      train_losses.append(epoch_loss/num_int)
      val_losses.append(val_loss)
      test_losses.append(test_loss)
      HR.append(hr)
      NDCG.append(ndcg)
      #print(analysis_df[:15])
      #print(epoch_sum_loss/len(self.train_iterator))

      if self.problem == 'prediction' :
        print('''epoch : {}, train_loss : {}, , train_reg_loss : {} , val_loss : {}, test_loss : {}'''.format(epoch , 
            epoch_loss/num_int, epoch_reg_loss/len(self.train_iterator), val_loss , test_loss))

      else: 
        print('''epoch : {}, train_loss : {}, , train_reg_loss : {} , val_loss : {} , HR : {} , NDCG : {}'''.format(epoch , 
            epoch_loss/num_int, epoch_reg_loss/len(self.train_iterator), val_loss , hr , ndcg))
    
    print(best_epoch_loss, best_epoch)
    print(best_ranks[:50])
    self.save(model_copy , best_epoch, best_epoch_hr , best_epoch_ndcg)
    return train_losses , val_losses , test_losses , HR , NDCG , model_copy , best_adf

if __name__ == "__main__" :
  obj = train(mlp_config)
  train_losses , val_losses , test_losses , HR , NDCG ,  model_copy , best_adf = obj.training()
  #print(train_losses)
  #print(best_adf[0:50])
  epochs = [x for x in range(mlp_config['num_epoch'])]
  fig1 = plt.figure(1)
  plt.plot(epochs , train_losses , label = 'train_loss' , color = 'red')
  plt.plot(epochs , val_losses , label = 'val_loss' , color = 'blue')
  #plt.plot(epochs , test_losses , label = 'test_loss' , color = 'brown')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title("Epoch vs Loss")
  plt.legend()
  #fig1.show()
  fig2 = plt.figure(2)
  plt.plot(epochs , HR , label = 'HR@10')
  plt.plot(epochs , NDCG , label = 'NDCG@10')
  plt.xlabel('Epoch')
  plt.ylabel('HR/NDCG')
  plt.title("Epoch vs HR/NDCG")
  # plt.legend()
  plt.show()