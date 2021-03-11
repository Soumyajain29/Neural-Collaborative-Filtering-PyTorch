from config import gmf_config , mlp_config , neumf_config
from data import PrepareDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pandas as pd

class Eval():
  def __init__(self , config):
    self.config = config
    self.val_path = config['val_path']
    self.dataset = PrepareDataset(config)
    self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.test_path = self.config['test_path']
    self.k  = self.config['k']

  def HR(self ,ranklist , gtItem):
    if gtItem in ranklist:
      return 1 , ranklist.index(gtItem)+1
    else:
      return 0 , 'item_not_in_topK'

  def NDCG(self , ranklist, gtItem):
    #topk_ranklist = ranklist[:self.k]
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log2(2) / math.log2(i+2)
    return 0
  
  def get_val_iterator(self , path , problem):
    if problem == 'prediction':
      users , items , labels , _ , _= self.dataset.get_instances(path , problem)
      val_iterator = self.dataset.generator(users , items , labels , self.config['val_batch_size'])
      return val_iterator
  
  def evaluate(self,model , criterian ,val_iterator) :
    model.eval()
    analysis_df = pd.DataFrame()
    with torch.no_grad():
      val_loss = 0.0
      num_int = 0
      for i , batch in enumerate(val_iterator):
        user_batch = batch[0].to(self.device)
        item_batch = batch[1].to(self.device)
        batch_labels = batch[2].to(self.device)
        num_int += len(user_batch)
        score  , sim_mat = model(user_batch , item_batch)
        #print(pred.shape)
        loss = criterian(score , batch_labels.view(-1,1))
        val_loss += loss.item()

        if i == 0 :
          for j in range(len(batch[0])):
            row = [[int(batch[0][j]),int(batch[1][j]),float(batch[2][j]) , float(score[j])]]
            analysis_df = analysis_df.append(row ,ignore_index=True) 
    return val_loss/num_int , analysis_df

  def evaluate_hr_ndcg(self, model , test_users , test_items):
    model.eval()
    HR_list = []
    NDCG_list = []
    test_item_ranks = []
    for user , items in zip(test_users , test_items):
      user_tensor = [] 
      item_tensor = []
      gtItem = items[0]
      for e in items:
        user_tensor.append(user)
        item_tensor.append(e)
      user_tensor = torch.tensor(user_tensor , dtype = torch.long).cuda()
      item_tensor = torch.tensor(item_tensor , dtype = torch.long).cuda()
      #print(item_tensor)
      #print(item_tensor[:5])
      scores ,  _ = model(user_tensor , item_tensor)
      #print(scores)
      idx_ranking = torch.argsort(scores.flatten() , descending = True)
      #print(idx_ranking)
      ranked_items = []
      for i in range(self.config['k']):
        ranked_items.append(item_tensor[idx_ranking[i]])
      ranked_items = list(map(int , ranked_items))

      #print(ranked_items)
      user_hr , rank = self.HR(ranked_items , gtItem)
      test_item_ranks.append((user , gtItem , rank))
      user_ndcg = self.NDCG(ranked_items , gtItem)
      HR_list.append(user_hr)
      NDCG_list.append(user_ndcg)
    #print(HR_list , NDCG_list)
    analysis_df = pd.DataFrame(test_item_ranks)
    return sum(HR_list)/len(HR_list) , sum(NDCG_list)/len(NDCG_list) , HR_list , NDCG_list , analysis_df
        