import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import random
from scipy.sparse import dok_matrix
from config import gmf_config ,mlp_config , neumf_config
import collections

SEED = 42      #132
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class PrepareDataset() :
  def __init__(self , config):
    self.config = config
    self.batch_size = config['batch_size']
    self.problem = self.config['problem']

  def read_file_as_matrix(self,file):
    #find number of users and items. Reading file twice to save space on ram
    with open(file, 'r') as f:
      num_users , num_items =  0 ,0
      for line in f:
        interaction = line.split()
        u , i = int(interaction[0]) , int(interaction[1])
        num_users , num_items = max(num_users , u) , max(num_items , i)

    num_users +=1
    num_items +=1
    print('number of users= {} and number of items = {}'.format(num_users , num_items))
    rating_mat = dok_matrix((num_users+1 , num_items+1) , dtype= np.float32) 
    
    with open(file, 'r') as f:
      for line in f:
        interaction = line.split()
        u , i , r = int(interaction[0]) , int(interaction[1]) , float(interaction[2])
        if r > 0:
          rating_mat[u ,i] = r
    return rating_mat , num_users , num_items


  def read_negatives_file(self, path):
    with open(path , 'r') as f:
      test_users = []
      test_items = []
      for line in f :
        user_items = []
        arr = line.split('\t')
        user =  int(eval(arr[0])[0])
        item =  int(eval(arr[0])[1])
        test_users.append(user)
        user_items.append(item)
        for i in arr[1:]:
          user_items.append(int(i))
        test_items.append(user_items)
    return test_users , test_items

  def get_validation_set(self , all_users , all_items , all_labels):
    idx_dict = collections.defaultdict(list)
    for i , key in enumerate(all_users):
        idx_dict[key].append(i)

    users , items , labels , val_users , val_items , val_labels = [] , [] , [] ,[] ,[] , []
    for user in idx_dict.keys():
      val_idx = random.choice(idx_dict[user])
      val_users.append(all_users[val_idx])
      val_items.append(all_items[val_idx])
      val_labels.append(all_labels[val_idx])

      idx_dict[user].remove(val_idx)
      for item_idx in idx_dict[user]:
        users.append(user)
        items.append(all_items[item_idx])
        labels.append(all_labels[item_idx])
    
    print('no. of validation interactions', len(val_users))
    print('no. of train interactions', len(users))
    return users , items , labels , val_users , val_items , val_labels
  
  def get_instances(self, path , problem , num_neg = 4 ):
    #num_neg is number of negative instances corresponding to each positive instance i.e negative sampling
    print(path)
    rating_matrix , num_users , num_items = self.read_file_as_matrix(path)
  
    users , items , labels  = [] , [] , []
    print(num_users , num_items)
    
    count = 0 
    if not problem == 'prediction'  :
      for u , i in rating_matrix.keys(): 
        count+=1
        users.append(u)
        items.append(i)
        labels.append(1)

        #negative sampling
        for t in range(num_neg):
          count+=1
          j = np.random.randint(low = 0, high = num_items-1)
          while (u, j) in rating_matrix.keys():
            j = np.random.randint(low = 0 , high = num_items-1)
          users.append(u)
          items.append(j)
          labels.append(0)

        #create validation set
      users, items, labels , val_users , val_items , val_labels  = self.get_validation_set(users , items , labels)
      print('number of interactions' , count)
      return users, items, labels , val_users , val_items , val_labels , num_users , num_items 
      
    else:
      for u , i in rating_matrix.keys(): 
        count+=1
        users.append(u)
        items.append(i)
        labels.append(rating_matrix[u,i])

      print('number of interactions' , count)
      return users, items, labels ,  num_users , num_items

  def generator(self, users , items , labels , batch_size):
    user_tensor  = torch.tensor(users , dtype = torch.long)
    item_tensor  = torch.tensor(items , dtype = torch.long)
    label_tensor = torch.tensor(labels , dtype = torch.float)
    dataset      = TensorDataset(user_tensor , item_tensor , label_tensor)
    iterator     = DataLoader(dataset , batch_size = batch_size)
    return iterator