base = './dataset/digital_music/'
gmf_config = {'num_epoch':85,
              'batch_size': 2048,  #7000 ok
              #'optimizer': 'sgd',
              'sgd_lr': 1e-2,
              'optimizer': 'adam',
              'adam_lr': 7e-3,   #1e-4  ok
              'num_users': 0,
              'num_items': 0,
              'latent_dim_gmf': 128,
              'problem' : 'topk' ,  #topk or prediction
              'k' : 12,
              'num_negative': 4,
              'l2_regularization': 0    , #ok  #1e-6 ok
              #'train_path': base + 'digital_music.train.rating' , 
              #'val_path'  : base + 'digital_music.valid.rating',
              #'test_path' : base + 'digital_music.test.rating' , 
              'train_path':  './dataset/ml-1m.train.rating' ,    
              'val_path'  :  None,
              'test_path' :  './dataset/ml-1m.test.negative' ,
              'model_dir':'checkpoints/new_prediction_gmf.model' ,
              'pretrained' : False ,
               'pretrained_gmf_dir' : 'checkpoints/new_prediction_gmf.model' , 
               'val_batch_size' : 2048
               }

mlp_config = {'num_epoch': 50,
              'batch_size': 2048,          #1024 ,          #256 ,     #10000 #25000,
              'val_batch_size' : 2048,
               #'optimizer': 'sgd',
              'sgd_lr':  5e-3,
              'optimizer': 'adam',
              'adam_lr':   5e-3,           #5e-5,  #1e-3 ok
              'num_users': 6040,
              'num_items': 3706,
              'num_layers_mlp' : 2,
              'latent_dim_mlp': 16,
              'problem' : 'topk' ,  #topk or prediction
              'num_negative': 4,   #3
              'k' : 13,
              'l2_regularization': 0 ,  #  0.0000001,  # MLP model is sensitive to hyper params
              # 'train_path' :  base + 'digital_music.train.rating' ,   
              # 'val_path'   :  base + 'digital_music.valid.rating',
              # 'test_path'  :  base + 'digital_music.test.rating' , 
              'train_path':  './dataset/ml-1m.train.rating' ,    
              'val_path'  :  None,
              'test_path' :  './dataset/ml-1m.test.negative' ,
              'pretrained' : False,
              'pretrained_mlp_dir': None,
              'model_dir':'checkpoints/new/new_topk_mlp.model'}

neumf_config = {'num_epoch': 60,
                'batch_size': 128,
                'val_batch_size' : 128,
                'num_layers_mlp' : 2,
                #'optimizer': 'sgd',
                'sgd_lr': 1e-3,
                'optimizer': 'adam',
                'adam_lr': 1e-4,                      #1e-2 best       #65.2 , 37.5
                'num_users': 6040,
                'num_items': 3706,
                'problem' :  'prediction' ,      #'topk' , 
                'latent_dim_gmf': 16,                  #16  best
                'latent_dim_mlp': 32,                  #64  best
                'num_negative': 4,
                'k' : 15,
                'l2_regularization': 1e-3,  #0.01,
                'gmf_out_dim' : 8 ,
                'mlp_out_dim' : 8 ,
                'pretrained': False,
                'pretrained_gmf_dir': 'checkpoints/prediction_gmf.model',
                'pretrained_mlp_dir': 'checkpoints/prediction_mlp.model',
                'model_dir':'checkpoints/topk_ncf.model',
                'train_path' :  base + 'digital_music.train.rating' ,   
                'val_path'   :  base + 'digital_music.valid.rating',
                'test_path'  :  base + 'digital_music.test.rating' , 
                # 'train_path':  './dataset/ml-1m.train.rating' ,    
                # 'val_path'  :  None,
                # 'test_path' :  './dataset/ml-1m.test.negative' ,
                }