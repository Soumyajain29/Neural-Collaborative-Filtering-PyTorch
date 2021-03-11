import torch

# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id = 0):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=0))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)
    return model

def use_optimizer(model, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=params['sgd_lr'],
                                    weight_decay= 0)
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                                          lr=params['adam_lr'],
                                                          weight_decay= params['l2_regularization'])

    return optimizer , model