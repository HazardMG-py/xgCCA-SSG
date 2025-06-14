import torch
import numpy as np
import random
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(model, log_dir, epoch):
    checkpoint_path = os.path.join(log_dir, f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def log_results(log_dir, message):
    with open(os.path.join(log_dir, 'results.txt'), 'a') as f:
        f.write(f"{message}\n")