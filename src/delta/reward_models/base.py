import torch.nn as nn

class BaseRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
            
    def forward(self, batch):
        raise NotImplementedError    
