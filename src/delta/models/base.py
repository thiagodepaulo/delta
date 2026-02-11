import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from typing import Any, Optional
import os

class BaseDeltaModel(nn.Module):
    
    def save_ckpt(self, path: str):
        torch.save(self.state_dict(), path)

    def load_ckpt(self, path: str, map_location: Optional[str] = None):
        self.load_state_dict(torch.load(path, map_location=map_location))

    def save_pretrained(self, path: str):
        raise NotImplementedError
        
    def forward(self, batch) :
        """
        General interface for delta models.

        - Some models may ignore user_u.
        - Some may not need prompt_x.
        - Some may only need answer_y.
        - Extra inputs can go in **kwargs (e.g. answer_y_neg, mask, metadata).
        """
        raise NotImplementedError
    
    def compute_regularization( self ) :
        """Regularization loss  override only if needed."""
        raise NotImplementedError
    
    
    @classmethod
    def load_pretrained(cls, path: str, device: str = "auto"):
        raise NotImplementedError

    