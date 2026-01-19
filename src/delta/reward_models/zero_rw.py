from delta.reward_models.base import BaseRewardModel
import torch

class ZeroRWModel(BaseRewardModel):
    def __init__(self):
        super().__init__()
        
        # buffer whose only job is to track device & dtype
        self.register_buffer("_device_ref", torch.empty(0))
        
    def forward(self, x, y):
        batch_size = len(x)
        return torch.zeros(batch_size, device=self._device_ref.device)        
