import torch
from delta.reward_models.base import BaseRewardModel


class MapRWModel(BaseRewardModel):
    def __init__(self, dataset_name: str):
        super().__init__()
        self.dataset_name = dataset_name
        self.reward_map = self.load_reward_map(dataset_name)
        
        # buffer whose only job is to track device & dtype
        self.register_buffer("_device_ref", torch.empty(0))
        
    def load_reward_map(self, dataset_name: str):
        
        if dataset_name == 'prism':
            train_file = '/home/thiagodepaulo/exp/delta/notebooks/reward_train_results_map.pt'
            test_file = '/home/thiagodepaulo/exp/delta/notebooks/reward_test_results_map.pt'
            eval_file = '/home/thiagodepaulo/exp/delta/notebooks/reward_eval_results_map.pt'
            
            map_train = torch.load(train_file)
            map_test = torch.load(test_file)
            map_eval = torch.load(eval_file)

            return map_train | map_test | map_eval
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        
    def forward(self, batch, split=None):
        x = batch['prompt_emb']
        y = batch['answer_emb'] 
        rw_values = [self.reward_map[xi + yi] for xi, yi in zip(x, y)]
        return torch.tensor(rw_values, device=self._device_ref.device)                  
        