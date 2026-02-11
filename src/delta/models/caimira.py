import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Optional
from delta.models.base import BaseDeltaModel
from delta.configs.caimira import CaimiraConfig


class CaimiraModel(BaseDeltaModel):
    
    def __init__(self, config: CaimiraConfig):
        super().__init__()
        
        self.config = config
        
        # item dificulty
        self.layer_d = nn.Linear(config.n_dim_item_embed, self.config.n_dim, bias=True)
        
        # item relevance
        self.layer_r = nn.Linear(config.n_dim_item_embed, self.config.n_dim, bias=True)
        
        # user 
        self.layer_s = nn.Embedding(self.config.n_users, self.config.n_dim)
        self.layer_s.weight.data.normal_(0, 0.001)

    def forward(self, batch):
        
        user_u = batch['u_id']
        answer_y = batch['answer_emb']
                
        # item difficulty
        d_raw = self.layer_d(answer_y)  # item difficulty
        d_norm = d_raw - d_raw.mean(dim=0)
        
        # item relevance
        r_raw = self.layer_r(answer_y)  
        r_norm = F.softmax(r_raw / self.config.r_temperature, dim=-1)
        
        # user 
        s = self.layer_s(user_u)  # user embedding
        
        latent_score = (s - d_norm)
        logits = torch.einsum("bn,bn->b", latent_score, r_norm)
        
        loss_reg_s = self.config.lambda_s * self.layer_s.weight.abs().sum()
        loss_reg_d = self.config.lambda_d * self.layer_d.weight.abs().sum()
        reg_loss = loss_reg_s + loss_reg_d
        
        rec_loss = torch.tensor(0.0)  # no reconstruction loss in this model        
        
        return {'logits': logits, 's': s, 'd': d_norm, 'r': r_norm, 'reg_loss': reg_loss, 'recon_loss': rec_loss}

    def compute_regularization( self ) :
        
        loss_reg_s = self.config.lambda_s * self.layer_s.weight.abs().sum()
        loss_reg_d = self.config.lambda_d * self.layer_d.weight.abs().sum()
        
        logs = {
            "loss_reg_s": loss_reg_s,
            "loss_reg_d": loss_reg_d,
        }
        
        total_loss = loss_reg_s + loss_reg_d
        
        return total_loss, logs    



            
    
if __name__ == "__main__":
    config = CaimiraConfig(
        n_dim=32,
        n_users=100,
        n_dim_item_embed=16,
        n_dim_user_embed=16
    )
    batch_size = 64    
    model = CaimiraModel(config=config)
    user_indices = torch.randint(0, 10, (batch_size,))
    item_embeddings = torch.randn(batch_size, 16)
    
    output = model(user_u=user_indices, answer_y=item_embeddings)
    print("Output logits shape:", output.logits.shape)
    print("Output difficulty shape:", output.d.shape)
    print("Output relevance shape:", output.r.shape)
    print("Output skill shape:", output.s.shape)
    #print(output.logits)    
    
    