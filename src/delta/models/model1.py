import torch
import math
import torch.nn as nn
from torch import Tensor
from typing import Any, Optional
from delta.configs.model1 import Model1Config
from delta.models.base import BaseDeltaModel


def create_zero_init_embedding(
    n: int, dim: int, dtype: Any = torch.float32, requires_grad: bool = True
):
    embedding = nn.Embedding(n, dim, _weight=torch.zeros((n, dim), dtype=dtype))
    embedding.weight.requires_grad = requires_grad
    return embedding


class Model1(BaseDeltaModel):
    
    def __init__(self, config: Model1Config):
        super().__init__()
        
        self.config = config
        
        # W_u, W_x : R^D -> R^K
        if self.config.has_user_features:
            self.W_u = nn.Linear(self.config.n_dim_user_features, self.config.n_dim, bias=True)
        self.W_x = nn.Linear(self.config.n_dim_prompt_embed, self.config.n_dim, bias=True)
        self.W_y = nn.Linear(self.config.n_dim_answer_embed, self.config.n_dim, bias=True)
        
        # LayerNorms (normalize last dimension = n_dim)
        self.ln_x = nn.LayerNorm(config.n_dim)
        self.ln_y = nn.LayerNorm(config.n_dim)
        if config.has_user_features:
            self.ln_u = nn.LayerNorm(config.n_dim)            
        
        # user 
        self.user_id = nn.Embedding(self.config.n_users, self.config.n_dim_user_embed_2)
        self.user_id.weight.data.normal_(0, 0.001)
        
        self.W_id = nn.Linear(self.config.n_dim_user_embed_2, self.config.n_dim, bias=False)
        
        self.drop = nn.Dropout(p=0.1)
        
        # Optional but very useful: learnable scale
        #self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self,             
            *,
            user_u: Optional[Tensor] = None,
            prompt_x: Optional[Tensor] = None,
            answer_y: Optional[Tensor] = None,
            user_features: Optional[Tensor] = None,
        ):        
        
        # Project + normalize
        f_x = self.drop(self.ln_x(self.W_x(prompt_x)))
        f_y = self.drop(self.ln_y(self.W_y(answer_y)))

        if self.config.has_user_features:
            f_u = self.ln_u(self.W_u(user_features))
            f_x = f_x * f_u

        r_u = self.user_id(user_u.long())          
        r_u = self.W_id(r_u)                       
        r_u = self.drop(r_u)
        
        logits = (f_x * r_u * f_y).sum(dim=-1) / math.sqrt(self.config.n_dim) #* self.scale
        return {"logits": logits}
    
    def compute_regularization( self ) -> Tensor:
        
        if self.config.has_user_features:
            loss_reg_u = self.config.lambda_u * self.W_u.weight.abs().sum()
        loss_reg_x = self.config.lambda_x * self.W_x.weight.abs().sum()
        loss_reg_y = self.config.lambda_y * self.W_y.weight.abs().sum()
        loss_reg_r = self.config.lambda_r * self.W_id.weight.abs().sum()
        
        logs = {
            "loss_reg_x": loss_reg_x,
            "loss_reg_y": loss_reg_y,
            "loss_reg_r": loss_reg_r,
        }
        
        if self.config.has_user_features:
            logs["loss_reg_u"] = loss_reg_u
            total_loss = loss_reg_u + loss_reg_x + loss_reg_y + loss_reg_r
        else:
            total_loss = loss_reg_x + loss_reg_y + loss_reg_r
        
        return total_loss, logs    

