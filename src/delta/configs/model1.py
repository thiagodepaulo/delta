from typing import Optional, Tuple
from pydantic import BaseModel


class Model1Config(BaseModel):
    
    n_dim: int
    # Number of users
    # If not provided, will be inferred from the data (agent_indexer)
    n_users: int

    # Number of dimensions in prompt embeddings
    n_dim_prompt_embed: int

    # Number of dimensions in answer embeddings
    n_dim_answer_embed: int

    # Number of dimensions in user embeddings
    n_dim_user_embed: int
    
    n_dim_user_features: int

    has_user_features: bool = False

    # Model1 regularization hyperparameters
    lambda_r : float #= 1e-6
    lambda_x : float #= 1e-6
    lambda_y : float #= 1e-6
    lambda_u : float #= 1e-6        

    @property
    def arch(self):
        return "three_way"


class TrainerConfig(BaseModel):
    # Train time
    max_epochs: int = 100
    max_steps: Optional[int] = None
    sampler: Optional[str] = None
    batch_size: int = 32
    
    scheduler: str = "none"
    
    num_workers: int    
    

    # Optimizer
    optimizer: str = "Adam"  # [Adam, RMSprop, SGD]
    learning_rate: float #= 1e-3
    cyclic_lr: bool = False
    weight_decay: float = 0.0
    
    betas: Tuple[float, float]
    eps: float = 1e-8
    
    #second_optimizer: str = "SGD"
    #second_learning_rate: float = 5e-4
    #second_optimizer_start_epoch: Optional[int] = 75
    
    

    #freeze_bias_after: Optional[int] = None

    #ckpt_savedir: str = "./checkpoints/irt"
    
    #c_reg_s: float = 1e-6
    #c_reg_d: float = 1e-6