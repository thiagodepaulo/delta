from typing import Optional, Tuple
from pydantic import BaseModel


class Model2Config(BaseModel):
    
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
    n_dim_user_embed_2: int
    
    n_dim_user_features: int

    has_user_features: bool = False

    # Model2 regularization hyperparameters
    lambda_r : float #= 1e-6
    lambda_x : float #= 1e-6
    lambda_y : float #= 1e-6
    lambda_u : float #= 1e-6        

    @property
    def arch(self):
        return "three_way"


