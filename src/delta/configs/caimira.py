from typing import Optional, Tuple
from pydantic import BaseModel


class CaimiraConfig(BaseModel):
    
    # Caimira regularization hyperparameters
    lambda_s : float = 1e-6
    lambda_d : float = 1e-6
    
    n_dim: int
    # Number of users
    # If not provided, will be inferred from the data (agent_indexer)
    n_users: Optional[int] = None

    # Number of agent types (e.g. humans, cbqa, ret)
    # If not provided, will be inferred from the data (agent_indexer)
    n_agent_types: Optional[int] = None

    # Boolean flags for trainable parameters
    fit_agent_type_embeddings: bool = False

    fit_guess_bias: bool = False
    #characteristics_bounder: Optional[BoundingConfig] = None
    
    # Number of dimensions in item embeddings
    n_dim_item_embed: int

    # Number of dimensions for the user embedding
    n_dim_user_embed: int

    rel_mode: str = "linear"  # [linear, mlp]
    dif_mode: str = "linear"  # [linear, mlp]

    # Number of hidden units for the MLPs if mode is mlp
    n_hidden_dif: int = 128
    n_hidden_rel: int = 128

    fit_r_importance: bool = True
    # Sparsity controls for importance [only used if fit_importance is True]
    # Temperature for importance
    r_temperature: float = 0.5
    fast: bool = False

    @property
    def arch(self):
        return "caimira"
