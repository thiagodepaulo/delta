import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from delta.configs.model1 import Model1Config
from delta.models.base import BaseDeltaModel


class Model2(BaseDeltaModel):
    """
    Overfitting-resistant version for 1536-d OpenAI embeddings + ~24k pairs + ~1500 users.

    Main ideas:
      - Bottleneck projection for 1536 -> D (reduces capacity + improves generalization)
      - Constrained + small user-id embedding (max_norm, small Du)
      - Stronger regularization on user embeddings (batch-aware)
      - Normalize inputs and user vector (stabilizes + reduces memorization)
    """

    def __init__(self, config: Model1Config):
        super().__init__()
        self.config = config

        D = config.n_dim
        H = int(getattr(config, "proj_hidden", 256))            # bottleneck hidden
        Du = int(getattr(config, "n_dim_user_embed_2", 16))     # user-id embed dim (small)
        drop = float(getattr(config, "dropout", 0.1))
        proj_dropout = float(getattr(config, "proj_dropout", 0.2))
        drop_p = float(getattr(config, "dropout", 0.1))
        user_drop_p = float(getattr(config, "user_dropout", 0.3))
        user_max_norm = float(getattr(config, "user_max_norm", 1.0))

        # 1536-d embeddings -> bottleneck -> D
        self.proj_x = nn.Sequential(
            nn.Linear(config.n_dim_prompt_embed, H),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(H, D),
        )
        self.proj_y = nn.Sequential(
            nn.Linear(config.n_dim_answer_embed, H),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(H, D),
        )

        self.ln_x = nn.LayerNorm(D)
        self.ln_y = nn.LayerNorm(D)
        
        # Joint interaction: (fx, fy) -> fxy
        # You can use Linear(2D->D) or a small MLP
        self.xy = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(D, D),
        )
        self.ln_xy = nn.LayerNorm(D)

        self.has_uf = bool(getattr(config, "has_user_features", False))
        if self.has_uf:
            self.W_u = nn.Linear(config.n_dim_user_features, D, bias=True)
            self.ln_u = nn.LayerNorm(D)

        # Constrained user-id embedding (prevents memorization blow-up)
        self.user_id = nn.Embedding(config.n_users, Du, max_norm=user_max_norm)
        self.user_id.weight.data.normal_(0, 0.01)

        # Map small ID embedding -> D
        self.W_id = nn.Linear(Du, D, bias=False)

        self.drop = nn.Dropout(drop_p)
        self.user_drop = nn.Dropout(user_drop_p)

        # Optional learnable scale (clamped for stability)
        self.scale = nn.Parameter(torch.tensor(float(getattr(config, "init_scale", 1.0))))

    def forward(
        self,
        *,
        user_u: Optional[Tensor] = None,
        prompt_x: Optional[Tensor] = None,
        answer_y: Optional[Tensor] = None,
        user_features: Optional[Tensor] = None,
    ):
        # Normalize OpenAI embeddings (helps a lot)
        prompt_x = F.normalize(prompt_x, dim=-1)
        answer_y = F.normalize(answer_y, dim=-1)

        # Project + normalize
        f_x = self.drop(self.ln_x(self.proj_x(prompt_x)))
        f_y = self.drop(self.ln_y(self.proj_y(answer_y)))
        
        # concat + joint network
        f_xy = self.xy(torch.cat([f_x, f_y], dim=-1))       # [B, D]
        f_xy = self.drop(self.ln_xy(f_xy))
        f_xy = F.normalize(f_xy, dim=-1)

        # Optional feature modulation (keep if you want)
        if self.has_uf and user_features is not None:
            f_u = self.drop(self.ln_u(self.W_u(user_features)))
            f_x = f_x * f_u

        # User-id path (small + constrained + dropout + normalization)
        r_u = self.user_id(user_u.long())     # [B, Du]
        r_u = self.W_id(r_u)                  # [B, D]
        r_u = self.user_drop(r_u)
        r_u = F.normalize(r_u, dim=-1)

        s = self.scale.clamp(0.1, 10.0)
        logits = s * (f_xy * r_u).sum(dim=-1) / math.sqrt(self.config.n_dim)
        return {"logits": logits}

    def compute_regularization(
        self,
        user_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Regularize what actually overfits here: the user embedding table.
        Batch-aware: pass user_ids=u from your batch for best results.
        """
        cfg = self.config
        device = self.W_id.weight.device
        logs: Dict[str, Tensor] = {}
        total = torch.zeros((), device=device)

        # Select embeddings touched this step (recommended)
        E = self.user_id.weight
        if user_ids is not None:
            uuniq = torch.unique(user_ids.long())
            E = E.index_select(0, uuniq)

        # Strong L2 on user embeddings (main control knob)
        lambda_uid_l2 = float(getattr(cfg, "lambda_uid_l2", 1e-3))
        reg_uid_l2 = lambda_uid_l2 * E.pow(2).sum()
        total += reg_uid_l2
        logs["reg_uid_l2"] = reg_uid_l2

        # Optional L1 for sparsity (start at 0, try 1e-4 if needed)
        lambda_uid_l1 = float(getattr(cfg, "lambda_uid_l1", 0.0))
        if lambda_uid_l1 > 0:
            reg_uid_l1 = lambda_uid_l1 * E.abs().sum()
            total += reg_uid_l1
            logs["reg_uid_l1"] = reg_uid_l1

        # Small L2 on W_id (secondary)
        lambda_w2_id = float(getattr(cfg, "lambda_w2_id", 1e-6))
        reg_w2_id = lambda_w2_id * self.W_id.weight.pow(2).sum()
        total += reg_w2_id
        logs["reg_w2_id"] = reg_w2_id

        logs["reg_total"] = total
        return total, logs
