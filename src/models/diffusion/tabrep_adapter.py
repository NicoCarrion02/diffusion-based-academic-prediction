"""
TabRep Continuous Generative Architecture for Tabular Data.
Faithfully implements the official TabRep-Flow methodology (Si et al., arXiv:2504.04798,
official repository: https://github.com/jacobyhsi/TabRep):

Original TabRep Components (Preserved Verbatim):
1. Unified Continuous Representation:
   - Numerical features: sklearn QuantileTransformer(output_distribution='normal',
     n_quantiles=max(min(N // 30, 1000), 10), subsample=1_000_000_000, random_state=seed)
     and exact .inverse_transform().
   - Categorical features: CatConverter 2D unit-circle Phase-Shift Keying (PSK)
     via convert_psk (sin/cos roots of unity) and psk2cat (atan2 angular decoding).
2. Generative Flow Matching Backbone & Training:
   - Linear Conditional Flow Matching (CFM) path: x_t = t * x_1 + (1 - t) * x_0,
     target velocity u_t = x_1 - x_0, unweighted MSE objective ||v_theta(x_t, t) - u_t||_2^2.
   - Official TabSYN / TabDDPM feed-forward velocity MLP (TabRepFlowMLP):
     Linear(d_in, dim_t) + SinusoidalTimeEmbedding MLP (dim_t=1024), followed by
     4-layer feed-forward MLP [1024, 2048, 2048, 1024] with ReLU activations,
     dropout=0.0, and no LayerNorm.
   - Adam optimizer (weight_decay=5e-4) with ReduceLROnPlateau scheduler and
     best-loss checkpoint restoration.
   - Deterministic 50-step ODE numerical integration over t in [0, 1] (Euler / Heun).

Dataset / Task-Specific Adaptations for OECD PISA 2022 (Explicitly Isolated):
- snap_discrete_levels (bool, default=True): Post-QuantileTransformer inverse-transform
  nearest-domain snapping for PISA ordinal Likert scales and binary indicators so
  synthesized item responses match valid OECD questionnaire scale codes. Can be disabled
  (snap_discrete_levels=False) for pure continuous-output ablations.
- Explicit "MISSING" string token fallback for unseen nominal categories during transform().
- Minority-class (Class 1) synthesis scope orchestrated upstream by DatasetBalancer.
"""

import copy
import math
import logging
import random
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Ensure strict reproducibility across Python, NumPy, and PyTorch CPU/CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def convert_psk(cat_tensor: torch.Tensor, category_sizes: List[int]) -> torch.Tensor:
    """
    CatConverter Phase-Shift Keying (PSK) encoding from official TabRep (jacobyhsi/TabRep).
    Maps discrete category indices k in {0, ..., K-1} to 2D unit circle (sin, cos).
    """
    device = cat_tensor.device
    cat_tensor = cat_tensor.long()
    cat_spacings = 2.0 * math.pi / torch.tensor(category_sizes, dtype=torch.float32, device=device)
    phases = cat_tensor.float() * cat_spacings
    psk_enc = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
    return psk_enc


def psk2cat(psk_tensor: torch.Tensor, category_sizes: List[int]) -> torch.Tensor:
    """
    CatConverter PSK decoding from official TabRep (jacobyhsi/TabRep).
    Inverts 2D (sin, cos) continuous coordinates back to discrete category indices using atan2.
    """
    device = psk_tensor.device
    cat_len = len(category_sizes)
    cat_spacings = 2.0 * math.pi / torch.tensor(category_sizes, dtype=torch.float32, device=device)

    sin_phases = psk_tensor[:, :cat_len]
    cos_phases = psk_tensor[:, cat_len : 2 * cat_len]

    phases = torch.atan2(sin_phases, cos_phases)
    phases = (phases + 2.0 * math.pi) % (2.0 * math.pi)

    feature_cat = torch.round(phases / cat_spacings).long()
    for i, K in enumerate(category_sizes):
        feature_cat[:, i] = torch.where(feature_cat[:, i] >= K, 0, feature_cat[:, i])

    return feature_cat


class TabRepContinuousEncoder:
    """
    Unified Continuous Tabular Representation (Official TabRep):
    - Numerical features: sklearn QuantileTransformer(output_distribution='normal').
    - Categorical features: CatConverter integer indexing for 2D circular PSK phases.

    Dataset-specific adaptation (isolated & optional):
    - snap_discrete_levels: When True, snaps inverted ordinal/binary survey items back to
      their observed valid OECD scale levels after QuantileTransformer.inverse_transform().
    """
    def __init__(self, random_state: int = 42, snap_discrete_levels: bool = True):
        self.random_state = random_state
        self.snap_discrete_levels = snap_discrete_levels
        self.feature_names: List[str] = []
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.category_sizes: List[int] = []
        self.cat_mappings: Dict[str, List[str]] = {}
        self.discrete_levels: Dict[str, np.ndarray] = {}
        self.quantile_transformer: Optional[QuantileTransformer] = None
        self.is_fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        discrete_cols: Optional[List[str]] = None,
        categorization: Optional[Any] = None,
    ) -> "TabRepContinuousEncoder":
        self.feature_names = df.columns.tolist()

        if categorization is not None:
            resolved_cat = [c for c in categorization.nominal_cols if c in self.feature_names]
            resolved_num = [c for c in categorization.numerical_cols if c in self.feature_names]
            resolved_disc = [c for c in categorization.ordinal_discrete_cols if c in self.feature_names]
        else:
            if cat_cols is None and num_cols is None:
                raise ValueError(
                    "TabRepContinuousEncoder requires either 'categorization' or explicit 'cat_cols' and 'num_cols'."
                )
            resolved_cat = list(cat_cols) if cat_cols is not None else []
            resolved_num = list(num_cols) if num_cols is not None else [c for c in self.feature_names if c not in resolved_cat]
            resolved_disc = list(discrete_cols) if discrete_cols is not None else []

        assigned = set(resolved_cat + resolved_num)
        unassigned = [c for c in self.feature_names if c not in assigned]
        if unassigned:
            raise ValueError(
                f"Unassigned columns found in TabRep input data: {unassigned}. "
                f"All features must be categorized."
            )

        overlap = set(resolved_cat).intersection(set(resolved_num))
        if overlap:
            raise ValueError(f"Columns assigned to both categorical and numerical: {list(overlap)}")

        self.cat_cols = resolved_cat
        self.num_cols = resolved_num

        logger.info(
            f"TabRep Continuous Encoder fit: {len(self.num_cols)} numerical/continuous features, "
            f"{len(self.cat_cols)} nominal categorical features."
        )

        # Official TabRep numerical representation: QuantileTransformer -> N(0, 1)
        self.discrete_levels = {}
        if self.num_cols:
            N = len(df)
            n_quantiles = max(min(N // 30, 1000), 10)
            self.quantile_transformer = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
                subsample=1_000_000_000,
                random_state=self.random_state,
            )
            self.quantile_transformer.fit(df[self.num_cols].values)

            # Dataset-specific PISA adaptation: record valid domain levels for ordinal/binary scales
            if self.snap_discrete_levels:
                target_discrete = [c for c in resolved_disc if c in self.num_cols]
                for col in target_discrete:
                    self.discrete_levels[col] = np.sort(df[col].dropna().unique())
                logger.info(
                    f"[PISA Adaptation] Configured {len(self.discrete_levels)} discrete ordinal/binary "
                    f"features for post-inversion domain snapping."
                )

        # Official CatConverter vocabulary indexing
        self.category_sizes = []
        self.cat_mappings = {}
        for col in self.cat_cols:
            cats = sorted(df[col].dropna().astype(str).unique().tolist())
            if not cats:
                cats = ["MISSING"]
            self.cat_mappings[col] = cats
            self.category_sizes.append(len(cats))

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transforms DataFrame to (continuous_numerical_array, discrete_categorical_indices).
        Unseen categorical tokens (if transform is called on external cohorts) fall back
        to 'MISSING' when present in the fitted vocabulary, else index 0.
        """
        if not self.is_fitted:
            raise ValueError("TabRepContinuousEncoder must be fitted before calling transform.")

        if self.num_cols:
            num_norm = self.quantile_transformer.transform(df[self.num_cols].values).astype(np.float32)
        else:
            num_norm = np.empty((len(df), 0), dtype=np.float32)

        if self.cat_cols:
            cat_indices_list = []
            for col in self.cat_cols:
                cats = self.cat_mappings[col]
                cat_to_idx = {c: i for i, c in enumerate(cats)}
                fallback_idx = cat_to_idx.get("MISSING", 0)
                mapped = df[col].fillna("MISSING").astype(str).map(cat_to_idx)
                n_unseen = int(mapped.isna().sum())
                if n_unseen > 0:
                    logger.debug(
                        f"TabRepContinuousEncoder.transform: mapped {n_unseen} unseen category tokens "
                        f"in '{col}' to fallback index {fallback_idx} ('{cats[fallback_idx]}')."
                    )
                col_idx = mapped.fillna(fallback_idx).values.astype(np.int64)
                cat_indices_list.append(col_idx[:, None])
            cat_indices = np.hstack(cat_indices_list)
        else:
            cat_indices = None

        return num_norm, cat_indices

    def inverse_transform(self, num_norm: np.ndarray, cat_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Inverts continuous representation back to original tabular space using official
        QuantileTransformer.inverse_transform() and CatConverter index lookup.
        Applies optional nearest-level snapping only to registered discrete survey items
        when snap_discrete_levels=True.
        """
        data_dict = {}

        if self.num_cols and num_norm.shape[1] > 0:
            num_orig = self.quantile_transformer.inverse_transform(num_norm)
            for i, col in enumerate(self.num_cols):
                col_vals = num_orig[:, i]
                if self.snap_discrete_levels and col in self.discrete_levels:
                    levels = self.discrete_levels[col]
                    dists = np.abs(col_vals[:, None] - levels[None, :])
                    nearest_idx = np.argmin(dists, axis=1)
                    col_vals = levels[nearest_idx]
                data_dict[col] = col_vals

        if self.cat_cols and cat_indices is not None:
            for i, col in enumerate(self.cat_cols):
                cats = self.cat_mappings[col]
                col_idx = np.clip(cat_indices[:, i], 0, len(cats) - 1).astype(int)
                data_dict[col] = np.array(cats)[col_idx]

        out_df = pd.DataFrame(data_dict)
        return out_df[self.feature_names]


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous flow time t in [0, 1]."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        half_dim = self.embedding_dim // 2
        denom = max(half_dim - 1, 1)
        emb_factor = math.log(10000.0) / denom
        freqs = torch.exp(-emb_factor * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device))
        args = timesteps.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class TabRepFlowMLP(nn.Module):
    """
    Official TabRep-Flow / TabSYN feed-forward velocity network (MLPDiffusion).
    Matches Si et al. (arXiv:2504.04798, Appendix D.4 / Table 5) & jacobyhsi/TabRep:
    - Input projection Linear(input_dim, dim_t) summed once with SinusoidalTimeEmbedding MLP (dim_t=1024).
    - 4-layer feed-forward MLP [1024, 2048, 2048, 1024] with ReLU activations,
      dropout=0.0, and no LayerNorm or per-block residual skip connections.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [1024, 2048, 2048, 1024],
        dim_t: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim_t = dim_t
        self.proj = nn.Linear(input_dim, dim_t)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(dim_t),
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )

        act_cls = nn.ReLU if activation.lower() == "relu" else nn.SiLU
        layers: List[nn.Module] = []
        prev_dim = dim_t
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.repeat(x.size(0))
        h = self.proj(x) + self.time_embed(t)
        return self.mlp(h)


class TabRepDiffusionModel:
    """
    Official TabRep-Flow generative model combining:
    - QuantileTransformer(output_distribution='normal') & 2D CatConverter PSK representations
    - Feed-forward MLPDiffusion velocity backbone (TabRepFlowMLP) trained via linear CFM
    - Adam + ReduceLROnPlateau + best-loss checkpoint restoration
    - Deterministic 50-step ODE integration (Heun / Euler)
    """
    def __init__(
        self,
        num_timesteps: int = 50,
        hidden_dims: Optional[List[int]] = None,
        dim_t: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        architecture: str = "tabrep_mlp",
        snap_discrete_levels: bool = True,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.num_timesteps = num_timesteps
        self.architecture = architecture.lower()
        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [1024, 2048, 2048, 1024]
        self.dim_t = dim_t
        self.dropout = dropout
        self.activation = activation
        self.snap_discrete_levels = snap_discrete_levels
        self.random_state = random_state

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(
            f"TabRep-Flow ({self.architecture}, hidden={self.hidden_dims}) initialized on "
            f"device: {self.device} (seed: {self.random_state})"
        )
        set_seed(self.random_state)

        self.encoder = TabRepContinuousEncoder(
            random_state=self.random_state,
            snap_discrete_levels=self.snap_discrete_levels,
        )
        self.network: Optional[nn.Module] = None
        self.continuous_dim: int = 0
        self.num_numerical_features: int = 0
        self.category_sizes: List[int] = []
        self.loss_history_: List[float] = []
        self.best_loss_: float = float("inf")

    def fit(
        self,
        df: pd.DataFrame,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        discrete_cols: Optional[List[str]] = None,
        categorization: Optional[Any] = None,
        epochs: int = 600,
        batch_size: int = 512,
        lr: float = 0.001,
        weight_decay: float = 5e-4,
        lr_scheduler_factor: float = 0.9,
        lr_scheduler_patience: int = 30,
        restore_best: bool = True,
    ) -> "TabRepDiffusionModel":
        """
        Fit TabRep continuous representation and train Flow Matching velocity field.
        """
        set_seed(self.random_state)
        logger.info(f"Fitting TabRep continuous encoder on {len(df)} samples...")
        self.encoder.fit(
            df,
            cat_cols=cat_cols,
            num_cols=num_cols,
            discrete_cols=discrete_cols,
            categorization=categorization,
        )
        self.num_numerical_features = len(self.encoder.num_cols)
        self.category_sizes = self.encoder.category_sizes

        num_norm, cat_indices = self.encoder.transform(df)
        tensors = []
        if self.num_numerical_features > 0:
            tensors.append(torch.tensor(num_norm, dtype=torch.float32))

        if len(self.category_sizes) > 0 and cat_indices is not None:
            cat_t = torch.tensor(cat_indices, dtype=torch.long)
            psk_t = convert_psk(cat_t, self.category_sizes)
            tensors.append(psk_t)

        continuous_data = torch.cat(tensors, dim=-1)
        self.continuous_dim = continuous_data.shape[1]

        logger.info(
            f"TabRep continuous manifold dimension: {self.continuous_dim} "
            f"(Original features: {df.shape[1]}, Numerical: {self.num_numerical_features}, "
            f"Categorical PSK: {self.continuous_dim - self.num_numerical_features})"
        )

        if self.architecture == "tabrep_mlp":
            self.network = TabRepFlowMLP(
                input_dim=self.continuous_dim,
                hidden_dims=self.hidden_dims,
                dim_t=self.dim_t,
                dropout=self.dropout,
                activation=self.activation,
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported TabRep architecture '{self.architecture}'. Expected 'tabrep_mlp'.")

        dataset = TensorDataset(continuous_data)
        gen = torch.Generator()
        gen.manual_seed(self.random_state)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=gen)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
        )

        self.network.train()
        self.loss_history_ = []
        self.best_loss_ = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        log_interval = max(epochs // 10, 10)

        logger.info(
            f"Starting Official TabRep-Flow training ({epochs} epochs, batch_size={batch_size}, "
            f"lr={lr}, weight_decay={weight_decay})..."
        )

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for (batch_x1,) in dataloader:
                batch_x1 = batch_x1.to(self.device)
                bs = batch_x1.size(0)

                x0 = torch.randn_like(batch_x1)
                t = torch.rand(bs, device=self.device)

                xt = t[:, None] * batch_x1 + (1.0 - t[:, None]) * x0
                ut = batch_x1 - x0

                vt = self.network(xt, t)
                loss = F.mse_loss(vt, ut)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * bs

            avg_loss = total_loss / len(dataset)
            self.loss_history_.append(float(avg_loss))
            scheduler.step(avg_loss)

            if avg_loss < self.best_loss_:
                self.best_loss_ = float(avg_loss)
                if restore_best:
                    best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}

            if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
                cur_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"TabRep-Flow Epoch [{epoch}/{epochs}] - CFM Loss: {avg_loss:.5f} "
                    f"(Best: {self.best_loss_:.5f}, LR: {cur_lr:.6f})"
                )

        if restore_best and best_state is not None:
            self.network.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
            logger.info(f"TabRep-Flow training complete. Restored best velocity field (Loss={self.best_loss_:.5f}).")
        else:
            logger.info("TabRep-Flow training complete.")
        return self

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        steps: Optional[int] = None,
        solver: str = "heun",
        batch_size: int = 512,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Sample synthetic tabular records by integrating the Flow Matching ODE over t in [0, 1].
        Supports 'heun' (2nd-order Runge-Kutta) and 'euler' (1st-order) solvers.
        """
        if self.network is None:
            raise ValueError("TabRep model must be fitted before sampling.")

        if seed is not None:
            set_seed(seed)

        self.network.eval()
        N_steps = steps if steps is not None else self.num_timesteps
        solver_key = solver.lower()
        if solver_key not in ("heun", "euler"):
            raise ValueError(f"Unsupported ODE solver '{solver}'. Expected 'heun' or 'euler'.")

        logger.info(
            f"Generating {num_samples} synthetic samples using TabRep-Flow ODE "
            f"(solver={solver_key}, steps={N_steps}) on {self.device}..."
        )

        all_num_samples = []
        all_cat_samples = []
        remaining = num_samples

        with tqdm(total=num_samples, desc="TabRep ODE Sampling", disable=num_samples < 500) as pbar:
            while remaining > 0:
                cur_bs = min(remaining, batch_size)
                z = torch.randn((cur_bs, self.continuous_dim), device=self.device)
                dt = 1.0 / N_steps

                for step_idx in range(N_steps):
                    t_curr = step_idx / N_steps
                    t_vec = torch.full((cur_bs,), t_curr, device=self.device)
                    vt = self.network(z, t_vec)

                    if solver_key == "heun":
                        t_next = min((step_idx + 1) / N_steps, 1.0)
                        t_next_vec = torch.full((cur_bs,), t_next, device=self.device)
                        z_pred = z + vt * dt
                        vt_next = self.network(z_pred, t_next_vec)
                        vt = 0.5 * (vt + vt_next)

                    z = z + vt * dt

                if self.num_numerical_features > 0:
                    syn_num = z[:, :self.num_numerical_features].cpu().numpy()
                    all_num_samples.append(syn_num)

                if len(self.category_sizes) > 0:
                    syn_psk = z[:, self.num_numerical_features:]
                    syn_cat_indices = psk2cat(syn_psk, self.category_sizes).cpu().numpy()
                    all_cat_samples.append(syn_cat_indices)

                remaining -= cur_bs
                pbar.update(cur_bs)

        final_num = np.vstack(all_num_samples) if all_num_samples else np.empty((num_samples, 0))
        final_cat = np.vstack(all_cat_samples) if all_cat_samples else None

        synthetic_df = self.encoder.inverse_transform(final_num, final_cat)
        logger.info(f"Successfully generated {len(synthetic_df)} synthetic tabular records.")
        return synthetic_df
