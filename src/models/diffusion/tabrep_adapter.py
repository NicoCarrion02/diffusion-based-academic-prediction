"""
TabRep Continuous Generative Architecture for Tabular Data.
Faithfully implements the official TabRep methodology (Si et al., arXiv:2504.04798):
1. Continuous Representation:
   - Numerical features: Normal-distributed Quantile Transformation.
   - Categorical features: CatConverter (2D circular PSK unit-circle embedding).
2. Generative Model:
   - Conditional Flow Matching (CFM) on continuous unified data manifold.
   - Deterministic ODE sampling (Heun / Euler numerical integration).
"""

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
    CatConverter Phase-Shift Keying (PSK) encoding from official TabRep.
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
    CatConverter PSK decoding from official TabRep.
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
    Unified Continuous Tabular Representation:
    - Continuous features: QuantileTransformer(output_distribution='normal').
    - Categorical features: CatConverter (2D circular PSK phases).
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_names: List[str] = []
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.category_sizes: List[int] = []
        self.cat_mappings: Dict[str, List] = {}
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

        # Fit numerical quantile transformer with adaptive quantiles
        self.discrete_levels = {}
        if self.num_cols:
            N = len(df)
            n_quantiles = max(min(N // 30, 1000), 10)
            self.quantile_transformer = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
                subsample=1_000_000,
                random_state=self.random_state,
            )
            self.quantile_transformer.fit(df[self.num_cols].values)

            # Register valid domain levels for discrete features
            if discrete_cols is not None:
                target_discrete = [c for c in discrete_cols if c in self.num_cols]
                for col in target_discrete:
                    self.discrete_levels[col] = np.sort(df[col].dropna().unique())

            logger.info(f"Configured {len(self.discrete_levels)} discrete ordinal/binary features for post-diffusion snapping.")

        # Fit categorical mappings
        self.category_sizes = []
        self.cat_mappings = {}
        for col in self.cat_cols:
            cats = sorted(df[col].dropna().unique().tolist())
            if not cats:
                cats = ["MISSING"]
            self.cat_mappings[col] = cats
            self.category_sizes.append(len(cats))

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transforms DataFrame to (continuous_numerical_array, discrete_categorical_indices).
        """
        if not self.is_fitted:
            raise ValueError("TabRepContinuousEncoder must be fitted before calling transform.")

        # Numerical transform to N(0, 1)
        if self.num_cols:
            num_norm = self.quantile_transformer.transform(df[self.num_cols].values).astype(np.float32)
        else:
            num_norm = np.empty((len(df), 0), dtype=np.float32)

        # Categorical transform to integer indices
        if self.cat_cols:
            cat_indices_list = []
            for col in self.cat_cols:
                cats = self.cat_mappings[col]
                cat_to_idx = {c: i for i, c in enumerate(cats)}
                col_idx = df[col].map(cat_to_idx).fillna(0).values.astype(np.int64)
                cat_indices_list.append(col_idx[:, None])
            cat_indices = np.hstack(cat_indices_list)
        else:
            cat_indices = None

        return num_norm, cat_indices

    def inverse_transform(self, num_norm: np.ndarray, cat_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Inverts continuous representation back to original tabular space.
        Guarantees strict support preservation via CDF quantile inversion and PSK modulo mapping.
        """
        data_dict = {}

        if self.num_cols and num_norm.shape[1] > 0:
            num_orig = self.quantile_transformer.inverse_transform(num_norm)
            for i, col in enumerate(self.num_cols):
                col_vals = num_orig[:, i]
                if hasattr(self, "discrete_levels") and col in self.discrete_levels:
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
    """Sinusoidal positional embedding for continuous flow time steps."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        half_dim = self.embedding_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(-emb_factor * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device))
        args = timesteps.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class FlowResidualBlock(nn.Module):
    """Residual dense block with time-step conditioning and layer normalization."""
    def __init__(self, in_features: int, out_features: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.time_proj = nn.Linear(time_emb_dim, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.fc1(x)))
        h = h + self.time_proj(self.act(t_emb))
        h = self.dropout(self.act(self.norm2(self.fc2(h))))
        return h + self.skip(x)


class MLPFlowNet(nn.Module):
    """Velocity prediction network v_theta(x_t, t) for Conditional Flow Matching."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 512, 256],
        time_emb_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(FlowResidualBlock(prev_dim, h_dim, time_emb_dim, dropout=dropout))
            prev_dim = h_dim
        self.blocks = nn.ModuleList(layers)
        self.out_proj = nn.Linear(prev_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.repeat(x.size(0))
        t_emb = self.time_embed(t)
        h = x
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(h)


class TabRepDiffusionModel:
    """
    TabRep tabular generative model combining:
    - Quantile Normalization & CatConverter PSK continuous representations
    - Conditional Flow Matching (CFM) vector field training
    - High-order numerical ODE integration (Heun / Euler)
    """
    def __init__(
        self,
        num_timesteps: int = 50,
        hidden_dims: List[int] = [256, 512, 256],
        dropout: float = 0.1,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.num_timesteps = num_timesteps
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.random_state = random_state

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"TabRep Flow initialized on device: {self.device} (seed: {self.random_state})")
        set_seed(self.random_state)

        self.encoder = TabRepContinuousEncoder(random_state=self.random_state)
        self.network: Optional[MLPFlowNet] = None
        self.continuous_dim: int = 0
        self.num_numerical_features: int = 0
        self.category_sizes: List[int] = []

    def fit(
        self,
        df: pd.DataFrame,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        discrete_cols: Optional[List[str]] = None,
        categorization: Optional[Any] = None,
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
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

        # Transform data into continuous representation
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

        self.network = MLPFlowNet(
            input_dim=self.continuous_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        dataset = TensorDataset(continuous_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=weight_decay)

        self.network.train()
        logger.info(f"Starting TabRep Flow Matching training ({epochs} epochs, batch_size={batch_size})...")

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for (batch_x1,) in dataloader:
                batch_x1 = batch_x1.to(self.device)
                bs = batch_x1.size(0)

                # Prior Gaussian noise x0 ~ N(0, I)
                x0 = torch.randn_like(batch_x1)

                # Uniformly sample flow time t in [0, 1]
                t = torch.rand(bs, device=self.device)

                # Linear probability path: x_t = t * x1 + (1 - t) * x0
                xt = t[:, None] * batch_x1 + (1.0 - t[:, None]) * x0

                # Target velocity field: u_t = x1 - x0
                ut = batch_x1 - x0

                # Predict velocity
                vt = self.network(xt, t)
                loss = F.mse_loss(vt, ut)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * bs

            avg_loss = total_loss / len(dataset)
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                logger.info(f"TabRep Flow Epoch [{epoch}/{epochs}] - CFM Loss: {avg_loss:.5f}")

        logger.info("TabRep Flow Matching training complete.")
        return self

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        steps: Optional[int] = None,
        solver: str = "heun",
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """
        Sample synthetic tabular records by integrating the Flow Matching ODE.
        Supports 'heun' (2nd order Runge-Kutta) and 'euler' (1st order) solvers.
        """
        if self.network is None:
            raise ValueError("TabRep model must be fitted before sampling.")

        self.network.eval()
        N_steps = steps if steps is not None else self.num_timesteps
        logger.info(
            f"Generating {num_samples} synthetic samples using TabRep Flow ODE "
            f"(solver={solver}, steps={N_steps}) on {self.device}..."
        )

        all_num_samples = []
        all_cat_samples = []
        remaining = num_samples

        with tqdm(total=num_samples, desc="TabRep ODE Sampling") as pbar:
            while remaining > 0:
                cur_bs = min(remaining, batch_size)
                # Sample prior noise z0 ~ N(0, I)
                z = torch.randn((cur_bs, self.continuous_dim), device=self.device)
                dt = 1.0 / N_steps

                for step_idx in range(N_steps):
                    t_curr = step_idx / N_steps
                    t_vec = torch.full((cur_bs,), t_curr, device=self.device)
                    vt = self.network(z, t_vec)

                    if solver == "heun":
                        # Heun predictor-corrector step (2nd order Runge-Kutta)
                        t_next = min((step_idx + 1) / N_steps, 1.0)
                        t_next_vec = torch.full((cur_bs,), t_next, device=self.device)
                        z_pred = z + vt * dt
                        vt_next = self.network(z_pred, t_next_vec)
                        vt = 0.5 * (vt + vt_next)

                    z = z + vt * dt

                # Split continuous latent vector into numerical and categorical PSK components
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
