# @2026 Matthew Taylor

"""Conditional Flow Matching (CFM) modules for visitation critic."""

from __future__ import annotations
import torch
import torch.nn as nn

from rsl_rl.modules import MLP


class MLPConditionalVectorField(nn.Module):
    """MLP vector field with discrete class conditioning via embedding.

    Predicts u_theta(x, t, y) where x is the state, t is time, and y is a discrete label.
    Uses an embedding layer for the class label and concatenates [x, emb(y), t] as input.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        class_dim: int = 8,
        activation: str = "swish",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_classes = num_classes
        # +1 for the null label used in classifier-free guidance
        self.class_embedding = nn.Embedding(num_classes + 1, class_dim)
        # Input: [x, emb(y), t] -> output: state_dim
        input_dim = state_dim + class_dim + 1
        self.mlp = MLP(input_dim, state_dim, hidden_dims, activation)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: State tensor, shape (B, state_dim).
            t: Time tensor, shape (B,).
            y: Discrete label tensor, shape (B,) with integer class indices.

        Returns:
            Predicted vector field, shape (B, state_dim).
        """
        y_emb = self.class_embedding(y)
        xyt = torch.cat([x, y_emb, t.unsqueeze(-1)], dim=-1)
        return self.mlp(xyt)


class MLPContinuousConditionalVectorField(nn.Module):
    """MLP vector field with continuous conditioning.

    Predicts u_theta(x, t, c) where x is the state, t is time, and c is a continuous condition
    (e.g., reward value or end state). Uses a fixed out-of-range null embedding for
    classifier-free guidance.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: list[int],
        cond_dim: int,
        null_value: float = -1.0,
        activation: str = "swish",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.cond_dim = cond_dim
        self.register_buffer("null_embedding", torch.full((cond_dim,), null_value))
        # Input: [x, c, t] -> output: state_dim
        input_dim = state_dim + cond_dim + 1
        self.mlp = MLP(input_dim, state_dim, hidden_dims, activation)

    def get_null_embedding(self, batch_size: int) -> torch.Tensor:
        """Return the null condition expanded to batch size."""
        return self.null_embedding.unsqueeze(0).expand(batch_size, -1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: State tensor, shape (B, state_dim).
            t: Time tensor, shape (B,).
            c: Continuous condition tensor, shape (B, cond_dim).

        Returns:
            Predicted vector field, shape (B, state_dim).
        """
        xct = torch.cat([x, c, t.unsqueeze(-1)], dim=-1)
        return self.mlp(xct)


# --- Probability Path ---


class GaussianConditionalProbabilityPath:
    """Linear Gaussian conditional probability path for flow matching.

    Implements the path x_t = alpha(t) * z + beta(t) * eps, where:
    - alpha(t) = t (linear schedule)
    - beta(t) = 1 - t
    - z is the data sample
    - eps ~ N(0, I) is Gaussian noise
    """

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_t from the conditional path p_t(x | z).

        Args:
            z: Data samples, shape (B, D).
            t: Time values in [0, 1], shape (B,).

        Returns:
            Interpolated samples x_t, shape (B, D).
        """
        alpha_t = t.unsqueeze(-1)  # (B, 1)
        beta_t = (1.0 - t).unsqueeze(-1)  # (B, 1)
        eps = torch.randn_like(z)
        return alpha_t * z + beta_t * eps

    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute the reference conditional vector field u_t(x | z).

        For linear alpha/beta: u_t(x | z) = (z - x) / (1 - t).
        We use the closed-form: dt_alpha * z + dt_beta / beta * (x - alpha * z)
        which equals: z + (-1) / (1 - t) * (x - t * z) = (z - x) / (1 - t).

        Args:
            x: Current position x_t, shape (B, D).
            z: Data sample, shape (B, D).
            t: Time values in [0, 1), shape (B,).

        Returns:
            Reference vector field, shape (B, D).
        """
        alpha_t = t.unsqueeze(-1)
        beta_t = (1.0 - t).unsqueeze(-1)
        # dt_alpha = 1, dt_beta = -1
        # u = dt_alpha * z + (dt_beta / beta) * (x - alpha * z)
        #   = z + (-1 / (1-t)) * (x - t*z)
        #   = z - (x - t*z) / (1-t)
        #   = (z*(1-t) - x + t*z) / (1-t)
        #   = (z - x) / (1-t)
        return (z - x) / beta_t


# --- ODE Solver ---


class EulerODESolver:
    """Forward Euler ODE solver for generating samples from a learned vector field.

    Integrates from t=0 (noise) to t=1 (data) using the learned vector field.
    """

    def solve(
        self,
        x0: torch.Tensor,
        vector_field_fn: callable,
        num_steps: int = 100,
        **condition_kwargs,
    ) -> torch.Tensor:
        """Solve the ODE from noise (t=0) to data (t=1).

        Args:
            x0: Initial noise samples, shape (B, D).
            vector_field_fn: Callable(x, t, **kwargs) -> velocity, where x is (B, D), t is (B,).
            num_steps: Number of Euler integration steps.
            **condition_kwargs: Additional conditioning arguments passed to vector_field_fn.

        Returns:
            Generated samples at t=1, shape (B, D).
        """
        dt = 1.0 / num_steps
        x = x0
        for i in range(num_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v = vector_field_fn(x, t, **condition_kwargs)
            x = x + v * dt
        return x


# --- Classifier-Free Guidance ---


def cfg_guided_velocity(
    model: MLPConditionalVectorField | MLPContinuousConditionalVectorField,
    x: torch.Tensor,
    t: torch.Tensor,
    guidance_scale: float,
    cond: torch.Tensor | None = None,
    null_cond: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute classifier-free guided velocity.

    u_cfg = (1 - w) * u_null + w * u_cond, where w is the guidance_scale.

    Args:
        model: The conditional vector field network.
        x: State tensor, shape (B, state_dim).
        t: Time tensor, shape (B,).
        guidance_scale: CFG guidance strength. 1.0 = no extra guidance.
        cond: Condition tensor (discrete labels or continuous values).
        null_cond: Null/unconditional condition (null label or null embedding).

    Returns:
        Guided velocity, shape (B, state_dim).
    """
    u_cond = model(x, t, cond)
    u_null = model(x, t, null_cond)
    return (1.0 - guidance_scale) * u_null + guidance_scale * u_cond
