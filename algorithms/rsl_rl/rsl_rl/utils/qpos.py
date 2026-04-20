from __future__ import annotations

import torch


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a quaternion (w, x, y, z) -> (w, -x, -y, -z)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions in (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to rotation vector (3D)."""
    w = q[..., :1].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    half_angle = torch.acos(w.abs())
    sin_half = torch.sqrt(1.0 - w * w).clamp(min=1e-10)
    scale = 2.0 * half_angle / sin_half
    scale = torch.where(w >= 0, scale, -scale)
    return xyz * scale


def _rotvec_to_quat(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert rotation vector (3D) to quaternion (w, x, y, z)."""
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp(min=1e-10)
    half_angle = angle * 0.5
    w = torch.cos(half_angle)
    xyz = rotvec / angle * torch.sin(half_angle)
    return torch.cat([w, xyz], dim=-1)


def differentiate_qpos(qpos: torch.Tensor, qpos_ref: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of mj_differentiatePos with dt=1."""
    pos_diff = qpos[..., :3] - qpos_ref[..., :3]
    q = qpos[..., 3:7]
    q_ref = qpos_ref[..., 3:7]
    q_rel = _quat_multiply(_quat_conjugate(q_ref), q)
    rotvec_diff = _quat_to_rotvec(q_rel)
    joint_diff = qpos[..., 7:] - qpos_ref[..., 7:]
    return torch.cat([pos_diff, rotvec_diff, joint_diff], dim=-1)


def integrate_qpos(qpos_ref: torch.Tensor, qvel_rel: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of mj_integratePos with dt=1."""
    pos_abs = qpos_ref[..., :3] + qvel_rel[..., :3]
    q_ref = qpos_ref[..., 3:7]
    delta_q = _rotvec_to_quat(qvel_rel[..., 3:6])
    q_abs = _quat_multiply(q_ref, delta_q)
    q_abs = q_abs / torch.linalg.norm(q_abs, dim=-1, keepdim=True)
    joint_abs = qpos_ref[..., 7:] + qvel_rel[..., 6:]
    return torch.cat([pos_abs, q_abs, joint_abs], dim=-1)
