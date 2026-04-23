# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for qpos differentiation/integration roundtrip (used by set_reset_states)."""

from __future__ import annotations

import torch
import pytest

from rsl_rl.utils.qpos import differentiate_qpos, integrate_qpos


def _random_qpos(batch: int, n_joints: int, device: str = "cpu") -> torch.Tensor:
    """Generate random qpos: [pos(3), quat(4), joints(n_joints)]."""
    pos = torch.randn(batch, 3, device=device)
    # Random unit quaternions (wxyz convention).
    raw = torch.randn(batch, 4, device=device)
    quat = raw / torch.linalg.norm(raw, dim=-1, keepdim=True)
    # Ensure positive w for consistency.
    quat = torch.where(quat[:, :1] < 0, -quat, quat)
    joints = torch.randn(batch, n_joints, device=device) * 0.5
    return torch.cat([pos, quat, joints], dim=-1)


class TestDifferentiateIntegrateRoundtrip:
    """differentiate_qpos and integrate_qpos should be exact inverses."""

    @pytest.mark.parametrize("n_joints", [0, 6, 23])
    def test_roundtrip_identity(self, n_joints: int) -> None:
        """integrate(ref, differentiate(qpos, ref)) == qpos."""
        batch = 16
        qpos = _random_qpos(batch, n_joints)
        qpos_ref = _random_qpos(batch, n_joints)

        rel = differentiate_qpos(qpos, qpos_ref)
        reconstructed = integrate_qpos(qpos_ref, rel)

        # Position and joints should match exactly.
        torch.testing.assert_close(reconstructed[:, :3], qpos[:, :3], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reconstructed[:, 7:], qpos[:, 7:], atol=1e-5, rtol=1e-5)

        # Quaternions: check that they represent the same rotation (q == -q).
        q_orig = qpos[:, 3:7]
        q_recon = reconstructed[:, 3:7]
        # Dot product of unit quaternions: |dot| == 1 means same rotation.
        dots = (q_orig * q_recon).sum(dim=-1).abs()
        torch.testing.assert_close(dots, torch.ones(batch), atol=1e-4, rtol=0)

    def test_zero_relative_state_is_identity(self) -> None:
        """integrate(ref, zeros) == ref (with normalized quaternion)."""
        batch = 8
        n_joints = 10
        qpos_ref = _random_qpos(batch, n_joints)
        nv = 3 + 3 + n_joints  # pos(3) + rotvec(3) + joints
        zero_rel = torch.zeros(batch, nv)

        result = integrate_qpos(qpos_ref, zero_rel)

        torch.testing.assert_close(result[:, :3], qpos_ref[:, :3], atol=1e-6, rtol=0)
        torch.testing.assert_close(result[:, 7:], qpos_ref[:, 7:], atol=1e-6, rtol=0)
        # Quaternion should be unchanged (normalized).
        dots = (result[:, 3:7] * qpos_ref[:, 3:7]).sum(dim=-1).abs()
        torch.testing.assert_close(dots, torch.ones(batch), atol=1e-5, rtol=0)

    def test_differentiate_same_gives_zeros(self) -> None:
        """differentiate_qpos(q, q) == zeros."""
        batch = 8
        n_joints = 23
        qpos = _random_qpos(batch, n_joints)
        rel = differentiate_qpos(qpos, qpos)
        torch.testing.assert_close(rel, torch.zeros_like(rel), atol=1e-5, rtol=0)
