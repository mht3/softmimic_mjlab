"""Self-intersection checks using the generation model's collision geometry.

MuJoCo's configured masks and parent-body filtering define eligible pairs.
Mesh collision shapes are convex proxies; this is a conservative check, not
an exact triangle-mesh or continuous-time collision certificate.
"""
import mujoco
import numpy as np

SELF_COLLISION_TOLERANCE_M = .002


def self_collision_pairs(model, data, tolerance=SELF_COLLISION_TOLERANCE_M):
    """Return maximum penetration per robot body pair deeper than tolerance.

    Call mj_forward/mj_collision after setting qpos before using this function.
    Floor/world contacts are excluded. Multiple points on one pair are merged.
    """
    pairs = {}
    for contact in data.contact:
        if contact.dist >= -tolerance:
            continue
        first, second = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        if not first or not second or first == second:
            continue
        names = sorted(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in (first, second))
        key = ' / '.join(names)
        pairs[key] = max(pairs.get(key, 0.), -float(contact.dist))
    return pairs


def collision_free_step(model, data, previous, candidate, max_joint_step=.025):
    """Backtrack a finite IK step and check its interpolated path.

    Differential IK constraints are linearized at the old configuration; they
    cannot guarantee collision-free finite integration. Check the complete
    emitted transition, including release decay and velocity limiting. The
    sampling resolution is a discrete safeguard, not continuous collision
    detection. Returning the old pose is preferable to crossing the body.
    """
    tangent = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, tangent, 1., previous, candidate)
    def clear(q):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_collision(model, data)
        return not self_collision_pairs(model, data, tolerance=0.)
    if not clear(previous):
        raise ValueError('Cannot backtrack from an already self-colliding pose')
    for halving in range(11):
        fraction = .5 ** halving
        step = tangent * fraction
        # Free-root translation needs a distance bound rather than an angle.
        free = model.njnt and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        count = max(1, int(np.ceil(np.max(np.abs(step[3:] if free else step), initial=0.) / max_joint_step)))
        if free:
            count = max(count, int(np.ceil(np.linalg.norm(step[:3]) / .005)))
        accepted = True
        for phase in np.linspace(1., 1./count, count):
            q = previous.copy()
            mujoco.mj_integratePos(model, q, step, float(phase))
            if not clear(q):
                accepted = False
                break
        if accepted:
            q = previous.copy()
            mujoco.mj_integratePos(model, q, step, 1.)
            return q, fraction
    return previous.copy(), 0.
