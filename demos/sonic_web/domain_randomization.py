"""CPU counterpart of js/randomization.js, for paired deployment stress tests.

Matches training parameter distributions, not GPU random streams or reset timing.
The browser/native parity test checks these mutations on the same compiled scene.
"""
MASS_BODIES = ('pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link',
               'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link')


def apply_draw(model, seed=42, mass=True, actuators=True, limits=True, friction=True,
               training_contacts=True):
    import mujoco
    import re
    if not isinstance(seed, int) or not 0 <= seed <= 0xffffffff:
        raise ValueError('Seed must be an unsigned 32-bit integer')
    def u(lo, hi):
        nonlocal seed
        seed = (1664525*seed+1013904223) & 0xffffffff
        return lo+(hi-lo)*seed/4294967296
    def change_mass(i, value):
        model.body_inertia[i] *= value/model.body_mass[i]
        model.body_mass[i] = value
    torso = model.body('torso_link').id
    payload = u(-2, 2)
    if mass: change_mass(torso, model.body_mass[torso]+payload)
    for name in MASS_BODIES:
        i=model.body(name).id; scale=u(.7, 1.3)
        if mass: change_mass(i, model.body_mass[i]*scale)
    for j in range(3):
        offset=u(-.02, .02)
        if mass: model.body_ipos[torso,j] += offset
    for i in range(model.nu):
        kd=u(0,2)
        if actuators: model.actuator_biasprm[i,2] -= kd
    for i in range(6,model.nv):
        armature=u(.01,.1)
        if actuators: model.dof_armature[i] += armature
    for i in range(6,model.nv):
        loss=u(0,.01)
        if actuators: model.dof_frictionloss[i] += loss
    for i in range(1,model.njnt):
        for j in range(2):
            offset=u(-.02,.02)
            if limits: model.jnt_range[i,j] += offset
    foot=u(.5,2)
    for i in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
        if not name.endswith('_collision'): continue
        is_foot=bool(re.fullmatch('(left|right)_foot[1-7]_collision',name))
        if is_foot and friction: model.geom_friction[i,0]=foot
        if training_contacts and not is_foot: model.geom_condim[i]=1
    # Caller recomputes model constants and resets data before stepping.
    return dict(payload_kg=payload if mass else 0,foot_friction=foot if friction else None)
