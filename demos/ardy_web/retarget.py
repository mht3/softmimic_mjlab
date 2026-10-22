"""Named 29-to-23 DOF, collision-constrained kinematic retargeting (not control)."""
from pathlib import Path
import hashlib
import itertools
import json
import sys
import xml.etree.ElementTree as ET
import mink
import mujoco
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'softmimic_mjlab'))
from compliant_motion_augmentation.collision_checks import self_collision_pairs, collision_free_step
TARGET = ROOT/'softmimic_mjlab/src/assets/robots/unitree_g1/xmls/g1_23dof.xml'
AUDIT = ROOT/'softmimic_mjlab/softmimic_deploy/src/assets/g1/g1_23dof.xml'
SOURCE = ROOT/'ardy/ardy/assets/skeletons/g1skel34/xml/g1.xml'
FEET = ['left_ankle_roll_link','right_ankle_roll_link']


def collision_limit(model):
    geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g] and (model.geom_contype[g] or model.geom_conaffinity[g])]
    excluded={((int(s)>>16),(int(s)&65535)) for s in model.exclude_signature}
    pairs=[([a],[b]) for a,b in itertools.combinations(geoms,2)
           if tuple(sorted((int(model.geom_bodyid[a]),int(model.geom_bodyid[b])))) not in excluded]
    return mink.CollisionAvoidanceLimit(model,geom_pairs=pairs,minimum_distance_from_collisions=.003,collision_detection_distance=.04)


class OtherGeometryLimit:
    """Constrain the independent geometry using its own FK and joint Jacobians.

    Both models have the same named generalized coordinates, but their bodies
    and geom indices differ; never pass one model's data to the other's limit.
    """
    def __init__(self,model):
        self.configuration=mink.Configuration(model)
        self.limit=collision_limit(model)
    def compute_qp_inequalities(self,configuration,dt):
        self.configuration.update(configuration.q)
        return self.limit.compute_qp_inequalities(self.configuration,dt)


def load_model(path, target=False):
    root = ET.parse(path).getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir', str((path.parent/compiler.get('meshdir','.')).resolve()))
    for side in ('left','right'):
        name = f'{side}_wrist_roll_rubber_hand' if target else f'{side}_wrist_yaw_link'
        body = root.find(f'.//body[@name="{name}"]')
        # Same anatomical point: 38 mm + 46 mm of removed wrist + 80 mm palm.
        existing = root.find(f'.//site[@name="{side}_palm"]')
        if existing is None:
            ET.SubElement(body,'site',name=f'{side}_palm',pos='.164 0 0' if target else '.08 0 0',size='.002')
    if target:
        ET.SubElement(root.find('worldbody'),'geom',name='retarget_ground',type='plane',size='0 0 .1',contype='1',conaffinity='1')
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))


class Retargeter:
    def __init__(self):
        self.src = load_model(SOURCE)
        self.sd = mujoco.MjData(self.src)
        self.model = load_model(TARGET, True)
        self.c = mink.Configuration(self.model)
        self.check = mujoco.MjData(self.model)
        self.audit_model = mujoco.MjModel.from_xml_path(str(AUDIT))
        self.audit_data = mujoco.MjData(self.audit_model)
        self.names = [self.model.joint(j).name for j in range(1,self.model.njnt)]
        self.source_names = [self.src.joint(j).name for j in range(1,self.src.njnt)]
        if len(self.names)!=23 or len(self.source_names)!=29:
            raise ValueError('Unexpected embodiment')
        if [self.audit_model.joint(j).name for j in range(1,self.audit_model.njnt)]!=self.names:
            raise ValueError('Independent collision model joint order differs')
        self.mapping = [self.source_names.index(n) for n in self.names]
        contract = json.loads((Path(__file__).with_name('robot_contract.json')).read_text())
        if self.names != contract['joint_order']:
            raise ValueError('Target ordering changed')
        self.nominal = np.r_[0.,0.,.79,1.,0.,0.,0.,contract['default_joint_position']]
        self.c.update(self.nominal)
        self.tasks = {}
        for name in FEET:
            self.tasks[(name,'body')] = mink.FrameTask(name,'body',position_cost=40.,orientation_cost=5.)
        for name in ('left_palm','right_palm'):
            self.tasks[(name,'site')] = mink.FrameTask(name,'site',position_cost=5.,orientation_cost=.05)
        self.tasks[('pelvis','body')] = mink.FrameTask('pelvis','body',position_cost=10.,orientation_cost=1.)
        self.tasks[('torso_link','body')] = mink.FrameTask('torso_link','body',position_cost=2.,orientation_cost=4.)
        self.posture = mink.PostureTask(self.model,cost=1.)
        geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])]
        self.limits = [mink.ConfigurationLimit(self.model),
                       mink.VelocityLimit(self.model,{n:6. for n in self.names}),
                       collision_limit(self.model),OtherGeometryLimit(self.audit_model),
                       mink.CollisionAvoidanceLimit(self.model,geom_pairs=[([self.model.geom('retarget_ground').id],geoms)],minimum_distance_from_collisions=0.,collision_detection_distance=.04)]

    def _targets(self, q):
        self.sd.qpos[:] = q
        mujoco.mj_forward(self.src,self.sd)
        targets = {}
        for (name,kind), task in self.tasks.items():
            frame = self.sd.body(name) if kind=='body' else self.sd.site(name)
            transform = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(frame.xmat.reshape(3,3)),frame.xpos.copy())
            task.set_target(transform)
            targets[(name,kind)] = frame.xpos.copy()
        posture = np.r_[q[:7],q[7:][self.mapping]]
        posture[7:] = np.clip(posture[7:],self.model.jnt_range[1:,0],self.model.jnt_range[1:,1])
        self.posture.set_target(posture)
        return targets

    def _step(self,dt):
        previous = self.c.q.copy()
        v = mink.solve_ik(self.c,[*self.tasks.values(),self.posture],dt,solver='daqp',damping=1e-4,limits=self.limits)
        self.c.integrate_inplace(v,dt)
        candidate,fraction = collision_free_step(self.model,self.check,previous,self.c.q)
        candidate,fraction2 = collision_free_step(self.audit_model,self.audit_data,previous,candidate)
        self.c.update(candidate)
        return min(fraction,fraction2)

    def _guard_output_segment(self,previous,candidate):
        tangent=np.empty(self.model.nv)
        mujoco.mj_differentiatePos(self.model,tangent,1.,previous,candidate)
        for halving in range(12):
            fraction=.5**halving
            clear=True
            # Include the exact playback audit samples, even for very short
            # steps whose endpoints happen to lie either side of a contact.
            for alpha in (.25,.5,.75,1.):
                middle=previous.copy()
                mujoco.mj_integratePos(self.model,middle,tangent,alpha*fraction)
                for model,data in ((self.model,self.check),(self.audit_model,self.audit_data)):
                    data.qpos[:]=middle;mujoco.mj_forward(model,data)
                    if self_collision_pairs(model,data,0.):clear=False;break
                if not clear:break
            if clear:
                result=previous.copy();mujoco.mj_integratePos(self.model,result,tangent,fraction)
                return result,fraction
        return previous.copy(),0.

    def run(self, q29, fps):
        q29 = np.asarray(q29,dtype=float).copy()
        if q29.ndim!=2 or q29.shape[1]!=36 or len(q29)<2 or not np.isfinite(q29).all() or not 1<=fps<=120:
            raise ValueError('Expected finite Nx36 ARDY qpos and valid fps')
        if not np.allclose(np.linalg.norm(q29[:,3:7],axis=1),1.,atol=1e-3):
            raise ValueError('Invalid WXYZ root quaternion')
        # Preserve the solved collision-avoiding path instead of replacing four
        # IK steps by a straight interpolation between 25 Hz endpoints.
        expanded=[]
        tangent=np.empty(self.src.nv)
        for i in range(len(q29)-1):
            mujoco.mj_differentiatePos(self.src,tangent,1.,q29[i],q29[i+1])
            for alpha in (0.,.25,.5,.75):
                pose=q29[i].copy();mujoco.mj_integratePos(self.src,pose,tangent,alpha)
                expanded.append(pose)
        expanded.append(q29[-1]);q29=np.array(expanded);fps*=4
        self.c.update(self.nominal)
        # Move the valid initialization with the source root XY/yaw before fitting frame 0.
        initial=self.nominal.copy(); initial[:2]=q29[0,:2]; initial[3:7]=q29[0,3:7]
        self.c.update(initial)
        self._targets(q29[0])
        for _ in range(160):
            self._step(.01)
        output=[]; errors=[]; limited=0
        previous_source=q29[0]
        for q in q29:
            last_output=self.c.q.copy()
            # Feed forward the source rigid-root displacement before solving the
            # articulated correction. A blocked hand must not freeze walking's
            # global translation via collision backtracking of the whole state.
            transported=self.c.q.copy()
            transported[:3]+=q[:3]-previous_source[:3]
            inverse=previous_source[3:7]*np.array([1.,-1.,-1.,-1.])
            delta=np.empty(4); rotation=np.empty(4)
            mujoco.mju_mulQuat(delta,q[3:7],inverse)
            mujoco.mju_mulQuat(rotation,delta,transported[3:7])
            transported[3:7]=rotation
            self.c.update(transported)
            targets=self._targets(q)
            limited += int(self._step(1/fps)<1.)
            guarded,fraction=self._guard_output_segment(last_output,self.c.q)
            limited+=int(fraction<1.)
            self.c.update(guarded)
            output.append(self.c.q.copy())
            errors.append([np.linalg.norm((self.c.data.body(n) if k=='body' else self.c.data.site(n)).xpos-p) for (n,k),p in targets.items()])
            previous_source=q
        result=np.array(output)
        report=self.audit(result,fps)
        # Different source/target sole proxies put the ground a few mm apart.
        # Align the entire clip once, preserving its root trajectory and velocities.
        # Large corrections indicate a bad fit and are rejected, not hidden.
        ground_lift=report['max_ground_penetration_m']+.0001 if report['max_ground_penetration_m']>0 else 0.
        if ground_lift>.03:
            raise ValueError(f'Retarget requires excessive ground correction: {ground_lift:.4f} m')
        if ground_lift:
            result[:,2]+=ground_lift
            report=self.audit(result,fps)
        report['constant_ground_alignment_m']=ground_lift
        report['output_fps']=fps
        e=np.array(errors)
        # Conservative upper bounds including the constant ground alignment.
        e+=ground_lift
        report.update(max_foot_tracking_error_m=float(e[:,:2].max()),max_palm_tracking_error_m=float(e[:,2:4].max()),
                      max_pelvis_tracking_error_m=float(e[:,4].max()),collision_limited_substeps=limited,
                      source_dof=29,target_dof=23,source_xml_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      target_xml_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),
                      independent_collision_xml_sha256=hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
                      tracking_error_metric='conservative upper bound including constant ground alignment',
                      method='Named posture prior + palm/foot/pelvis/torso IK; joint/velocity/collision constraints; dual-model path audit',
                      dynamics_validated=False)
        report['fidelity_review_required'] = bool(e[:,:2].max()>.08 or e[:,2:4].max()>.12 or e[:,4].max()>.08)
        if not report['geometry_passed']:
            raise ValueError('Retarget audit failed: '+json.dumps(report))
        return result,report

    def audit(self,qpos,fps):
        collisions=0; paths=0; ground=0.; limit=0.; max_speed=0.
        velocity=np.zeros(self.model.nv)
        for i,q in enumerate(qpos):
            limit=max(limit,float(np.max(np.maximum(self.model.jnt_range[1:,0]-q[7:],q[7:]-self.model.jnt_range[1:,1]))))
            for m,d in ((self.model,self.check),(self.audit_model,self.audit_data)):
                d.qpos[:]=q; mujoco.mj_forward(m,d)
                collisions+=int(bool(self_collision_pairs(m,d,0.)))
                for c in d.contact:
                    if not m.geom_bodyid[c.geom1] or not m.geom_bodyid[c.geom2]:
                        ground=max(ground,-float(c.dist))
            if i:
                mujoco.mj_differentiatePos(self.model,velocity,1/fps,qpos[i-1],q)
                max_speed=max(max_speed,float(np.max(np.abs(velocity[6:]))))
                for alpha in (.25,.5,.75):
                    middle=qpos[i-1].copy(); mujoco.mj_integratePos(self.model,middle,velocity,alpha/fps)
                    for m,d in ((self.model,self.check),(self.audit_model,self.audit_data)):
                        d.qpos[:]=middle; mujoco.mj_forward(m,d)
                        paths+=int(bool(self_collision_pairs(m,d,0.)))
                        for contact in d.contact:
                            if not m.geom_bodyid[contact.geom1] or not m.geom_bodyid[contact.geom2]:
                                ground=max(ground,-float(contact.dist))
        return dict(frames=len(qpos),self_collision_checks_failed=collisions,path_collision_checks_failed=paths,
                    max_ground_penetration_m=ground,max_joint_limit_error_rad=max(0.,limit),max_joint_speed_rad_s=max_speed,
                    geometry_passed=bool(collisions==0 and paths==0 and ground<=.002 and limit<=1e-5 and max_speed<=6.001))
