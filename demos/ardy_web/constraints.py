"""Browser Z-up keyframes → ARDY's native Y-up constraint classes.

Uses the same condition normalization/history masking as upstream
scripts/interactive_demo/{gen_constraints,generation,window_budget}.py.
The qpos import follows upstream motion_io.py, with metres/radians/WXYZ.
"""
import math
import xml.etree.ElementTree as ET
import numpy as np

TYPES=('root2d','fullbody','left-hand','right-hand','left-foot','right-foot')

def validate_constraints(value,seconds):
    if not isinstance(value,list) or len(value)>100:raise ValueError('Use at most 100 constraint keyframes')
    clean=[];seen=set();counts={}
    def vector(v,n,label,limit):
        if not isinstance(v,list) or len(v)!=n or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or abs(x)>limit for x in v):raise ValueError('Invalid '+label)
        return [float(x) for x in v]
    for item in value:
        if not isinstance(item,dict) or item.get('type') not in TYPES:raise ValueError('Unknown constraint track')
        kind=item['type'];t=item.get('time')
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or not 0<=t<=seconds:raise ValueError('Constraint time lies outside generation')
        frame=min(round(t*25),round(seconds*25)-1);key=(kind,frame)
        if key in seen:raise ValueError('Two keyframes overlap on the same constraint track')
        seen.add(key);counts[kind]=counts.get(kind,0)+1
        if counts[kind]>20:raise ValueError('Use at most 20 keyframes per track')
        result=dict(type=kind,time=frame/25,frame=frame)
        if kind=='root2d':result['position']=vector(item.get('position'),2,'root XY target',20)
        else:
            q=vector(item.get('pose'),30,'captured 23DOF pose',20)
            if abs(np.linalg.norm(q[3:7])-1)>1e-5:raise ValueError('Captured pose quaternion must be normalized')
            result['pose']=q
            if kind!='fullbody':result['offset']=vector(item.get('offset',[0,0,0]),3,'end-effector XYZ offset',2)
        clean.append(result)
    # ARDY's named hand/foot sets also constrain the captured pelvis. Reject
    # contradictory pelvis/root goals instead of silently overwriting a mask.
    for frame in {x['frame'] for x in clean}:
        at=[x for x in clean if x['frame']==frame];poses=[x['pose'][:7] for x in at if 'pose' in x]
        if poses and any(not np.allclose(p,poses[0],atol=1e-5,rtol=0) for p in poses):raise ValueError('Conflicting captured pelvis poses at the same time')
        roots=[x['position'] for x in at if x['type']=='root2d']
        if poses and roots and not np.allclose(poses[0][:2],roots[0],atol=1e-5,rtol=0):raise ValueError('Root waypoint conflicts with the captured pose at the same time')
        if any(x['type']=='fullbody' for x in at) and any(x.get('offset') and np.linalg.norm(x['offset'])>1e-8 for x in at):raise ValueError('Full-body and moved end-effector targets conflict at the same time')
    return sorted(clean,key=lambda x:(x['frame'],x['type']))

class ConstraintAdapter:
    def __init__(self,engine):
        self.engine=engine;self.skeleton=engine.model.skeleton
        tree=ET.parse(engine.converter.xml_path);root=tree.getroot()
        defaults={d.get('class'):d.find('joint').get('axis') for d in root.findall('.//default') if d.get('class') and d.find('joint') is not None}
        parents={c:p for p in root.iter() for c in p}
        self.spec=[]
        for j in root.find('worldbody').findall('.//joint'):
            axis=[float(x) for x in (j.get('axis') or defaults[j.get('class')]).split()]
            quat=[float(x) for x in parents[j].get('quat','1 0 0 0').split()]
            self.spec.append((j.get('name'),self.skeleton.bone_index[j.get('name').replace('_joint','_skel')],axis,quat))
        self.names=engine.retargeter.names
        self.C=np.array([[0,1,0],[0,0,1],[1,0,0]],dtype=float)

    def pose(self,q):
        import torch
        from scipy.spatial.transform import Rotation
        q=np.asarray(q)
        limits=self.engine.retargeter.model.jnt_range[1:]
        if np.any(q[7:]<limits[:,0]-1e-5) or np.any(q[7:]>limits[:,1]+1e-5):raise ValueError('Captured pose exceeds 23DOF joint limits')
        joints=dict(zip(self.names,q[7:]));local=np.tile(np.eye(3),(1,self.skeleton.nbjoints,1,1))
        local[0,0]=self.C@Rotation.from_quat(q[[4,5,6,3]]).as_matrix()@self.C.T
        for name,index,axis,quat in self.spec:
            extra=Rotation.from_quat([*quat[1:],quat[0]])
            r=extra*Rotation.from_rotvec(np.array(axis)*joints.get(name,0.))
            local[0,index]=self.C@r.as_matrix()@self.C.T
        local=torch.tensor(local,dtype=torch.float32,device=self.engine.device)
        root=torch.tensor((self.C@q[:3])[None],dtype=torch.float32,device=self.engine.device)
        rotations,positions,_=self.skeleton.fk(local,root)
        return positions,rotations

    def conditions(self,items,total):
        import torch
        from ardy.constraints import TYPE_TO_CLASS,Root2DConstraintSet,FullBodyConstraintSet
        sets=[];s=self.skeleton;device=self.engine.device
        for item in items:
            indices=torch.tensor([item['frame']],dtype=torch.long)
            if item['type']=='root2d':
                x,y=item['position'];sets.append(Root2DConstraintSet(s,indices,torch.tensor([[y,x]],dtype=torch.float32,device=device)));continue
            pos,rot=self.pose(item['pose'])
            if item['type']=='fullbody':sets.append(FullBodyConstraintSet(s,indices,pos,rot));continue
            group={'left-hand':'LeftHand','right-hand':'RightHand','left-foot':'LeftFoot','right-foot':'RightFoot'}[item['type']]
            _,bones=s.expand_joint_names([group]);delta=torch.tensor(self.C@np.array(item['offset']),dtype=torch.float32,device=device)
            pos=pos.clone();pos[:,[s.bone_index[b] for b in bones]]+=delta
            sets.append(TYPE_TO_CLASS[item['type']](s,indices,pos,rot,root_2d=None))
        if not sets:return None,None
        rep=self.engine.model.motion_rep
        observed,mask=rep.create_conditions_from_constraints(sets,length=total,to_normalize=False,device=device)
        observed=rep.normalize(observed)*mask
        return mask[None],observed[None]

    def anchors(self,q):
        pos,_=self.pose(q);pos=pos[0].detach().cpu().numpy()@self.C
        result={}
        for kind,attr in [('left-hand','left_hand_joint_names'),('right-hand','right_hand_joint_names'),('left-foot','left_foot_joint_names'),('right-foot','right_foot_joint_names')]:
            result[kind]=pos[self.skeleton.bone_index[getattr(self.skeleton,attr)[0]]].tolist()
        return result


def window_conditions(mask,observed,generated,history,horizon,token,fps):
    """Native demo's ten-second bounded future context; past goals are masked."""
    if mask is None:return history+horizon,None,None
    start=generated-history;max_window=int(10*fps)//token*token
    future=(mask[0,generated:].any(dim=-1)).nonzero().flatten()
    if not len(future):return history+horizon,None,None
    last=generated+int(future[-1]);size=max(history+horizon,last-start+1)
    size=max(history+horizon,min(math.ceil(size/token)*token,max_window))
    import torch
    m=mask[:,start:start+size].clone();o=observed[:,start:start+size].clone()
    if m.shape[1]<size:
        pad=size-m.shape[1];m=torch.cat([m,torch.zeros_like(m[:,:1]).expand(-1,pad,-1)],dim=1);o=torch.cat([o,torch.zeros_like(o[:,:1]).expand(-1,pad,-1)],dim=1)
    m[:,:history]=0;o[:,:history]=0
    return size,m,o


def constraint_fit(adapter,items,q23,fps):
    """Report realized kinematic goal errors separately from geometry acceptance."""
    report=[]
    for goal in items:
        frame=round(goal['time']*fps)
        if frame>=len(q23):continue  # future goal in a progressive preview
        q=q23[frame];kind=goal['type'];row=dict(type=kind,time=goal['time'])
        if kind=='root2d':
            row.update(position_error_m=float(np.linalg.norm(q[:2]-goal['position'])),review_threshold_m=.10)
        elif kind=='fullbody':
            row.update(position_error_m=float(np.linalg.norm(q[:3]-np.array(goal['pose'][:3]))),review_threshold_m=.08,max_joint_error_rad=float(np.max(np.abs(q[7:]-np.array(goal['pose'][7:])))))
        else:
            target=np.array(adapter.anchors(goal['pose'])[kind])+goal['offset'];actual=np.array(adapter.anchors(q)[kind])
            row.update(position_error_m=float(np.linalg.norm(actual-target)),review_threshold_m=.12 if 'hand' in kind else .08)
        row['needs_review']=row['position_error_m']>row['review_threshold_m'] or row.get('max_joint_error_rad',0)>.2
        report.append(row)
    return report
