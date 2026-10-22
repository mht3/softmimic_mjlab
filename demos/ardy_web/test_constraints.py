import unittest
from types import SimpleNamespace
import numpy as np
import torch
from demos.ardy_web.constraints import validate_constraints,ConstraintAdapter,window_conditions
from demos.ardy_web.streaming import validate_request

class RequestConstraints(unittest.TestCase):
 def test_times_duplicates_conflicts_and_request_preservation(self):
  c={'type':'root2d','time':4,'position':[.2,.1]}
  self.assertEqual(validate_request({'prompt':'walk','seconds':4,'constraints':[c]})['constraints'][0]['frame'],99)
  for value in [[c,c],[dict(c,time=5)],[dict(c,position=[float('nan'),0])],[dict(c,type='unknown')]]:
   with self.assertRaises(ValueError):validate_constraints(value,4)
  pose=[0,0,.8,1,0,0,0]+[0]*23
  with self.assertRaisesRegex(ValueError,'conflicts'):validate_constraints([dict(c,time=1),{'type':'left-hand','time':1,'pose':pose}],4)
 def test_lookahead_and_history_mask(self):
  m=torch.zeros(1,750,2);o=m.clone();m[0,600]=1;o[0,600]=7
  n,a,b=window_conditions(m,o,8,4,8,4,25);self.assertEqual(n,248);self.assertFalse(a.any())
  n,a,b=window_conditions(m,o,592,4,8,4,25);self.assertEqual(n,16);self.assertTrue(a[0,12].all());self.assertEqual(b[0,12,0],7)
  self.assertFalse(a[:,:4].any());self.assertIsNone(window_conditions(m,o,608,4,8,4,25)[1])
 def test_padding_near_last_frame(self):
  m=torch.zeros(1,50,3);m[0,49]=1
  n,a,b=window_conditions(m,m,48,4,8,4,25);self.assertEqual(n,12);self.assertEqual(a.shape,(1,12,3));self.assertTrue(a[0,5].all());self.assertFalse(a[0,6:].any())

class NativePoseConstraints(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from demos.ardy_web.engine import Engine
  cls.engine=Engine();cls.adapter=ConstraintAdapter(cls.engine)
  from pathlib import Path
  p=Path(__file__).resolve().parents[2]/'datasets/streamer/stand.csv'
  row=np.loadtxt(p,delimiter=',')[0];cls.q=np.r_[row[:3],row[6],row[3:6],row[7:]]
 def test_named_pose_roundtrip_and_coordinate_conversion(self):
  q=self.q.copy();q[:2]=[1,2];pos,rot=self.adapter.pose(q)
  np.testing.assert_allclose(pos[0,0].numpy(),[2,q[2],1],atol=1e-6)
  s=self.engine.model.skeleton;local=s.global_rots_to_local_rots(rot)
  result=np.asarray(self.engine.converter.dict_to_qpos({'local_rot_mats':local[None],'root_positions':pos[:,0][None]}))[0,0]
  expected=np.zeros(36);expected[:7]=q[:7];expected[7+np.array(self.engine.retargeter.mapping)]=q[7:]
  np.testing.assert_allclose(result,expected,atol=2e-6)
 def test_all_six_native_constraint_types_have_finite_condition_masks(self):
  items=[{'type':'root2d','time':.4,'position':[.1,.2]},{'type':'fullbody','time':.8,'pose':self.q.tolist()}]
  for i,kind in enumerate(['left-hand','right-hand','left-foot','right-foot']):items.append({'type':kind,'time':1.2+i*.4,'pose':self.q.tolist(),'offset':[.01,0,0]})
  items=validate_constraints(items,4);m,o=self.adapter.conditions(items,100)
  self.assertTrue(torch.isfinite(o).all());self.assertEqual(m.shape,(1,100,414))
  for x in items:self.assertTrue(m[0,x['frame']].any())
  self.assertTrue(torch.all(o[m==0]==0))

if __name__=='__main__':unittest.main()
