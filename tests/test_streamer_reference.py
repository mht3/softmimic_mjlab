"""CPU contract checks between the editor, IK importer and standing deployment."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'demos/sonic_web'))
from prepare_streamer_reference import prepare,read_csv
from reference import read_reference,validate_reference

class StreamerReferenceTest(unittest.TestCase):
 def test_saved_stand_roundtrip_and_ik_layout(self):
  source=ROOT/'datasets/streamer/stand.csv';data,meta=read_csv(source)
  with tempfile.TemporaryDirectory() as tmp:
   dest=Path(tmp)/'stand.csv';prepare(source,dest,standing=True,duration=1)
   ik=np.loadtxt(dest,delimiter=',');self.assertEqual(ik.shape,(31,32))
   np.testing.assert_allclose(ik[:,:30],np.repeat(data[:1],31,axis=0),atol=1e-9)
   self.assertTrue(np.all(ik[:,30:]==1))
   ref=read_reference(dest);np.testing.assert_allclose(ref['root_quat'][0],data[0,[6,3,4,5]])
 def test_moving_animation_and_named_order_rejected(self):
  source=ROOT/'datasets/streamer/stand.csv';data,meta=read_csv(source)
  with tempfile.TemporaryDirectory() as tmp:
   dest=Path(tmp)/'source.csv';data[-1,0]+=.1
   np.savetxt(dest,data,delimiter=',',header=json.dumps(meta))
   with self.assertRaisesRegex(ValueError,'fixed pose'):prepare(dest,Path(tmp)/'out.csv',standing=True)
   meta['joint_names'].reverse();np.savetxt(dest,data,delimiter=',',header=json.dumps(meta))
   with self.assertRaisesRegex(ValueError,'joint order'):read_reference(dest)
 def test_matching_standing_contract_and_wrong_height(self):
  ref=read_reference(ROOT/'datasets/streamer/stand.csv');j=ref['joint_pos'][0];q=ref['root_quat'][0];w,x,y,z=q
  values={'reference_dof_pos':j,'reference_root_height':[ref['root_pos'][0][2]],'reference_gravity_vector':[2*(w*y-x*z),-2*(y*z+w*x),2*(x*x+y*y)-1],'reference_xy_vel':[0,0],'reference_yaw_vel':[0],'reference_foot_contact':[1,1]}
  c={'reference_mode':'constant_nominal','observations':{k:{'params':{'value':v}} for k,v in values.items()}}
  validate_reference(ref,c);bad=copy.deepcopy(c);bad['observations']['reference_root_height']['params']['value'][0]+=.1
  with self.assertRaisesRegex(ValueError,'reference_root_height'):validate_reference(ref,bad)

if __name__=='__main__':unittest.main()
