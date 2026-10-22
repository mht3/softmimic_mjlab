import unittest
import numpy as np
from demos.ardy_web.retarget import Retargeter
from demos.ardy_web.engine import presets

class RetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r=Retargeter()
    def source(self,count=16):
        r=self.r;q=np.r_[r.nominal[:7],np.zeros(29)];q[7+np.array(r.mapping)]=r.nominal[7:]
        return np.tile(q,(count,1))
    def test_named_embodiment_and_moving_reference(self):
        q=self.source();q[:,0]=np.linspace(0,.03,len(q))
        q[:,7+self.r.source_names.index('waist_pitch_joint')]=np.linspace(0,.08,len(q))
        out,report=self.r.run(q,25.)
        self.assertEqual(out.shape,(61,30));self.assertEqual(report['output_fps'],100.);self.assertTrue(report['geometry_passed'])
        self.assertGreater(np.ptp(out[:,0]),.015)
        self.assertLess(report['max_foot_tracking_error_m'],.01)
        self.assertFalse(report['fidelity_review_required'])
        np.testing.assert_allclose(np.linalg.norm(out[:,3:7],axis=1),1,atol=1e-6)
    def test_invalid_source_rejected(self):
        q=self.source();q[0,3:7]=0
        with self.assertRaises(ValueError):self.r.run(q,25.)
        q=self.source();q[1,10]=np.nan
        with self.assertRaises(ValueError):self.r.run(q,25.)
    def test_audit_rejects_limit_violation(self):
        q=np.tile(self.r.nominal,(2,1));q[:,7]=self.r.model.jnt_range[1,1]+.2
        self.assertFalse(self.r.audit(q,25.)['geometry_passed'])
    def test_presets_are_upstream(self):
        self.assertEqual(len(presets()),10)
        self.assertIn('A person bows down and then stands upright.',presets())

if __name__=='__main__':unittest.main()
