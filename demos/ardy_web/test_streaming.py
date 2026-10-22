import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from demos.ardy_web.streaming import validate_request, generate_stream, Cancelled


class RequestTests(unittest.TestCase):
    def test_thirty_seconds_and_schedule_bounds(self):
        self.assertEqual(validate_request({'prompt':'stand','seconds':30})['seconds'],30)
        self.assertEqual(validate_request({'segments':[{'prompt':'walk','seconds':26},{'prompt':'stand','seconds':4}]})['seconds'],30)
        for body in [{'prompt':'x','seconds':float('nan')},{'prompt':'x','seconds':31},
                     {'prompt':'x','seconds':True},{'prompt':'x','seed':False},[],
                     {'segments':[{'prompt':'walk','seconds':30},{'prompt':'stand','seconds':4}]}]:
            with self.subTest(body=body),self.assertRaises(ValueError):validate_request(body)

    def test_history_continuity_prompt_switch_and_final_trim(self):
        calls=[]
        class Model:
            motion_rep=SimpleNamespace(fps=25,inverse=lambda x,**kw:x)
            gen_horizon_len=8;num_frames_per_token=4;diffusion=SimpleNamespace(num_base_steps=10)
            def autoregressive_step(self,**kw):
                history=kw['init_history_sequence'];start=0 if history is None else int(history[0,-1,0])+1
                calls.append((start,int(kw['text_feat'].item()),None if history is None else history.clone()))
                new=torch.zeros(1,8,36);new[0,:,0]=torch.arange(start,start+8);new[:,:,3]=1
                return new if history is None else torch.cat([history,new],dim=1)
        engine=SimpleNamespace(model=Model(),device='cpu',encode_prompt=lambda p,*a:torch.tensor(0 if p=='walk' else 1),
                               converter=SimpleNamespace(dict_to_qpos=lambda x,**kw:x.numpy()),retargeter=SimpleNamespace(source_names=[]))
        snapshots=[]
        def export(folder,q,*args,**kw):
            snapshots.append(q.copy());return {'id':folder.name}
        job=validate_request({'segments':[{'prompt':'walk','seconds':2},{'prompt':'stand','seconds':28}]})
        with tempfile.TemporaryDirectory() as tmp,patch('demos.ardy_web.export.export_motion',export):
            previews=[]
            generate_stream(engine,job,Path(tmp)/'test',lambda x:None,previews.append,lambda:False)
            self.assertTrue(previews[-1]['final']);self.assertEqual(previews[-1]['seconds'],30)
            self.assertEqual(snapshots[-1].shape,(750,36))
            np.testing.assert_array_equal(snapshots[-1][:,0],np.arange(750))
            for q in snapshots:np.testing.assert_array_equal(q,snapshots[-1][:len(q)])
            self.assertEqual(next(start for start,p,h in calls if p==1),56) # next 8-frame boundary
            for start,p,h in calls[1:]:self.assertEqual(h.shape[1],4);self.assertEqual(h[0,-1,0],start-1)

    def test_cancel_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(Cancelled):
            generate_stream(SimpleNamespace(model=SimpleNamespace(motion_rep=SimpleNamespace(fps=25),gen_horizon_len=8,num_frames_per_token=4)),
                            validate_request({'prompt':'stand'}),Path(tmp)/'cancel',lambda x:None,lambda x:None,lambda:True)


if __name__=='__main__':unittest.main()
