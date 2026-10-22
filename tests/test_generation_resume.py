"""Interrupted generation must recover completed work without mixing datasets."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest


def load():
    path=Path(__file__).resolve().parents[1]/'scripts/generate_compliant_23dof.py'
    spec=importlib.util.spec_from_file_location('generation_resume_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("foot_error", [0.01, 0.0718])
def test_resume_reuses_completed_mode_and_replaces_only_after_audit(tmp_path,monkeypatch,foot_error):
    m=load();repo=tmp_path/'repo';root=repo/'data';stage=root/'.generate-test'
    ff=stage/'stand/forcefield';ff.mkdir(parents=True)
    csv=ff/'stand_augmented_mink_001.csv';csv.write_text('preserve this generated motion')
    old=root/'stand/collision-emulator';old.mkdir(parents=True);(old/'old.npz').write_text('old')
    ref=repo/'ref.csv';ref.write_text('reference')
    (repo/'scripts').mkdir();(repo/'scripts/compliant_csv_to_npz.py').write_text('converter')
    monkeypatch.setattr(m,'REPO_ROOT',repo)
    aug_dir=repo/'compliant_motion_augmentation';aug_dir.mkdir()
    (aug_dir/'generator.py').write_text('mock generator')
    monkeypatch.setattr(m,'AUG_DIR',aug_dir)
    monkeypatch.setattr(m,'filter_reference',lambda *a:ref)
    generated=[]
    def augment(motion,mode,ref,n,seed,output_root,cost):
        generated.append(mode);p=output_root/motion/mode;p.mkdir(parents=True)
        (p/f'{motion}_augmented_mink_001.csv').write_text('new zero wrench');return p
    monkeypatch.setattr(m,'augment',augment)
    def convert(directory,*args):
        for p in directory.glob('*.csv'):p.with_suffix('.npz').write_text('converted')
    monkeypatch.setattr(m,'to_npz',convert)
    def audit(cmd,**kw):
        assert (old/'old.npz').exists(),'old data replaced before audit'
        Path(cmd[-1]).write_text(json.dumps([{'npz_finite':True,'npz_joint_count':23,'npz_has_nominal':True,'sampled_max_foot_displacement_m':foot_error}]*2))
    monkeypatch.setattr(m,'_run',audit)
    monkeypatch.setattr(sys,'argv',['generate','--motions','stand','--modes','forcefield','zero-wrench',
        '--num-files','1','1','--replace','--resume-stage',str(stage),'--output-root',str(root)])
    if foot_error > .05:
        with pytest.raises(RuntimeError, match='old data retained'):
            m.main()
        assert (old/'old.npz').read_text() == 'old'
        assert stage.exists()
        assert not (root/'.backups').exists()
        return
    m.main()
    assert generated==['zero-wrench']
    assert (root/'stand/forcefield'/csv.name).read_text()=='preserve this generated motion'
    assert not (root/'stand/collision-emulator').exists()
    assert len(list((root/'.backups').glob('*/stand/collision-emulator/old.npz')))==1
    assert not stage.exists()
    assert json.loads((root/'stand/generation.json').read_text())['legacy_resume_unverified_generation_settings']


def test_resume_rejects_outside_directory(tmp_path,monkeypatch):
    m=load();outside=tmp_path/'.generate-outside';outside.mkdir()
    monkeypatch.setattr(sys,'argv',['generate','--motions','stand','--replace','--resume-stage',str(outside),
                                  '--output-root',str(tmp_path/'different')])
    with pytest.raises(SystemExit):m.main()
    assert outside.exists()
