"""Training discovery must not mix abandoned generations into canonical data."""
from src.tasks.compliant_tracking.mdp.commands import resolve_motion_files


def test_parent_discovery_excludes_staging_and_backups(tmp_path):
    canonical = [
        tmp_path / "stand/forcefield/current.npz",
        tmp_path / "stand/zero-wrench/current.npz",
        tmp_path / "walk/forcefield/current.npz",
    ]
    excluded = [
        tmp_path / ".generate-failed/stand/forcefield/new.npz",
        tmp_path / ".backups/20261021/stand/forcefield/old.npz",
        tmp_path / "stand/.nested/hidden.npz",
        tmp_path / "stand/forcefield/.partial.npz",
    ]
    for path in canonical + excluded:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert resolve_motion_files(str(tmp_path)) == sorted(canonical)


def test_explicit_stage_can_be_audited_without_loading_sibling_stages(tmp_path):
    stage = tmp_path / ".generate-selected"
    clip = stage / "stand/forcefield/new.npz"
    clip.parent.mkdir(parents=True)
    clip.touch()
    assert resolve_motion_files(str(stage)) == [clip]


def test_hybrid_discovery_excludes_hidden_partial_files(tmp_path):
    canonical = []
    for mode in ("forcefield", "collision-emulator", "zero-wrench"):
        folder = tmp_path / mode
        folder.mkdir()
        clip = folder / "current.npz"
        clip.touch()
        (folder / ".partial.npz").touch()
        canonical.append(clip)
    assert set(resolve_motion_files(str(tmp_path))) == set(canonical)
