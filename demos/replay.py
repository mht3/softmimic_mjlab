"""Browser mesh replay; the displayed IK pose is never labeled a policy rollout."""
import argparse
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
import viser
from mjlab.viewer.viser.scene import ViserMujocoScene

ROOT = Path(__file__).resolve().parents[2]


def qpos(row, offset):
    return np.concatenate((row[offset:offset+3], row[[offset+6, offset+3, offset+4, offset+5]], row[offset+7:offset+30]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = np.loadtxt(args.csv, delimiter=',', ndmin=2)
    if rows.shape[1] != 83 or len(rows) < 2 or not np.isfinite(rows).all():
        raise ValueError('Expected a finite 83-column, 23-DOF compliant CSV with at least two frames')
    for offset in (0, 32):
        if not np.allclose(np.linalg.norm(rows[:, offset+3:offset+7], axis=1), 1, atol=1e-3):
            raise ValueError('Invalid reference quaternion')
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'softmimic_mjlab/softmimic_deploy/src/assets/g1/g1_23dof.xml'))
    data = mujoco.MjData(model)
    server = viser.ViserServer(host='127.0.0.1', port=args.port, label='G1 · Compliance Lab')
    server.scene.set_up_direction('+z')
    scene = ViserMujocoScene.create(server, model, num_envs=1)
    scene.camera_tracking_enabled = False
    scene.debug_visualization_enabled = True
    server.scene.add_grid('/ground', width=6, height=6, plane='xy')
    server.gui.add_markdown('## Reference inspection\n**Kinematic IK replay — physics OFF, policy OFF.**\nGhost: nominal input. Solid: adapted target. Red arrow: applied force in world coordinates.')
    server.gui.add_markdown(f'File: `{args.csv.name}`\nThis viewer does not certify the file; the full pipeline audit remains required.')
    play = server.gui.add_checkbox('Play', initial_value=False)
    frame = server.gui.add_slider('Frame', min=0, max=len(rows)-1, step=1, initial_value=0)
    speed = server.gui.add_slider('Playback speed', min=.1, max=2., step=.1, initial_value=1.)
    ghost = server.gui.add_checkbox('Nominal ghost', initial_value=True)
    stats = server.gui.add_markdown('')
    restart = server.gui.add_button('Restart')
    @restart.on_click
    def _(_event):
        frame.value = 0
    @server.on_client_connect
    def _(client):
        client.camera.position = (2.4, -2.4, 1.5)
        client.camera.look_at = (0, 0, .7)
    try:
        while True:
            started = time.monotonic()
            index = int(frame.value)
            row = rows[index]
            data.qpos[:] = qpos(row, 32)
            mujoco.mj_forward(model, data)
            scene.clear()
            if ghost.value:
                scene.add_ghost_mesh(qpos(row, 0), model, alpha=.25)
            body = int(row[62])
            force = row[63:66]
            if 0 < body < model.nbody:
                start = data.xpos[body].copy()
                scene.add_arrow(start, start + .003*force, (1., .18, .12, 1.))
            scene.update_from_mjdata(data)
            stats.content = (f'**{index/30:.2f} s** · force **{np.linalg.norm(force):.1f} N**\n'
                             f'World force: {force.round(1).tolist()} N\n'
                             f'Desired stiffness: {row[69]:.1f} N/m\n'
                             f'Pelvis drop: {100*(row[2]-row[34]):.1f} cm')
            if play.value:
                frame.value = (index + 1) % len(rows)
            time.sleep(max(.001, 1/(30*speed.value) - (time.monotonic()-started)))
    finally:
        server.stop()


if __name__ == '__main__':
    main()
