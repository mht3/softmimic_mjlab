from pathlib import Path
import re, shutil
p=Path(__file__).resolve().parent; u=p/'upstream'
(p/'js').mkdir(exist_ok=True);(p/'css').mkdir(exist_ok=True)
shutil.copy2(u/'css/style.css',p/'css/style.css')
shutil.copy2(u/'coi-serviceworker.js',p/'coi-serviceworker.js')
s=(u/'demo.html').read_text()
s=re.sub(r'      <!-- Generate Motion -->.*?(?=      <!-- Simulation controls -->)','',s,flags=re.S)
s=s.replace('SONIC Demo – Motion Tracking','Compliant Humanoid').replace('<span>SONIC</span> Demo','<span>Compliant</span> Humanoid').replace('Whole-Body Motion Tracking · G1 Humanoid','Standing Compliance · G1 23DOF')
s=s.replace('./static/pdf/sonic_paper.pdf','https://improbable-ai.github.io/softmimic/').replace('Paper PDF','SoftMimic')
s=s.replace('Toggle SONIC policy','Toggle SoftMimic policy').replace('href="index.html" class="back-link">← SONIC','href="https://nvlabs.github.io/GEAR-SONIC/demo.html" class="back-link">Viewer source: SONIC')
s=s.replace('https://cdn.jsdelivr.net/npm/three@0.162.0/build/three.module.js','./lib/three/three.module.js').replace('"three/addons/": "https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/"','"three/addons/controls/OrbitControls.js": "./lib/three/OrbitControls.js"')
(p/'index.html').write_text(s)
s=(u/'js/viewer.js').read_text().replace('Float32Array[29]','Float32Array[23]').replace('j < 29','j < model.nu')
s=s.replace('resetToDefault(defaultAngles, rootHeight) {','resetToDefault(defaultAngles, rootHeight, rootPos = [0, 0, rootHeight], rootQuat = [1, 0, 0, 0]) {')
s=s.replace('    mujoco.mj_resetData(model, data);','''    this.clearDragForce();
    mujoco.mj_resetData(model, data);''')
s=s.replace('    data.qpos[0] = 0; data.qpos[1] = 0; data.qpos[2] = rootHeight;\n    data.qpos[3] = 1; data.qpos[4] = 0; data.qpos[5] = 0; data.qpos[6] = 0;','    data.qpos.set(rootPos, 0);\n    data.qpos.set(rootQuat, 3);')
s=s.replace('    mujoco.mj_kinematics(model, data);\n    mujoco.mj_comPos(model, data);\n    this._syncGeoms();\n    // Snap','    mujoco.mj_forward(model, data);\n    this._syncGeoms();\n    // Snap')
s=s.replace('  resetToDefault(defaultAngles,','''  clearDragForce() {
    if (this.data) this.data.xfrc_applied.fill(0);
    if (this._drag) { this._drag.active = false; this._drag.bodyId = -1; }
    if (this._dragArrow) this._dragArrow.visible = false;
    if (this._dragSphere) this._dragSphere.visible = false;
    if (this.controls) this.controls.enabled = true;
  }

  resetToDefault(defaultAngles,''')
# Recompute force at the body anchor each substep, including free physics.
s=s.replace('    for (let s = 0; s < 4; s++) mujoco.mj_step(model, data);','    for (let s = 0; s < 4; s++) { this._applyDragForce(); mujoco.mj_step(model, data); }')
(p/'js/viewer.js').write_text(s)
s=(u/'demo.js').read_text()
s=s.replace("import { SonicPolicy, DEFAULT_ANGLES_MJ, DEFAULT_ROOT_HEIGHT } from './policy.js?v=2';","import { SoftMimicPolicy } from './policy.js';")
s=re.sub(r'// Generate panel DOM refs.*?(?=// ── Constants)', '', s, flags=re.S)
s=re.sub(r'// Backend state.*?(?=// FPS)', '', s, flags=re.S)
s=s.replace('new Float32Array(29)','new Float32Array(23)')
s=s.replace('  viewer.resetToDefault(DEFAULT_ANGLES_MJ, DEFAULT_ROOT_HEIGHT);','  resetPose();')
s=s.replace('  // Start backend health polling\n  checkBackend();\n  setInterval(checkBackend, 5000);','')
start=s.index('async function loadPolicy()');end=s.index('// ── Motion index loading',start)
s=s[:start]+'''async function loadPolicy() {
  const configResponse = await fetch('assets/policy/config.json');
  if (!configResponse.ok) {
    btnPolicy.textContent = 'Awaiting checkpoint';
    btnPolicy.classList.remove('policy-loading');
    btnPolicy.disabled = true;
    setStatus('ready', 'Reference ready — policy not installed');
    return;
  }
  try {
    policy = new SoftMimicPolicy();
    await policy.load(await configResponse.json(), 'assets/policy/policy.onnx');
    policyReady = true;
    btnPolicy.classList.remove('policy-loading');
    btnPolicy.textContent = 'OFF';
    btnPolicy.title = 'SoftMimic standing policy (O)';
    setStatus('ready', 'Ready — press O for policy');
  } catch (err) {
    btnPolicy.textContent = 'Policy error';
    btnPolicy.disabled = true;
    setStatus('error', `Policy failed: ${err.message}`);
  }
}

'''+s[end:]
s=s.replace('let selectedGenIdx = -1;  // index into generatedMotions, -1 means a reference clip is selected','')
s=s.replace('selectedGenIdx === -1 && ','')
a=s.index('  // Generated clips');b=s.index('// ── Select and load a reference',a)
s=s[:a]+'}\n\n'+s[b:]
s=s.replace('  selectedGenIdx = -1;','')
s=s.replace('    if (viewer && viewer.data) policy.initHeading(viewer.data.qpos, motionData);','')
s=s.replace('  renderMotionList();\n\n  infoMotion','  resetPose();\n  renderMotionList();\n\n  infoMotion')
a=s.index('// ── Select and load a generated');b=s.index('// ── Animation loop',a)
s=s[:a]+s[b:]
s=s.replace('  const action  = await policy.infer(motionData, refFi, data.qpos);','  const revision = resetRevision;\n  const action = await policy.infer();\n  if (revision !== resetRevision || !policyEnabled || !simRunning) return;')
s=s.replace('      } finally {','      } catch (error) {\n        simRunning = false; policyEnabled = false;\n        viewer.clearDragForce();\n        btnPhysics.textContent = btnPolicy.textContent = "OFF";\n        infoPhysicsMode.textContent = infoPolicyMode.textContent = "OFF";\n        btnPhysics.classList.remove("active"); btnPolicy.classList.remove("active");\n        setStatus("error", `Simulation stopped: ${error.message}`);\n      } finally {')
s=s.replace('  simRunning = !simRunning;','  resetRevision++;\n  simRunning = !simRunning;')
s=s.replace('  policyEnabled = !policyEnabled;','  resetRevision++;\n  policyEnabled = !policyEnabled;')
s=s.replace('    if (motionData) policy.initHeading(viewer.data.qpos, motionData);','')
s=s.replace('  if (policy && motionData) policy.initHeading(viewer.data.qpos, motionData);','')
s=s.replace("    setStatus('ready', 'Policy loading, please wait…');","    setStatus('ready', 'Install a trained standing policy to enable control');")
a=s.index('// Generate button + Enter');b=s.index('// ── Keyboard shortcuts',a)
s=s[:a]+s[b:]
s=s.replace('window._sonic =','window._compliant =')
s=s.replace('// ── Boot', '''let resetRevision = 0;
function resetPose() {
  resetRevision++;
  if (!motionData) return;
  viewer.resetToDefault(motionData.joint_pos[0], motionData.root_pos[0][2], motionData.root_pos[0], motionData.root_quat[0]);
  if (policy) policy.reset();
}

// ── Boot''')
s=s.replace('SONIC','SoftMimic').replace('SonicPolicy','SoftMimicPolicy').replace(' * Also integrates MFM backend for text-to-motion generation.',' * Adapted from NVIDIA GEAR-SONIC browser demo; standing reference only.')
(p/'js/demo.js').write_text(s)
