/**
 * demo.js – SONIC physics demo controller.
 *
 * Modes:
 *  - Static:  robot in default standing pose (no simulation)
 *  - Physics: simulation running, zero control (robot falls) — press P
 *  - Policy:  SONIC policy drives the robot — click Policy button
 *
 * Also integrates MFM backend for text-to-motion generation.
 */

import { MuJoCoViewer } from './viewer.js';
import { SonicPolicy, DEFAULT_ANGLES_MJ, DEFAULT_ROOT_HEIGHT } from './policy.js?v=2';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const canvas      = document.getElementById('canvas');
const overlay     = document.getElementById('loading-overlay');
const loadStatus  = document.getElementById('loading-status');
const statusDot   = document.getElementById('status-dot');
const statusText  = document.getElementById('status-text');
const statusFps   = document.getElementById('status-fps');
const motionList  = document.getElementById('motion-list');
const btnPhysics  = document.getElementById('btn-physics');
const btnRestart  = document.getElementById('btn-restart');
const infoMotion  = document.getElementById('info-motion');
const btnPolicy   = document.getElementById('btn-policy');
const infoPhysicsMode = document.getElementById('info-physics-mode');
const infoPolicyMode  = document.getElementById('info-policy-mode');

// Generate panel DOM refs
const promptInput    = document.getElementById('prompt-input');
const durationInput  = document.getElementById('duration-input');
const btnGenerate    = document.getElementById('btn-generate');
const generateStatus = document.getElementById('generate-status');

// ── Backend config ──────────────────────────────────────────────────────────
// Default: Kimodo bridge via ngrok tunnel. Override with ?backend=URL.
const _params = new URLSearchParams(window.location.search);
const BACKEND_URL = _params.get('backend') || 'https://nonconfident-subcortically-jonell.ngrok-free.app';

// ── Constants ─────────────────────────────────────────────────────────────────
const ZERO_CTRL    = new Float32Array(29);  // pre-allocated zero control vector
const POLICY_DT_MS = 20;   // policy/physics step interval: 1000ms / 50Hz = 20ms

// ── State ─────────────────────────────────────────────────────────────────────
let viewer        = null;
let motions       = [];      // [{ id, display, frames, fps }] from index.json
let motionData    = null;    // loaded motion JSON { fps, frames, joint_pos, root_pos, root_quat }
let simRunning    = false;   // true = physics simulation active
let policyEnabled = false;   // true = SONIC policy controls the robot

// Backend state
let backendAvailable = false;
let generating       = false;
let generatedMotions = [];  // [{ name, display, fps, frames, root_pos, root_quat, joint_pos }]

// FPS / physics Hz tracking
let fpsFrames   = 0;
let physFrames  = 0;
let fpsStart    = performance.now();

// Performance tracking
let _perfStepTimes  = [];   // wall-clock time per policy+physics step (ms)
let _perfSkipCount  = 0;    // frames skipped due to _policyRunning guard

// Policy
let policy         = null;   // SonicPolicy instance
let policyReady    = false;
let policyRefFi    = 0.0;    // reference frame index (float)
let _policyRunning = false;  // guard against overlapping async policy calls

// ── Browser compatibility check ──────────────────────────────────────────────
const _isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

// ── Entry point ───────────────────────────────────────────────────────────────
async function main() {
  fitCanvas();

  if (_isSafari) {
    loadStatus.textContent = 'Safari has known issues with MuJoCo WASM threading. Please use Chrome or Firefox for the best experience.';
  }

  viewer = new MuJoCoViewer(canvas);
  viewer.initThree();

  try {
    await viewer.loadScene(msg => {
      loadStatus.textContent = msg;
    });
  } catch (err) {
    const safariHint = _isSafari ? ' (Safari is not fully supported — try Chrome or Firefox)' : '';
    loadStatus.textContent = `Error: ${err?.message ?? String(err)}${safariHint}`;
    statusDot.className = 'status-dot error';
    statusText.textContent = 'Load failed';
    return;
  }

  // Enable drag-to-push interaction
  viewer.initDragForce();

  // Show robot in default standing pose (static, no simulation)
  viewer.resetToDefault(DEFAULT_ANGLES_MJ, DEFAULT_ROOT_HEIGHT);

  // Load motion index (needed as policy reference)
  await loadMotionIndex();

  // Hide overlay
  overlay.classList.add('fade-out');
  setTimeout(() => { overlay.style.display = 'none'; }, 600);

  setStatus('ready', 'Ready — press P for physics');

  // Begin loading SONIC policy in the background (non-blocking)
  loadPolicy();

  // Start backend health polling
  checkBackend();
  setInterval(checkBackend, 5000);

  // Start animation loop
  requestAnimationFrame(tick);
}

// ── Canvas sizing ─────────────────────────────────────────────────────────────
function fitCanvas() {
  canvas.style.width  = window.innerWidth  + 'px';
  canvas.style.height = window.innerHeight + 'px';
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener('resize', () => { fitCanvas(); });

// ── SONIC policy loading ──────────────────────────────────────────────────────
async function loadPolicy() {
  if (!window.ort) {
    if (btnPolicy) { btnPolicy.textContent = 'ORT unavailable'; btnPolicy.classList.remove('policy-loading'); }
    return;
  }
  try {
    policy = new SonicPolicy();
    if (btnPolicy) btnPolicy.textContent = 'Loading encoder…';
    await policy.load(
      'assets/policy/model_encoder.onnx',
      'assets/policy/model_decoder.onnx',
      msg => {
        if (btnPolicy) btnPolicy.textContent = msg;
      },
    );
    policyReady = true;
    btnPolicy.classList.remove('policy-loading');
    // Auto-start policy
    togglePolicy();
  } catch (err) {
    if (btnPolicy) { btnPolicy.textContent = 'Policy error'; btnPolicy.classList.remove('policy-loading'); }
    setStatus('ready', `Policy failed: ${err.message}`);
  }
}

// ── Backend health check ────────────────────────────────────────────────────
async function checkBackend() {
  try {
    const resp = await fetch(`${BACKEND_URL}/api/health`, { signal: AbortSignal.timeout(3000) });
    if (resp.ok) {
      const data = await resp.json();
      backendAvailable = true;
      btnGenerate.disabled = generating;
      if (!generating) {
        generateStatus.textContent = `Backend ready (${data.model}, ${data.fps} fps)`;
      }
      return;
    }
  } catch (_) { /* offline */ }
  backendAvailable = false;
  btnGenerate.disabled = true;
  if (!generating) {
    generateStatus.textContent = 'Backend offline';
  }
}

// ── Motion index loading ──────────────────────────────────────────────────────
async function loadMotionIndex() {
  const resp = await fetch('assets/motions/index.json');
  motions = await resp.json();
  renderMotionList();
  if (motions.length > 0) {
    await selectMotion(0);
  }
}

// ── Render motion list UI ─────────────────────────────────────────────────────
let selectedIdx = -1;
let selectedGenIdx = -1;  // index into generatedMotions, -1 means a reference clip is selected

function renderMotionList() {
  motionList.innerHTML = '';

  // Reference clips
  motions.forEach((m, i) => {
    const btn = document.createElement('button');
    btn.className = 'motion-item';
    btn.textContent = m.display || m.id;
    btn.title = `${m.frames} frames @ ${m.fps} fps`;
    btn.addEventListener('click', () => selectMotion(i));
    if (selectedGenIdx === -1 && i === selectedIdx) btn.classList.add('active');
    motionList.appendChild(btn);
  });

  // Generated clips
  generatedMotions.forEach((m, i) => {
    const btn = document.createElement('button');
    btn.className = 'motion-item';
    btn.innerHTML = `<span class="gen-dot"></span>${m.display || m.name}`;
    btn.title = `Generated · ${m.frames} frames @ ${m.fps} fps`;
    btn.addEventListener('click', () => selectGeneratedMotion(i));
    if (selectedGenIdx === i) btn.classList.add('active');
    motionList.appendChild(btn);
  });
}

// ── Select and load a reference motion clip ───────────────────────────────────
async function selectMotion(idx) {
  const m = motions[idx];
  if (!m) return;

  setStatus('loading', `Loading "${m.display}"…`);
  try {
    const resp = await fetch(`assets/motions/${m.id}.json`);
    motionData = await resp.json();
  } catch (err) {
    setStatus('error', 'Failed to load motion');
    return;
  }

  selectedIdx    = idx;
  selectedGenIdx = -1;
  policyRefFi    = 0;
  if (policy) {
    if (viewer && viewer.data) policy.initHeading(viewer.data.qpos, motionData);
    policy.reset();
  }

  renderMotionList();

  infoMotion.textContent = m.display || m.id;

  const stateMsg = simRunning
    ? (policyEnabled ? 'Policy running' : 'Physics running')
    : 'Ready';
  setStatus('ready', stateMsg);
}

// ── Select and load a generated motion ──────────────────────────────────────
function selectGeneratedMotion(idx) {
  const m = generatedMotions[idx];
  if (!m) return;

  motionData     = m;
  selectedIdx    = -1;
  selectedGenIdx = idx;
  policyRefFi    = 0;
  if (policy) {
    if (viewer && viewer.data) policy.initHeading(viewer.data.qpos, motionData);
    policy.reset();
  }

  renderMotionList();

  infoMotion.textContent = m.display || m.name;

  const stateMsg = simRunning
    ? (policyEnabled ? 'Policy running' : 'Physics running')
    : 'Ready';
  setStatus('ready', stateMsg);
}

// ── Generate motion from text ───────────────────────────────────────────────
async function generateMotion() {
  const prompt = promptInput.value.trim();
  if (!prompt || !backendAvailable || generating) return;

  const duration = parseFloat(durationInput.value) || 5.0;
  if (duration < 2 || duration > 10) {
    generateStatus.textContent = 'Duration must be between 2 and 10 seconds.';
    return;
  }

  generating = true;
  btnGenerate.disabled = true;
  btnGenerate.textContent = 'Generating…';
  btnGenerate.classList.add('generating');
  generateStatus.textContent = 'Running inference…';

  try {
    const resp = await fetch(`${BACKEND_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, duration, diffusion_steps: 100 }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      let msg = err.detail;
      if (Array.isArray(msg)) {
        msg = msg.map(e => `${e.loc?.slice(-1)[0]}: ${e.msg}`).join('; ');
      }
      throw new Error(msg || 'Generation failed');
    }

    const result = await resp.json();
    generatedMotions.push(result);
    generateStatus.textContent = `Generated "${result.display}" (${result.frames} frames)`;

    // Auto-select the new motion as policy reference
    selectGeneratedMotion(generatedMotions.length - 1);
    showToast(`Generated: ${result.display}`);
  } catch (err) {
    generateStatus.textContent = `Error: ${err.message}`;
  } finally {
    generating = false;
    btnGenerate.disabled = !backendAvailable;
    btnGenerate.textContent = 'Generate';
    btnGenerate.classList.remove('generating');
  }
}

// ── Animation loop ────────────────────────────────────────────────────────────
let _lastPhysicsTs = 0;  // timestamp of last physics step (ms)

async function tick(ts) {
  requestAnimationFrame(tick);

  // Throttle physics + policy to 50 Hz regardless of RAF frame rate.
  // Use accumulator-based timing: advance _lastPhysicsTs by POLICY_DT_MS (not ts)
  // so partial-frame remainders carry forward.  On a 60 Hz display, this gives
  // a run-run-run-run-run-skip pattern (50/60) instead of the old run-skip (30/60).
  const sinceLast = ts - _lastPhysicsTs;
  if (viewer.ready && simRunning && sinceLast >= POLICY_DT_MS && !_policyRunning) {
    _lastPhysicsTs += POLICY_DT_MS;
    // Spiral-of-death guard: if too far behind (tab backgrounded, etc.), reset
    if (ts - _lastPhysicsTs > POLICY_DT_MS * 5) _lastPhysicsTs = ts;

    if (policyEnabled && policyReady && motionData) {
      // ── SONIC policy mode ──────────────────────────────────────────────────
      _policyRunning = true;
      const stepT0 = performance.now();
      try {
        await runPolicyStep();
      } finally {
        _policyRunning = false;
        _perfStepTimes.push(performance.now() - stepT0);
      }
    } else {
      // ── Free physics: zero control (robot falls under gravity) ─────────────
      viewer.stepPhysics(ZERO_CTRL);
    }
    physFrames++;
  } else if (viewer.ready && simRunning && sinceLast >= POLICY_DT_MS && _policyRunning) {
    // Frame was eligible but blocked by ongoing async inference
    _perfSkipCount++;
  }

  viewer.render();

  // FPS / physics Hz counter + performance report
  fpsFrames++;
  const elapsed = ts - fpsStart;
  if (elapsed >= 1000) {
    const renderFps = Math.round(fpsFrames  * 1000 / elapsed);
    const physHz    = Math.round(physFrames * 1000 / elapsed);
    fpsFrames  = 0;
    physFrames = 0;
    fpsStart   = ts;
    statusFps.textContent = `${renderFps} render fps · ${physHz} Hz sim`;

    _perfStepTimes = [];
    _perfSkipCount = 0;
  }
}

// ── Run one SONIC policy step ─────────────────────────────────────────────────
async function runPolicyStep() {
  const { data } = viewer;

  const refFi = Math.floor(policyRefFi) % motionData.frames;

  // Snapshot state for policy inference (before any sub-stepping)
  policy.pushHistory(data.qpos, data.qvel);
  const action  = await policy.infer(motionData, refFi, data.qpos);
  const targets = policy.actionToTargets(action);

  // Re-evaluate PD torques from live qpos/qvel at each sub-step.
  // Matches base_sim.py sim_step() at 200 Hz (4× per 50 Hz policy tick):
  // the KD damping term responds to velocity built up within the substep
  // window, which prevents the oscillation of holding torque constant.
  for (let s = 0; s < 4; s++) {
    const ctrl = policy.targetsToCtrl(targets, data.qpos, data.qvel);
    viewer.stepPhysicsOnce(ctrl);
  }
  viewer.finalizeStep();

  // Advance reference frame (policy at 50 Hz; motionFps/50 frames per tick)
  // Clamp at last frame — simulation keeps running, no loop.
  policyRefFi = Math.min(policyRefFi + motionData.fps / 50, motionData.frames - 1);
}

// ── Toggle physics simulation ─────────────────────────────────────────────────
function togglePhysics() {
  simRunning = !simRunning;

  if (simRunning) {
    btnPhysics.classList.add('active');
    btnPhysics.textContent = 'ON';
    if (infoPhysicsMode) infoPhysicsMode.textContent = 'ON';
    viewer.resetToDefault(DEFAULT_ANGLES_MJ, DEFAULT_ROOT_HEIGHT);
    setStatus('ready', 'Physics running');
  } else {
    // Stop physics and disable policy
    if (policyEnabled) {
      policyEnabled = false;
      if (btnPolicy && policyReady) {
        btnPolicy.classList.remove('active');
        btnPolicy.textContent = 'OFF';
      }
      if (infoPolicyMode) infoPolicyMode.textContent = 'OFF';
    }
    btnPhysics.classList.remove('active');
    btnPhysics.textContent = 'OFF';
    if (infoPhysicsMode) infoPhysicsMode.textContent = 'OFF';
    viewer.resetToDefault(DEFAULT_ANGLES_MJ, DEFAULT_ROOT_HEIGHT);
    setStatus('ready', 'Ready');
  }
}

// ── Toggle SONIC policy ───────────────────────────────────────────────────────
function togglePolicy() {
  if (!policyReady) {
    setStatus('ready', 'Policy loading, please wait…');
    return;
  }

  policyEnabled = !policyEnabled;

  if (policyEnabled) {
    // Auto-enable physics if not already running
    if (!simRunning) {
      simRunning = true;
      btnPhysics.classList.add('active');
      btnPhysics.textContent = 'ON';
      if (infoPhysicsMode) infoPhysicsMode.textContent = 'ON';
    }
    btnPolicy.classList.add('active');
    btnPolicy.classList.remove('policy-ready');
    btnPolicy.textContent = 'ON';
    if (infoPolicyMode) infoPolicyMode.textContent = 'ON';
    policyRefFi = 0;
    viewer.resetToDefault(DEFAULT_ANGLES_MJ, DEFAULT_ROOT_HEIGHT);
    if (motionData) policy.initHeading(viewer.data.qpos, motionData);
    policy.reset();
    setStatus('ready', 'Policy running');
  } else {
    btnPolicy.classList.remove('active');
    btnPolicy.classList.add('policy-ready');
    btnPolicy.textContent = 'OFF';
    if (infoPolicyMode) infoPolicyMode.textContent = 'OFF';
    setStatus('ready', simRunning ? 'Physics running' : 'Ready');
  }
}

// ── Restart simulation ────────────────────────────────────────────────────────
function restartSim() {
  policyRefFi = 0;
  viewer.resetToDefault(DEFAULT_ANGLES_MJ, DEFAULT_ROOT_HEIGHT);
  if (policy && motionData) policy.initHeading(viewer.data.qpos, motionData);
  if (policy) policy.reset();
}

// ── UI event handlers ─────────────────────────────────────────────────────────
if (btnPhysics) btnPhysics.addEventListener('click', togglePhysics);
if (btnRestart) btnRestart.addEventListener('click', restartSim);
if (btnPolicy)  btnPolicy.addEventListener('click', togglePolicy);
const btnFollow = document.getElementById('btn-follow');
if (btnFollow)  btnFollow.addEventListener('click', toggleFollowCamera);

// Generate button + Enter key in prompt
btnGenerate.addEventListener('click', generateMotion);
promptInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    e.preventDefault();
    generateMotion();
  }
});

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  switch (e.key) {
    case 'p': case 'P':
      e.preventDefault();
      togglePhysics();
      break;
    case 'r': case 'R':
      restartSim();
      break;
    case 'f': case 'F':
      toggleFollowCamera();
      break;
    case 'o': case 'O':
      e.preventDefault();
      togglePolicy();
      break;
  }
});

// ── Camera follow ─────────────────────────────────────────────────────────────
function toggleFollowCamera() {
  if (!viewer) return;
  viewer.followCamera = !viewer.followCamera;
  if (viewer.followCamera) viewer.centerOnRobot();
  const btn = document.getElementById('btn-follow');
  if (btn) {
    btn.classList.toggle('active', viewer.followCamera);
    btn.textContent = viewer.followCamera ? 'ON' : 'OFF';
  }
  showToast(viewer.followCamera ? 'Camera following robot' : 'Camera free');
}

// ── Toast helper ──────────────────────────────────────────────────────────────
function showToast(msg, duration = 4500) {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(el._hideTimer);
  el._hideTimer = setTimeout(() => el.classList.remove('show'), duration);
}

// ── Status helper ─────────────────────────────────────────────────────────────
function setStatus(state, text) {
  statusDot.className = `status-dot ${state === 'ready' ? '' : state}`;
  statusText.textContent = text;
}

// ── Expose internals for console debugging ───────────────────────────────────
window._sonic = { get viewer() { return viewer; }, get policy() { return policy; } };

// ── Boot ──────────────────────────────────────────────────────────────────────
main().catch(err => {
  loadStatus.textContent = `Fatal error: ${err.message}`;
});
