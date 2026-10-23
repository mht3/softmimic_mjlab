/**
 * viewer.js – MuJoCo WASM + Three.js renderer for the SONIC demo.
 *
 * Pipeline:
 *  1. loadScene()  – fetch scene XML + STL meshes, write to MuJoCo VFS, build model.
 *  2. buildScene() – create Three.js objects for each MuJoCo geom.
 *  3. setFrame()   – write qpos, call mj_kinematics, sync Three.js transforms.
 *  4. render()     – Three.js render call.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// MuJoCo geom type constants (mjGeomType enum)
const mjGEOM_PLANE    = 0;
const mjGEOM_SPHERE   = 2;
const mjGEOM_CAPSULE  = 3;
const mjGEOM_ELLIPSOID= 4;
const mjGEOM_CYLINDER = 5;
const mjGEOM_BOX      = 6;
const mjGEOM_MESH     = 7;

// Helper – extract a sub-slice from a TypedArray-like WASM object
function slice(src, start, len) {
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) out[i] = src[start + i];
  return out;
}
function sliceInt(src, start, len) {
  const out = new Int32Array(len);
  for (let i = 0; i < len; i++) out[i] = src[start + i];
  return out;
}

export class MuJoCoViewer {
  constructor(canvas) {
    this.canvas   = canvas;
    this.mujoco   = null;
    this.model    = null;
    this.data     = null;
    this.renderer = null;
    this.scene    = null;
    this.camera   = null;
    this.controls = null;

    this._geomMeshes = [];     // THREE.Object3D per geom
    this._meshGeoms  = [];     // THREE.BufferGeometry per MuJoCo mesh id
    this._ready      = false;
    this._drag       = null;   // drag-force state (set by initDragForce)
    this.followCamera   = true;   // whether camera tracks root XY
    this._followPrevXY  = [0, 0]; // previous root XY for delta tracking
  }

  // ── Initialise Three.js ─────────────────────────────────────────────────────
  initThree() {
    const { canvas } = this;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x071422);
    this.scene.fog = new THREE.FogExp2(0x071422, 0.04);

    // Camera – use Z-up (MuJoCo convention)
    this.camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.01, 100);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(3.5, -3.5, 2.5);
    this.camera.lookAt(0, 0, 0.9);

    // OrbitControls
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0.9);
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 20;
    this.controls.update();

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.65);
    this.scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
    keyLight.position.set(4, -2, 6);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    keyLight.shadow.camera.near = 0.1;
    keyLight.shadow.camera.far = 30;
    keyLight.shadow.camera.left = -5;
    keyLight.shadow.camera.right = 5;
    keyLight.shadow.camera.top = 5;
    keyLight.shadow.camera.bottom = -5;
    this.scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x62d8ff, 0.7);
    fillLight.position.set(-3, 4, 2);
    this.scene.add(fillLight);

    // Back light to lift dark side
    const backLight = new THREE.DirectionalLight(0x8de5ff, 0.5);
    backLight.position.set(-2, 2, 4);
    this.scene.add(backLight);

    // Hemisphere for natural sky/ground tint
    const hemi = new THREE.HemisphereLight(0x91d9ff, 0x102d48, 0.8);
    this.scene.add(hemi);

    // Resize handler
    window.addEventListener('resize', () => this._onResize());
  }

  // ── Load MuJoCo WASM + scene ────────────────────────────────────────────────
  async loadScene(onProgress) {
    const progress = onProgress || (() => {});

    // 1. Load mujoco_wasm module
    progress('Loading MuJoCo WASM…');
    const loadMujoco = (await import('../lib/mujoco_wasm.js')).default;
    this.mujoco = await loadMujoco({ INITIAL_MEMORY: 512 * 1024 * 1024 });
    const mujoco = this.mujoco;

    // 2. Set up virtual filesystem
    mujoco.FS.mkdir('/working');
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
    mujoco.FS.mkdir('/working/meshes');

    // 3. Fetch + write scene XML
    progress('Loading robot model…');
    const xmlText = await fetch('assets/robot/scene.xml').then(r => r.text());
    mujoco.FS.writeFile('/working/scene.xml', xmlText);

    // 4. Parse mesh filenames from scene XML
    const xmlDoc = new DOMParser().parseFromString(xmlText, 'application/xml');
    const meshFiles = [...xmlDoc.querySelectorAll('mesh[file]')]
      .map(m => m.getAttribute('file'))
      .filter(Boolean);

    // 5. Fetch + write all mesh STL files
    progress(`Loading meshes (0 / ${meshFiles.length})…`);
    const meshBuffers = {};  // fname → ArrayBuffer (for both VFS and geometry)
    let loaded = 0;
    await Promise.all(meshFiles.map(async fname => {
      const buf = await fetch(`assets/robot/meshes/${fname}`)
        .then(r => r.ok ? r.arrayBuffer() : Promise.reject(new Error(`404: ${fname}`)));
      mujoco.FS.writeFile(`/working/meshes/${fname}`, new Uint8Array(buf));
      meshBuffers[fname] = buf;
      loaded++;
      progress(`Loading meshes (${loaded} / ${meshFiles.length})…`);
    }));

    // 6. Compile MuJoCo model
    progress('Compiling model…');
    this.model = mujoco.MjModel.mj_loadXML('/working/scene.xml');
    this.data  = new mujoco.MjData(this.model);

    // Run one step to initialise default state properly
    mujoco.mj_kinematics(this.model, this.data);

    // 7. Build Three.js geometry from MuJoCo mesh data
    progress('Building render scene…');
    this._buildMeshGeometries();
    this._buildGeomObjects();

    this._ready = true;
  }

  // ── Build Three.js BufferGeometry for every MuJoCo mesh ────────────────────
  _buildMeshGeometries() {
    const { model } = this;
    const nmesh = model.nmesh;
    this._meshGeoms = new Array(nmesh).fill(null);

    for (let mi = 0; mi < nmesh; mi++) {
      const vertadr = model.mesh_vertadr[mi];
      const vertnum = model.mesh_vertnum[mi];
      const faceadr = model.mesh_faceadr[mi];
      const facenum = model.mesh_facenum[mi];

      if (vertnum === 0 || facenum === 0) continue;

      // Extract vertices (Float32, 3 per vertex)
      const verts = new Float32Array(vertnum * 3);
      for (let i = 0; i < vertnum * 3; i++) verts[i] = model.mesh_vert[vertadr * 3 + i];

      // Extract faces (Int32, 3 per face)
      const faces = new Uint32Array(facenum * 3);
      for (let i = 0; i < facenum * 3; i++) faces[i] = model.mesh_face[faceadr * 3 + i];

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
      geo.setIndex(new THREE.BufferAttribute(faces, 1));
      geo.computeVertexNormals();
      this._meshGeoms[mi] = geo;
    }
  }

  // ── Build Three.js Object3D for every MuJoCo geom ──────────────────────────
  _buildGeomObjects() {
    const { model, scene } = this;
    const ngeom = model.ngeom;
    this._geomMeshes = new Array(ngeom).fill(null);

    // Shared materials
    const matDark  = new THREE.MeshStandardMaterial({ color: 0x333333, metalness: 0.7, roughness: 0.4 });
    const matGrey  = new THREE.MeshStandardMaterial({ color: 0xb3b3b3, metalness: 0.5, roughness: 0.5 });
    const matFloor = new THREE.MeshStandardMaterial({ color: 0x10263b, metalness: 0.0, roughness: 0.9 });
    const matShadow = new THREE.ShadowMaterial({ opacity: 0.35 });

    for (let gi = 0; gi < ngeom; gi++) {
      const gtype  = model.geom_type[gi];

      // Skip non-visual geoms (MuJoCo group >= 3 convention)
      if (model.geom_group[gi] >= 3) continue;

      // Size (half-extents for box, radius+half-len for capsule, etc.)
      const gs0 = model.geom_size[gi * 3 + 0];
      const gs1 = model.geom_size[gi * 3 + 1];
      const gs2 = model.geom_size[gi * 3 + 2];

      // MJCF visual geoms inherit named materials. geom_rgba alone is the
      // default grey even when the mesh is assigned the black/silver material.
      const materialId = model.geom_matid[gi];
      const rgba = materialId >= 0 ? model.mat_rgba : model.geom_rgba;
      const colorIndex = 4 * (materialId >= 0 ? materialId : gi);
      const [r,g,b,a] = Array.from(rgba.slice(colorIndex,colorIndex+4));

      let geometry = null;

      switch (gtype) {
        case mjGEOM_PLANE: {
          // Large checkerboard floor
          geometry = new THREE.PlaneGeometry(40, 40, 40, 40);
          const mat = new THREE.MeshStandardMaterial({
            color: 0x10263b,
            metalness: 0.0,
            roughness: 0.95,
            wireframe: false,
          });
          const mesh = new THREE.Mesh(geometry, mat);
          mesh.receiveShadow = true;
          mesh.matrixAutoUpdate = false;
          this._geomMeshes[gi] = mesh;
          scene.add(mesh);
          // Fine grid
          const grid = new THREE.GridHelper(20, 40, 0x2585b0, 0x194c70);
          grid.rotation.x = Math.PI / 2;
          scene.add(grid);
          // Coarse major grid (every 1m)
          const majorGrid = new THREE.GridHelper(20, 20, 0x339ac4, 0x339ac4);
          majorGrid.rotation.x = Math.PI / 2;
          majorGrid.position.z = 0.001; // slight offset to avoid z-fight
          scene.add(majorGrid);
          continue;
        }
        case mjGEOM_SPHERE: {
          geometry = new THREE.SphereGeometry(gs0, 16, 10);
          break;
        }
        case mjGEOM_CAPSULE: {
          // MuJoCo capsule: radius=gs0, half-cyl-len=gs1, axis=Z
          // Three.js CapsuleGeometry: radius, length (full cyl), cap segs, body segs, axis=Y
          geometry = new THREE.CapsuleGeometry(gs0, gs1 * 2, 8, 14);
          // Rotate Y→Z (MuJoCo capsules point along Z)
          geometry.applyMatrix4(new THREE.Matrix4().makeRotationX(Math.PI / 2));
          break;
        }
        case mjGEOM_ELLIPSOID: {
          const geo = new THREE.SphereGeometry(1, 12, 8);
          geo.applyMatrix4(new THREE.Matrix4().makeScale(gs0, gs1, gs2));
          geometry = geo;
          break;
        }
        case mjGEOM_CYLINDER: {
          // MuJoCo cylinder: radius=gs0, half-len=gs1, axis=Z
          geometry = new THREE.CylinderGeometry(gs0, gs0, gs1 * 2, 16);
          geometry.applyMatrix4(new THREE.Matrix4().makeRotationX(Math.PI / 2));
          break;
        }
        case mjGEOM_BOX: {
          geometry = new THREE.BoxGeometry(gs0 * 2, gs1 * 2, gs2 * 2);
          break;
        }
        case mjGEOM_MESH: {
          const meshId = model.geom_dataid[gi];
          geometry = meshId >= 0 ? this._meshGeoms[meshId] : null;
          break;
        }
        default:
          continue;
      }

      if (!geometry) continue;

      // Pick material from rgba
      let mat;
      if (gtype === mjGEOM_PLANE) {
        mat = matFloor;
      } else {
        const isDark = r < 0.4 && g < 0.4 && b < 0.4;
        const col = new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
        mat = new THREE.MeshStandardMaterial({
          color: col,
          metalness: isDark ? 0.15 : 0.5,
          roughness: isDark ? 0.6 : 0.38,
          transparent: a < 0.99,
          opacity: a,
        });
      }

      const mesh = new THREE.Mesh(geometry, mat);
      mesh.castShadow    = true;
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      mesh.userData.geomIdx = gi;
      this._geomMeshes[gi] = mesh;
      scene.add(mesh);
    }
  }

  // ── Set robot state from motion data and run kinematics ────────────────────
  setFrame(rootPos, rootQuat, jointPos) {
    if (!this._ready) return;
    const { mujoco, model, data } = this;
    const qpos = data.qpos;

    // Root position [x, y, z]
    qpos[0] = rootPos[0];
    qpos[1] = rootPos[1];
    qpos[2] = rootPos[2];

    // Root quaternion [w, x, y, z] – MuJoCo free joint convention
    qpos[3] = rootQuat[0];  // w
    qpos[4] = rootQuat[1];  // x
    qpos[5] = rootQuat[2];  // y
    qpos[6] = rootQuat[3];  // z

    // Joint angles (29 DOF)
    const nj = Math.min(jointPos.length, model.nq - 7);
    for (let i = 0; i < nj; i++) qpos[7 + i] = jointPos[i];

    // Forward kinematics only (no dynamics, no contact)
    mujoco.mj_kinematics(model, data);
    mujoco.mj_comPos(model, data);

    // Sync Three.js geom transforms
    this._syncGeoms();
  }

  // A second FK-only MjData: the ghost never contributes contacts or forces.
  setGhost(rootPos,rootQuat,jointPos,offset=[0,1.25,0]) {
    if(!this._ready)return;
    if(!this._ghostData){
      this._ghostData=new this.mujoco.MjData(this.model);this._ghostMeshes=[];
      for(let gi=0;gi<this.model.ngeom;gi++){
        const source=this._geomMeshes[gi];if(!source||!this.model.geom_bodyid[gi])continue;
        const mesh=new THREE.Mesh(source.geometry,new THREE.MeshStandardMaterial({color:0x57d9ff,emissive:0x126078,emissiveIntensity:.45,transparent:true,opacity:.35,depthWrite:false,roughness:.5}));
        mesh.matrixAutoUpdate=false;mesh.userData.referenceGhost=true;this.scene.add(mesh);this._ghostMeshes.push([gi,mesh]);
      }
    }
    const d=this._ghostData;d.qpos.set([...rootPos,...rootQuat,...jointPos]);
    this.mujoco.mj_kinematics(this.model,d);this.mujoco.mj_comPos(this.model,d);
    for(const [gi,mesh] of this._ghostMeshes){const r=d.geom_xmat,o=gi*9,p=gi*3;mesh.matrix.set(r[o],r[o+1],r[o+2],d.geom_xpos[p]+offset[0],r[o+3],r[o+4],r[o+5],d.geom_xpos[p+1]+offset[1],r[o+6],r[o+7],r[o+8],d.geom_xpos[p+2]+offset[2],0,0,0,1);}
  }

  // ── Sync Three.js meshes from data.geom_xpos / data.geom_xmat ──────────────
  _syncGeoms() {
    const { model, data } = this;
    const ngeom     = model.ngeom;
    const geom_xpos = data.geom_xpos;
    const geom_xmat = data.geom_xmat;  // row-major 3×3 per geom

    const m = new THREE.Matrix4();

    for (let gi = 0; gi < ngeom; gi++) {
      const obj = this._geomMeshes[gi];
      if (!obj) continue;

      const gtype = model.geom_type[gi];

      // Floor plane: handled separately (stays at origin)
      if (gtype === mjGEOM_PLANE) {
        m.identity();
        obj.matrix.copy(m);
        continue;
      }

      const px = geom_xpos[gi * 3 + 0];
      const py = geom_xpos[gi * 3 + 1];
      const pz = geom_xpos[gi * 3 + 2];

      // Row-major 3×3 rotation matrix → Three.js Matrix4
      // MuJoCo stores row-major: R[0..8] where R[row*3+col]
      // THREE.Matrix4.set() takes row-major arguments, stored column-major internally
      const R = geom_xmat;
      const o = gi * 9;
      m.set(
        R[o+0], R[o+1], R[o+2], px,
        R[o+3], R[o+4], R[o+5], py,
        R[o+6], R[o+7], R[o+8], pz,
        0,      0,      0,      1
      );
      obj.matrix.copy(m);
    }
  }

  // ── Step physics: apply ctrl torques, advance simulation, sync render ──────
  // ctrl: Float32Array[23] torques in MuJoCo actuator order.
  // Runs 4 sub-steps at dt=0.005s each (= 0.02s = 50 Hz policy rate).

  setPolicyActuation(enabled) {
    if (!this._actuatorParameters) this._actuatorParameters = {
      gain: Float64Array.from(this.model.actuator_gainprm),
      bias: Float64Array.from(this.model.actuator_biasprm)};
    this.model.actuator_gainprm.set(enabled ? this._actuatorParameters.gain : new Float64Array(this._actuatorParameters.gain.length));
    this.model.actuator_biasprm.set(enabled ? this._actuatorParameters.bias : new Float64Array(this._actuatorParameters.bias.length));
  }

  stepPhysics(ctrl) {
    if (!this._ready) return;
    this._applyDragForce();
    const { mujoco, model, data } = this;
    for (let j = 0; j < ctrl.length; j++) data.ctrl[j] = ctrl[j];
    for (let s = 0; s < 4; s++) { this._applyDragForce(); mujoco.mj_step(model, data); }
    this._syncGeoms();
  }

  // ── Single physics sub-step (no geom sync) ─────────────────────────────────
  // The prepared policy scene evaluates PD implicitly inside MuJoCo.
  // Position targets update at 50 Hz; drag forces update every substep.

  stepPhysicsOnce(ctrl) {
    if (!this._ready) return;
    this._applyDragForce();
    const { mujoco, model, data } = this;
    for (let j = 0; j < ctrl.length; j++) data.ctrl[j] = ctrl[j];
    mujoco.mj_step(model, data);
  }

  // Sync Three.js geoms after a manual substep loop.
  finalizeStep() {
    if (this._ready) this._syncGeoms();
  }

  // ── Drag-to-push: pointer interaction applies a spring force to bodies ──────
  initDragForce() {
    const { canvas, camera } = this;
    const raycaster = new THREE.Raycaster();
    raycaster.firstHitOnly = true;

    const getNdc = (e) => {
      const r = canvas.getBoundingClientRect();
      return new THREE.Vector2(
        ((e.clientX - r.left) / r.width)  *  2 - 1,
        ((e.clientY - r.top)  / r.height) * -2 + 1
      );
    };

    this._drag = {
      active:      false, bodyId: -1,
      localAnchor: new THREE.Vector3(),  // hit point in body-local frame
      anchor:      new THREE.Vector3(),  // hit point in world frame (updated each frame)
      current:     new THREE.Vector3(),
      plane:       new THREE.Plane(),
    };

    // Arrow visualisation: origin sphere + ArrowHelper
    const arrowDir    = new THREE.Vector3(1, 0, 0);
    const arrowOrigin = new THREE.Vector3();
    this._dragArrow = new THREE.ArrowHelper(arrowDir, arrowOrigin, 0.001, 0x57d9ff, 0.06, 0.04);
    this._dragArrow.visible = false;
    // Render on top of robot (depthTest off so arrow is always visible)
    this._dragArrow.line.material.depthTest = false;
    this._dragArrow.cone.material.depthTest = false;
    this._dragArrow.renderOrder = 999;
    this.scene.add(this._dragArrow);

    // Small sphere at the anchor point
    const sphereGeo = new THREE.SphereGeometry(0.018, 12, 8);
    const sphereMat = new THREE.MeshBasicMaterial({ color: 0x57d9ff, depthTest: false });
    this._dragSphere = new THREE.Mesh(sphereGeo, sphereMat);
    this._dragSphere.renderOrder = 999;
    this._dragSphere.visible = false;
    this.scene.add(this._dragSphere);

    const updateArrow = () => {
      this._syncDragAnchor();
      const { anchor, current } = this._drag;
      const delta = new THREE.Vector3().subVectors(current, anchor);
      const len   = delta.length();
      this._dragSphere.position.copy(anchor);
      if (len > 0.001) {
        this._dragArrow.position.copy(anchor);
        this._dragArrow.setDirection(delta.clone().normalize());
        this._dragArrow.setLength(len, Math.min(0.08, len * 0.35), Math.min(0.05, len * 0.25));
        this._dragArrow.visible  = true;
      } else {
        this._dragArrow.visible = false;
      }
      this._dragSphere.visible = true;
    };

    canvas.addEventListener('pointerdown', (e) => {
      if (!this._ready || e.button !== 0) return;
      this.scene.updateMatrixWorld(false);
      raycaster.setFromCamera(getNdc(e), camera);
      const hits = raycaster.intersectObjects(this._geomMeshes.filter(Boolean), false);
      if (!hits.length) return;

      const hit    = hits[0];
      const gi     = hit.object.userData.geomIdx;
      if (gi === undefined) return;
      const bodyId = this.model.geom_bodyid[gi];
      if (bodyId <= 0) return;  // skip world body / floor

      // Store hit point in body-local frame (so anchor follows the body)
      const xpos = this.data.xpos;
      const xmat = this.data.xmat;
      const o    = bodyId * 9;
      const rx = hit.point.x - xpos[bodyId*3+0];
      const ry = hit.point.y - xpos[bodyId*3+1];
      const rz = hit.point.z - xpos[bodyId*3+2];
      // localAnchor = R^T * (hitPoint - bodyPos)
      this._drag.localAnchor.set(
        xmat[o+0]*rx + xmat[o+3]*ry + xmat[o+6]*rz,
        xmat[o+1]*rx + xmat[o+4]*ry + xmat[o+7]*rz,
        xmat[o+2]*rx + xmat[o+5]*ry + xmat[o+8]*rz,
      );
      this._drag.anchor.copy(hit.point);

      const camDir = new THREE.Vector3();
      camera.getWorldDirection(camDir);
      this._drag.plane.setFromNormalAndCoplanarPoint(camDir, hit.point);
      this._drag.current.copy(hit.point);
      this._drag.bodyId = bodyId;
      this._drag.active = true;
      this.controls.enabled = false;
      canvas.setPointerCapture(e.pointerId);
      updateArrow();
    });

    canvas.addEventListener('pointermove', (e) => {
      if (!this._drag.active) return;
      raycaster.setFromCamera(getNdc(e), camera);
      const pt = new THREE.Vector3();
      if (raycaster.ray.intersectPlane(this._drag.plane, pt)) {
        this._drag.current.copy(pt);
        updateArrow();
      }
    });

    const stopDrag = () => {
      if (!this._drag.active) return;
      if (this._ready && this.data && this._drag.bodyId >= 0) {
        const base = this._drag.bodyId * 6;
        for (let i = 0; i < 6; i++) this.data.xfrc_applied[base + i] = 0;
      }
      this._drag.active  = false;
      this._drag.bodyId  = -1;
      this.controls.enabled = true;
      this._dragArrow.visible  = false;
      this._dragSphere.visible = false;
    };
    canvas.addEventListener('pointerup',     stopDrag);
    canvas.addEventListener('pointercancel', stopDrag);
  }

  // Update _drag.anchor to follow the body in world space.
  _syncDragAnchor() {
    if (!this._drag?.active || this._drag.bodyId < 0 || !this._ready) return;
    const { data, _drag } = this;
    const id = _drag.bodyId;
    const xmat = data.xmat;
    const o  = id * 9;
    const lx = _drag.localAnchor.x, ly = _drag.localAnchor.y, lz = _drag.localAnchor.z;
    // worldAnchor = R * localAnchor + xpos
    _drag.anchor.set(
      xmat[o+0]*lx + xmat[o+1]*ly + xmat[o+2]*lz + data.xpos[id*3+0],
      xmat[o+3]*lx + xmat[o+4]*ly + xmat[o+5]*lz + data.xpos[id*3+1],
      xmat[o+6]*lx + xmat[o+7]*ly + xmat[o+8]*lz + data.xpos[id*3+2],
    );
  }

  _applyDragForce() {
    if (!this._drag?.active || this._drag.bodyId < 0) return;
    this._syncDragAnchor();
    const { data, _drag } = this;
    const id = _drag.bodyId;

    // Linear spring: F = K * (mouse_world - anchor_world)  — MuJoCo-style
    const DRAG_K = 50;  // N/m
    const fx = (_drag.current.x - _drag.anchor.x) * DRAG_K;
    const fy = (_drag.current.y - _drag.anchor.y) * DRAG_K;
    const fz = (_drag.current.z - _drag.anchor.z) * DRAG_K;

    // Torque: τ = (anchor - CoM) × F  — applies force at anchor point, not CoM
    const cx = data.xipos[id*3+0], cy = data.xipos[id*3+1], cz = data.xipos[id*3+2];
    const rx = _drag.anchor.x - cx;
    const ry = _drag.anchor.y - cy;
    const rz = _drag.anchor.z - cz;

    const base = id * 6;
    data.xfrc_applied[base + 0] = fx;
    data.xfrc_applied[base + 1] = fy;
    data.xfrc_applied[base + 2] = fz;
    data.xfrc_applied[base + 3] = ry*fz - rz*fy;
    data.xfrc_applied[base + 4] = rz*fx - rx*fz;
    data.xfrc_applied[base + 5] = rx*fy - ry*fx;
  }

  // ── Reset robot to default standing pose ───────────────────────────────────
  // defaultAngles: Float32Array[23] joint angles in MuJoCo order.
  // rootHeight: z position of the root body (metres).

  clearDragForce() {
    if (this.data) this.data.xfrc_applied.fill(0);
    if (this._drag) { this._drag.active = false; this._drag.bodyId = -1; }
    if (this._dragArrow) this._dragArrow.visible = false;
    if (this._dragSphere) this._dragSphere.visible = false;
    if (this.controls) this.controls.enabled = true;
  }

  resetToDefault(defaultAngles, rootHeight, rootPos = [0, 0, rootHeight], rootQuat = [1, 0, 0, 0]) {
    if (!this._ready) return;
    this._dbgSubstep = 0;  // reset substep counter so we log first 8 after each restart
    const { mujoco, model, data } = this;

    this.clearDragForce();
    mujoco.mj_resetData(model, data);

    data.qpos.set(rootPos, 0);
    data.qpos.set(rootQuat, 3);
    for (let j = 0; j < model.nu; j++) data.qpos[7 + j] = defaultAngles[j];
    for (let i = 0; i < model.nv; i++) data.qvel[i] = 0;
    for (let j = 0; j < model.nu; j++) data.ctrl[j] = 0;

    mujoco.mj_forward(model, data);
    this._syncGeoms();
    // Snap camera back to origin so follow-cam doesn't strand it
    if (this.followCamera) {
      const dx = -this._followPrevXY[0];
      const dy = -this._followPrevXY[1];
      this.camera.position.x   += dx; this.camera.position.y   += dy;
      this.controls.target.x   += dx; this.controls.target.y   += dy;
    }
    this._followPrevXY[0] = 0;
    this._followPrevXY[1] = 0;
  }

  // ── Center camera on robot's current XY position ────────────────────────────
  centerOnRobot() {
    if (!this._ready) return;
    const rx = this.data.qpos[0], ry = this.data.qpos[1];
    const dx = rx - this.controls.target.x;
    const dy = ry - this.controls.target.y;
    this.camera.position.x += dx;  this.camera.position.y += dy;
    this.controls.target.x += dx;  this.controls.target.y += dy;
    this._followPrevXY[0] = rx;    this._followPrevXY[1] = ry;
  }

  // ── Render frame ───────────────────────────────────────────────────────────
  render() {
    if (!this.renderer) return;

    // Camera follow: shift camera + orbit target by root XY delta each frame
    if (this.followCamera && this._ready) {
      const rx = this.data.qpos[0], ry = this.data.qpos[1];
      const dx = rx - this._followPrevXY[0];
      const dy = ry - this._followPrevXY[1];
      this.camera.position.x   += dx; this.camera.position.y   += dy;
      this.controls.target.x   += dx; this.controls.target.y   += dy;
      this._followPrevXY[0] = rx;     this._followPrevXY[1] = ry;
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  // ── Resize handler ──────────────────────────────────────────────────────────
  _onResize() {
    const { canvas, renderer, camera } = this;
    if (!renderer) return;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  get ready() { return this._ready; }
}
