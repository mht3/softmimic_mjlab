/** SoftMimic actor adapter. Ordered, per-term histories match deploy.yaml.
 * No SONIC weights or 29-to-23 action truncation are used.
 */
export function gravityInBody(q) {
  const [w,x,y,z] = q;
  return [2*(w*y-x*z), -2*(y*z+w*x), 2*(x*x+y*y)-1];
}
const sizes = {base_ang_vel:3, projected_gravity:3, joint_pos_rel:23,
  joint_vel_rel:23, last_action:23, reference_dof_pos:23, reference_xy_vel:2,
  reference_yaw_vel:1, reference_root_height:1, reference_gravity_vector:3,
  reference_foot_contact:2, desired_stiffness_log:1, desired_rotational_stiffness_log:1};
export class SoftMimicPolicy {
  configure(config) {
    if (!['constant_nominal','nominal_motion'].includes(config.reference_mode) || config.observation_dim !== 327 || config.step_dt !== .02)
      throw new Error('Expected 327-input, 23-joint constant-reference standing export');
    const hardwareIds = [...Array(13).keys(),15,16,17,18,19,22,23,24,25,26];
    if (JSON.stringify(config.joint_ids_map) !== JSON.stringify(hardwareIds))
      throw new Error('Unexpected 23DOF hardware joint ordering');
    if (!(config.stiffness_command >= 40 && config.stiffness_command <= 1000 &&
          config.rotational_stiffness_command >= .1 && config.rotational_stiffness_command <= 10))
      throw new Error('Stiffness command outside the training range');
    if (JSON.stringify(Object.keys(config.observations)) !== JSON.stringify(Object.keys(sizes)))
      throw new Error('Unsupported observation ordering');
    for (const [name,term] of Object.entries(config.observations)) {
      if (term.history_length !== 3 || term.scale.length !== sizes[name]) throw new Error(`Unsupported term ${name}`);
      if (name.startsWith('reference_') && (term.params.reference_mode !== config.reference_mode || (config.reference_mode==='constant_nominal' && term.params.value.length !== sizes[name]))) throw new Error(`Invalid reference ${name}`);
    }
    const a = config.actions.JointPositionAction;
    for (const v of [config.default_joint_pos,config.stiffness,config.damping,config.effort_limits,a.scale,a.offset])
      if (v.length !== 23 || !v.every(Number.isFinite)) throw new Error('Invalid joint configuration');
    if (a.joint_ids !== null || a.clip !== null) throw new Error('Unsupported action indexing or clipping');
    if (config.policy_action_clip != null && (!Number.isFinite(config.policy_action_clip) || config.policy_action_clip<=0)) throw new Error('Invalid policy action bound');
    this.config=config;
    this.reset();
  }
  async load(config, url) {
    this.configure(config);
    const ort = window.ort;
    ort.env.wasm.wasmPaths = new URL('../lib/', import.meta.url).href;
    ort.env.wasm.numThreads = 1;
    this.session = await ort.InferenceSession.create(url, {executionProviders:['wasm']});
    if (this.session.inputNames.length !== 1 || this.session.outputNames.length !== 1) throw new Error('Unexpected ONNX interface');
    // Actual shape check occurs before allowing policy control.
    const input = new ort.Tensor('float32',new Float32Array(327),[1,327]);
    const out = await this.session.run({[this.session.inputNames[0]]:input});
    if (out[this.session.outputNames[0]].data.length !== 23) throw new Error('Expected 23 actions');
  }
  reset() { this.history = {}; this.lastAction = new Float32Array(23); this.generation=(this.generation||0)+1; }
  pushHistory(qpos,qvel,reference=null,time=0) {
    const c=this.config;
    const live={
      base_ang_vel:Array.from(qvel.slice(3,6)),
      projected_gravity:gravityInBody(qpos.slice(3,7)),
      joint_pos_rel:Array.from(qpos.slice(7,30),(x,i)=>x-c.default_joint_pos[i]),
      joint_vel_rel:Array.from(qvel.slice(6,29)),last_action:Array.from(this.lastAction),
      desired_stiffness_log:[Math.log(c.stiffness_command+1e-6)],
      desired_rotational_stiffness_log:[Math.log(c.rotational_stiffness_command+1e-6)]};
    for (const [name,term] of Object.entries(c.observations)) {
      let raw = live[name] || term.params.value;
      if (name.startsWith('reference_') && c.reference_mode==='nominal_motion') {
        if(!reference?.channels) throw new Error('Missing nominal motion channels');
        const frame=Math.max(0,Math.min(reference.frames-1,Math.floor(time*reference.fps+1e-6)));
        raw=reference.channels[name]?.[frame];
      }
      if(!raw || raw.length!==sizes[name]) throw new Error(`Missing nominal channel ${name}`);
      // Training clips each raw term before scaling and history insertion.
      const value = raw.map((v,i)=> (term.clip ? Math.max(term.clip[0],Math.min(term.clip[1],v)) : v)*term.scale[i]);
      if (!value.every(Number.isFinite)) throw new Error(`Non-finite observation: ${name}`);
      if (!this.history[name]) this.history[name]=[value.slice(),value.slice(),value.slice()];
      else {this.history[name].shift();this.history[name].push(value);}
    }
    this.observation = Float32Array.from(Object.keys(c.observations).flatMap(name=>this.history[name].flat()));
  }
  async infer() {
    const generation=this.generation;
    const input = new window.ort.Tensor('float32',this.observation,[1,327]);
    const out = await this.session.run({[this.session.inputNames[0]]:input});
    const action = Float32Array.from(out[this.session.outputNames[0]].data);
    if (action.length!==23 || !action.every(Number.isFinite)) throw new Error('Invalid policy action');
    const bounded=this.boundAction(action);
    if (generation===this.generation) this.lastAction=bounded;
    return bounded;
  }
  boundAction(action) {
    const limit=this.config.policy_action_clip;
    if (action.length!==23 || !Array.from(action).every(Number.isFinite)) throw new Error('Invalid policy action');
    return Float32Array.from(action,x=>limit == null ? x : Math.max(-limit,Math.min(limit,x)));
  }
  actionToTargets(action) {
    action=this.boundAction(action);
    const a=this.config.actions.JointPositionAction;
    return Float32Array.from(action,(x,i)=>x*a.scale[i]+a.offset[i]);
  }
  targetsToCtrl(target,qpos,qvel) {
    const c=this.config;
    return Float32Array.from(target,(x,i)=>Math.max(-c.effort_limits[i],Math.min(c.effort_limits[i],c.stiffness[i]*(x-qpos[7+i])-c.damping[i]*qvel[6+i])));
  }
}
