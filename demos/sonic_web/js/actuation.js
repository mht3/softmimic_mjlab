/** Damping-only fallback for the prepared affine PD actuators. */
export function applyPassiveActuation(model,data,damping=3) {
  if(!Number.isFinite(damping)||damping<0)throw Error('Invalid passive damping');
  model.actuator_gainprm.fill(0);
  model.actuator_biasprm.fill(0);
  for(let i=0;i<model.nu;i++)model.actuator_biasprm[i*10+2]=-damping;
  data.ctrl.fill(0);
}
