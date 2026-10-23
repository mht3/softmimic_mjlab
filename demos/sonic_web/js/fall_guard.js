/** Latched pelvis-IMU fall guard. Only explicit reset clears a trip. */
export class FallGuard {
  constructor(limitAngle=1.25) {
    if(!Number.isFinite(limitAngle)||limitAngle<=0||limitAngle>=Math.PI)throw Error('Invalid fall angle');
    this.limitAngle=limitAngle;this.reset();
  }
  reset(){this.tripped=false;this.reason='';}
  trip(reason){if(!this.tripped){this.tripped=true;this.reason=reason;}return true;}
  check(quaternion){
    if(this.tripped)return true;
    const norm=Math.hypot(...quaternion);
    if(quaternion.length!==4||!Number.isFinite(norm)||norm<.9||norm>1.1)return this.trip('Invalid pelvis IMU orientation');
    const [,x,y]=Array.from(quaternion,q=>q/norm);
    const tilt=Math.acos(Math.max(-1,Math.min(1,1-2*(x*x+y*y))));
    return tilt>this.limitAngle?this.trip(`Fall detected: pelvis tilt ${(tilt*180/Math.PI).toFixed(1)}°`):false;
  }
}
