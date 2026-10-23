"""CPU-only cross-runtime IMU cutoff tests, including latch and passive output."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parents[1]

def test_fall_cutoff_and_passive_output(tmp_path):
    cpp=tmp_path/'fall.cpp'
    cpp.write_text(r'''
#include "isaaclab/utils/fall_guard.h"
#include <cassert>
#include <limits>
int main(){
 for(float angle : {0.f,.5f,1.24f,1.26f,2.f,3.14f}) {
  for(auto axis:{Eigen::Vector3f::UnitX().eval(),Eigen::Vector3f::UnitY().eval()}) {
   Eigen::Quaternionf q(Eigen::AngleAxisf(angle,axis));
   assert(isaaclab::fallen_or_invalid(q,1.25f)==(angle>1.25f));
  }
 }
 Eigen::Quaternionf yaw(Eigen::AngleAxisf(3.14f,Eigen::Vector3f::UnitZ()));
 assert(!isaaclab::fallen_or_invalid(yaw,1.25f));
 assert(isaaclab::fallen_or_invalid(Eigen::Quaternionf(0,0,0,0),1.25f));
 assert(isaaclab::fallen_or_invalid(Eigen::Quaternionf(std::numeric_limits<float>::quiet_NaN(),0,0,0),1.25f));
}
''')
    exe=tmp_path/'fall'
    subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'deploy/include'),'-I/usr/include/eigen3',str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
    module=(ROOT/'demos/sonic_web/js/fall_guard.js').as_uri()
    actuation=(ROOT/'demos/sonic_web/js/actuation.js').as_uri()
    js=f"import {{FallGuard}} from {json.dumps(module)};import {{applyPassiveActuation}} from {json.dumps(actuation)};"+r'''
import assert from 'node:assert/strict';
for(const angle of [0,.5,1.24,1.26,2,3.14]) for(const axis of [1,2]) {
 const q=[Math.cos(angle/2),0,0,0];q[axis]=Math.sin(angle/2);
 const guard=new FallGuard();assert.equal(guard.check(q),angle>1.25);
 if(angle>1.25) {
  assert.equal(guard.check([1,0,0,0]),true,'Recovery must not auto-enable policy');
  guard.reset();assert.equal(guard.check([1,0,0,0]),false);
 }
}
const guard=new FallGuard();assert.equal(guard.check([0,0,0,1]),false);
assert.equal(guard.check([NaN,0,0,0]),true);
const model={nu:23,actuator_gainprm:new Float64Array(230).fill(200),actuator_biasprm:new Float64Array(230).fill(-200)};
const data={ctrl:new Float64Array(23).fill(9)};
applyPassiveActuation(model,data,3);
assert.ok(model.actuator_gainprm.every(x=>x===0));assert.ok(data.ctrl.every(x=>x===0));
for(let i=0;i<230;i++)assert.equal(model.actuator_biasprm[i],i%10===2?-3:0);
'''
    subprocess.run(['node','--input-type=module','-e',js],check=True)
