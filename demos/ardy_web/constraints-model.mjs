// Generation goals are independent of the saved-motion editing timeline.
export const TRACKS=[['root2d','2D Root'],['fullbody','Full-Body'],['left-hand','Left Hand'],['right-hand','Right Hand'],['left-foot','Left Foot'],['right-foot','Right Foot']];
export const snap=t=>Math.round(t*25)/25;
export function examplePlan(kind){
 if(kind==='skills')return {segments:[{prompt:'A person side steps to the right.',seconds:4},{prompt:'A person is jogging.',seconds:4},{prompt:'A person is jumping.',seconds:4},{prompt:'A person is standing still.',seconds:4}],constraints:[]};
 if(kind==='circle')return {segments:[{prompt:'A person is walking.',seconds:12}],constraints:Array.from({length:13},(_,i)=>({id:crypto.randomUUID(),type:'root2d',time:Math.min(i,11.96),position:[.6*Math.sin(i*Math.PI/6),.6*(1-Math.cos(i*Math.PI/6))]}))};
 return {segments:[{prompt:'A person is walking.',seconds:8}],constraints:[]};
}
export function checkPlan(plan){
 const total=plan.segments.reduce((v,s)=>v+s.seconds,0);
 if(!plan.segments.length||!Number.isFinite(total)||total>30||plan.segments.some(s=>!s.prompt.trim()||!Number.isFinite(s.seconds)||s.seconds<2||s.seconds>30))throw Error('Each prompt needs 2–30 seconds; the complete generation must fit 30 seconds.');
 if(plan.constraints.some(c=>!Number.isFinite(c.time)||c.time<0||c.time>total))throw Error('A keyframe is outside the generation duration. Move or delete it first.');
 return total;
}
