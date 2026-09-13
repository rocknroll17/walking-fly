// Browser viewer: MuJoCo WASM physics + the trained PPO policy evaluated in JS.
// Observation/action code mirrors flybiped/env.py (sensor-only observations).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import load_mujoco from '../node_modules/@mujoco/mujoco/mujoco.js';

const $ = (id) => document.getElementById(id);
const status = (t) => { $('status').textContent = t; };

// ----------------------------------------------------------------- MuJoCo
const mujoco = await load_mujoco({
  locateFile: (p, prefix) => p.endsWith('.wasm') ? new URL('../node_modules/@mujoco/mujoco/mujoco.wasm', import.meta.url).href : prefix + p,
});
mujoco.FS.mkdir('/working');
mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
status('모델 파일 내려받는 중...');
const manifest = (await (await fetch('./assets/manifest.txt')).text()).trim().split('\n');
await Promise.all(['biped.xml', ...manifest].map(async (f) => {
  mujoco.FS.writeFile('/working/' + f, new Uint8Array(await (await fetch('./assets/' + f)).arrayBuffer()));
}));
const model = mujoco.MjModel.mj_loadXML('/working/biped.xml');
const data = new mujoco.MjData(model);

// ----------------------------------------------------------------- Policy
let policy = null, E = null;
async function loadPolicy() {
  try {
    const spec = await (await fetch('./assets/policy.json?t=' + Date.now())).json();
    spec.layers = spec.layers.map((l) => ({ w: l.w.map((r) => Float32Array.from(r)), b: Float32Array.from(l.b), nin: l.w.length, nout: l.b.length }));
    spec.obs_mean = Float32Array.from(spec.obs_mean); spec.obs_std = Float32Array.from(spec.obs_std);
    policy = spec; E = spec.env;
    $('policy').textContent = spec.trained ? `학습 ${(spec.step / 1e6).toFixed(1)}M 스텝` : '학습 전 (무작위)';
  } catch (e) { status('정책 파일을 못 읽었습니다: ' + e); }
}
await loadPolicy();

let noiseOn = true;
function gauss() { let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function mlp(obs) {
  let x = new Float32Array(obs.length);
  for (let i = 0; i < obs.length; i++) x[i] = (obs[i] - policy.obs_mean[i]) / policy.obs_std[i];
  policy.layers.forEach((l, li) => {
    const y = new Float32Array(l.nout);
    for (let j = 0; j < l.nout; j++) {
      let s = l.b[j];
      for (let i = 0; i < l.nin; i++) s += x[i] * l.w[i][j];
      y[j] = li < policy.layers.length - 1 ? s / (1 + Math.exp(-s)) : s;   // SiLU hidden, linear output
    }
    x = y;
  });
  const a = new Float32Array(policy.action_size);
  for (let i = 0; i < a.length; i++) {
    let v = x[i];
    if (noiseOn) {                       // sample like training: tanh(loc + std * N(0,1)), std = softplus(raw) + 0.001
      const std = Math.log1p(Math.exp(x[policy.action_size + i])) + 0.001;
      v += std * gauss();
    }
    a[i] = Math.tanh(v);
  }
  return a;
}

// ------------------------------------------------------------ Observation
const GEOM = { SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6 };
function lowestZ(g) {           // lowest world-z point of a primitive geom (matches env._lowest_z)
  const R = data.geom_xmat, o = g * 9, z0 = Math.abs(R[o + 6]), z1 = Math.abs(R[o + 7]), z2 = Math.abs(R[o + 8]);
  const s0 = model.geom_size[g * 3], s1 = model.geom_size[g * 3 + 1], s2 = model.geom_size[g * 3 + 2];
  const t = model.geom_type[g];
  let drop = s0;
  if (t === GEOM.CAPSULE || t === GEOM.CYLINDER) drop = s0 + z2 * s1;
  else if (t === GEOM.ELLIPSOID) drop = Math.hypot(s0 * z0, s1 * z1, s2 * z2);
  else if (t === GEOM.BOX) drop = s0 * z0 + s1 * z1 + s2 * z2;
  return data.geom_xpos[g * 3 + 2] - drop;
}
const minLow = (geoms) => geoms.reduce((m, g) => Math.min(m, lowestZ(g)), Infinity);
function quatRotate(q, v) {      // rotate vector v by quaternion q (w,x,y,z)
  const [w, x, y, z] = q, [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  return [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)];
}
const thoraxQuat = () => [0, 1, 2, 3].map((i) => data.xquat[E.thorax_body * 4 + i]);
const thoraxPos = () => [0, 1, 2].map((i) => data.site_xpos[E.thorax_site * 3 + i]);

const state = { goal: [1, 0, E.goal.height], lastAction: new Float32Array(E.nu), reached: 0, trail: [],
                gait: { flight: 0, walk: 0, alt: 0, same: 0, both: 0 }, lastFc: [1, 1], lastTd: -1, lastMidFc: [0, 0] };
const hindL = E.hind_geoms.filter(g => (geomName(g) || '').includes('T1_left')), hindR = E.hind_geoms.filter(g => (geomName(g) || '').includes('T1_right'));
const midL = Array.from({length: model.ngeom}).map((_, i) => i).filter(g => model.geom_contype[g] > 0 && (geomName(g) || '').includes('T2_left'));
const midR = Array.from({length: model.ngeom}).map((_, i) => i).filter(g => model.geom_contype[g] > 0 && (geomName(g) || '').includes('T2_right'));
function geomName(g) { const a = model.name_geomadr[g]; let e = a; while (model.names[e] !== 0) e++; return new TextDecoder().decode(model.names.subarray(a, e)); }

// Footprint system
const footprints = [];
const numFootprints = 200;
const fpGeom = new THREE.CylinderGeometry(0.015, 0.015, 0.002, 16);
const fpMat = new THREE.MeshBasicMaterial({ color: 0xff3333, transparent: true, opacity: 0.8 });
function initFootprints() {
  for(let i=0; i<numFootprints; i++) {
    const m = new THREE.Mesh(fpGeom, fpMat.clone());
    m.visible = false;
    scene.add(m);
    footprints.push({ mesh: m, time: 0 });
  }
}
let fpIdx = 0;
function dropFootprint(geoms) {
  let minZ = Infinity, minG = -1;
  for (const g of geoms) {
    const z = lowestZ(g);
    if (z < minZ) { minZ = z; minG = g; }
  }
  if (minG !== -1) {
    const fp = footprints[fpIdx];
    fpIdx = (fpIdx + 1) % numFootprints;
    fp.mesh.visible = true;
    fp.time = 5.0; // visible for 5.0 seconds
    fp.mesh.material.opacity = 0.8;
    setPos(data.geom_xpos, minG, fp.mesh.position);
    fp.mesh.position.y = 0.001; // slightly above floor
  }
}

function updateGait() {                  // same bookkeeping as flybiped.evaluate.CpuEnv
  const fc = [minLow(hindL) < 0.003 ? 1 : 0, minLow(hindR) < 0.003 ? 1 : 0];
  const c = contacts(), g = state.gait;
  const airborne = minLow(E.fore_geoms) > E.clearance && minLow(E.body_geoms) >= 0 && fc[0] + fc[1] === 0 && thoraxPos()[2] > E.biped_min_height;
  if (c.bipedal || airborne) { g.walk++; if (airborne) g.flight++; }
  const td = [fc[0] * (1 - state.lastFc[0]), fc[1] * (1 - state.lastFc[1])];
  if (td[0] + td[1] === 2) { if (c.bipedal) g.both++; state.lastTd = -1; }
  else if (td[0] + td[1] === 1) { const foot = td[0] ? 0 : 1; if (c.bipedal) { if (foot === state.lastTd) g.same++; else g.alt++; } state.lastTd = foot; }
  state.lastFc = fc;
}
function goalDist() { const p = thoraxPos(); return Math.hypot(p[0] - state.goal[0], p[1] - state.goal[1]); }
function contacts() {           // same rule as env._contacts
  const fore = minLow(E.fore_geoms), hind = minLow(E.hind_geoms), body = minLow(E.body_geoms), h = thoraxPos()[2];
  return { fore: fore < 0.003, body: body < 0, hind: hind < 0.003,
           bipedal: fore > E.clearance && hind < 0.003 && !(body < 0) && h > E.biped_min_height };
}
function seeGoal() {            // env._see_goal: goal in the head frame, limited field of view
  const hp = [0, 1, 2].map((i) => data.site_xpos[E.head_site * 3 + i]);
  const R = data.site_xmat, o = E.head_site * 9;
  const dw = [state.goal[0] - hp[0], state.goal[1] - hp[1], state.goal[2] - hp[2]];
  const d = [R[o] * dw[0] + R[o + 3] * dw[1] + R[o + 6] * dw[2],           // R^T d
             R[o + 1] * dw[0] + R[o + 4] * dw[1] + R[o + 7] * dw[2],
             R[o + 2] * dw[0] + R[o + 5] * dw[1] + R[o + 8] * dw[2]];
  const dist = Math.hypot(d[0], d[1], d[2]) + 1e-6;
  const az = Math.atan2(d[1], d[0]), el = Math.asin(Math.max(-1, Math.min(1, d[2] / dist)));
  const v = Math.abs(az) < E.azimuth_limit_deg * Math.PI / 180 ? 1 : 0;
  return [v, v * Math.sin(az), v * Math.cos(az), v * Math.sin(el), v * 2 * Math.atan(E.goal_radius / dist)];
}
function observe() {
  const q = thoraxQuat(), qinv = [q[0], -q[1], -q[2], -q[3]];
  const grav = quatRotate(qinv, [0, 0, -1]);
  const sd = data.sensordata;
  const obs = [];
  for (let i = 7; i < E.nq; i++) obs.push(data.qpos[i]);
  for (let i = 6; i < E.nv; i++) obs.push(data.qvel[i] * 0.05);
  obs.push(...grav);
  for (let i = 0; i < 3; i++) obs.push(sd[E.gyro_adr + i] * 0.05);
  for (let i = 0; i < 3; i++) obs.push(sd[E.accel_adr + i] * 1e-3);
  for (let i = 0; i < 3; i++) obs.push(sd[E.velocimeter_adr + i]);
  for (const a of E.force_adr) for (let i = 0; i < 3; i++) obs.push(Math.tanh(sd[a + i]));
  for (const a of E.touch_adr) obs.push(Math.tanh(sd[a]));
  obs.push(...seeGoal());
  for (let i = 0; i < E.nu; i++) obs.push(state.lastAction[i]);
  return obs;
}
function applyAction(a) {
  for (let i = 0; i < E.nu; i++) {
    const lo = E.ctrl_lo[i], hi = E.ctrl_hi[i];
    let c = E.ctrl0[i] + a[i] * 0.5 * (hi - lo) * E.action_scale;
    if ((E.torque_act || E.wing_act || []).includes(i)) c = a[i];
    if (E.adh_act.includes(i)) c = 0.5 * (a[i] + 1);
    data.ctrl[i] = Math.min(hi, Math.max(lo, c));
  }
  state.lastAction.set(a);
}
function sampleGoal() {
  const p = thoraxPos(), ang = Math.random() * 2 * Math.PI;
  const r = E.goal.dist_min + Math.random() * (E.goal.dist_max - E.goal.dist_min);
  state.goal = [p[0] + r * Math.cos(ang), p[1] + r * Math.sin(ang), E.goal.height];
  for (let i = 0; i < 3; i++) data.mocap_pos[i] = state.goal[i];
}
let assistOn = false;
function applyAssist() {   // training harness: upward pull on the thorax (fraction of body weight)
  const f = assistOn ? (E.assist || 0) * (E.weight || 0) : 0;
  data.xfrc_applied[E.thorax_body * 6 + 2] = f;
}
function reset(mode = 'stance') {
  mujoco.mj_resetData(model, data);
  const biped = mode === 'biped';
  const q = Array.from(biped ? E.q_biped : E.q_stance), c = biped ? E.ctrl_biped : E.ctrl_stance;
  const yaw = Math.random() * 2 * Math.PI, hw = Math.cos(yaw / 2), hz = Math.sin(yaw / 2);
  const [w, x, y, z] = [q[3], q[4], q[5], q[6]];   // yaw * q
  q[3] = hw * w - hz * z; q[4] = hw * x - hz * y; q[5] = hw * y + hz * x; q[6] = hw * z + hz * w;
  if (mode === 'flip') {                          // on its back
    q[2] = 0.25; const [w, x, y, z] = [q[3], q[4], q[5], q[6]];   // q * rot_x(pi) = (w,x,y,z)*(0,1,0,0)
    q[3] = -x; q[4] = w; q[5] = z; q[6] = -y;
  }
  if (mode === 'drop') {                          // random orientation from a height (fall recovery)
    q[2] = 1.0; const r = [0, 1, 2, 3].map(() => gauss()), n = Math.hypot(...r);
    for (let i = 0; i < 4; i++) q[3 + i] = r[i] / n;
  }
  for (let i = 0; i < E.nq; i++) data.qpos[i] = q[i];
  applyAssist();
  for (let i = 0; i < E.nu; i++) { data.ctrl[i] = c[i]; if (E.act_adr[i] >= 0) data.act[E.act_adr[i]] = c[i]; }
  state.lastAction.fill(0); state.reached = 0; state.trail.length = 0;
  state.gait = { flight: 0, walk: 0, alt: 0, same: 0, both: 0 }; state.lastFc = [1, 1]; state.lastTd = -1;
  mujoco.mj_forward(model, data);
  /* 
  if (mode === 'drop' || mode === 'flip') {       // like training: land and settle before the policy acts
    const n = Math.round((E.settle_time || 0.25) / model.opt.timestep);
    for (let i = 0; i < n; i++) mujoco.mj_step(model, data);
    data.time = 0;
  }
  */
  sampleGoal();
  simTime = 0; substep = 0; wall0 = performance.now(); simAtWall0 = 0;
}

// ----------------------------------------------------------------- Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a2230);
const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.01, 100);
camera.position.set(0.55, 0.35, 0.55);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xdde6ff, 0x30281c, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 2.2); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048); sun.shadow.camera.left = sun.shadow.camera.bottom = -0.8; sun.shadow.camera.right = sun.shadow.camera.top = 0.8;
sun.shadow.camera.near = 0.5; sun.shadow.camera.far = 12; sun.shadow.bias = -0.0002; sun.shadow.normalBias = 0.002;
const sunTarget = new THREE.Object3D(); scene.add(sunTarget); sun.target = sunTarget; scene.add(sun);
const cv = document.createElement('canvas'); cv.width = cv.height = 256; const ctx = cv.getContext('2d');
for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { ctx.fillStyle = (i + j) % 2 ? '#2b3b52' : '#334763'; ctx.fillRect(i * 128, j * 128, 128, 128); }
const floorTex = new THREE.CanvasTexture(cv); floorTex.wrapS = floorTex.wrapT = THREE.RepeatWrapping; floorTex.repeat.set(40, 40);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.9 }));
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
initFootprints();
// Trail of the thorax path.
const trailGeom = new THREE.BufferGeometry(); const trailPos = new Float32Array(3 * 2000);
trailGeom.setAttribute('position', new THREE.BufferAttribute(trailPos, 3)); trailGeom.setDrawRange(0, 0);
scene.add(new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0xffcc66 })));

// MuJoCo (z-up) -> three.js (y-up): (x, y, z) -> (x, z, -y)
const setPos = (buf, i, target) => target.set(buf[i * 3], buf[i * 3 + 2], -buf[i * 3 + 1]);
const setQuat = (buf, i, target) => target.set(-buf[i * 4 + 1], -buf[i * 4 + 3], buf[i * 4 + 2], -buf[i * 4]);
const bodies = {}, meshCache = {};
for (let g = 0; g < model.ngeom; g++) {
  if (model.geom_group[g] >= 3) continue;                       // collision geoms are hidden
  const b = model.geom_bodyid[g], t = model.geom_type[g];
  if (t === 0) continue;                                       // plane: drawn above
  const s = [model.geom_size[g * 3], model.geom_size[g * 3 + 1], model.geom_size[g * 3 + 2]];
  if (!bodies[b]) { bodies[b] = new THREE.Group(); scene.add(bodies[b]); }
  let geom;
  if (t === 7) {
    const mid = model.geom_dataid[g];
    if (!meshCache[mid]) {
      const v = model.mesh_vert.slice(model.mesh_vertadr[mid] * 3, (model.mesh_vertadr[mid] + model.mesh_vertnum[mid]) * 3);
      for (let i = 0; i < v.length; i += 3) { const y = v[i + 1]; v[i + 1] = v[i + 2]; v[i + 2] = -y; }
      const f = model.mesh_face.slice(model.mesh_faceadr[mid] * 3, (model.mesh_faceadr[mid] + model.mesh_facenum[mid]) * 3);
      geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(v, 3)); geom.setIndex(Array.from(f)); geom.computeVertexNormals();
      meshCache[mid] = geom;
    }
    geom = meshCache[mid];
  } else if (t === GEOM.SPHERE) geom = new THREE.SphereGeometry(s[0], 16, 12);
  else if (t === GEOM.CAPSULE) geom = new THREE.CapsuleGeometry(s[0], s[1] * 2, 6, 12);
  else if (t === GEOM.ELLIPSOID) geom = new THREE.SphereGeometry(1, 16, 12);
  else if (t === GEOM.CYLINDER) geom = new THREE.CylinderGeometry(s[0], s[0], s[1] * 2, 16);
  else if (t === GEOM.BOX) geom = new THREE.BoxGeometry(s[0] * 2, s[2] * 2, s[1] * 2);
  else continue;
  let rgba = [model.geom_rgba[g * 4], model.geom_rgba[g * 4 + 1], model.geom_rgba[g * 4 + 2], model.geom_rgba[g * 4 + 3]];
  const mat = model.geom_matid[g];
  const isDefault = Math.abs(rgba[0] - 0.5) < 1e-6 && Math.abs(rgba[1] - 0.5) < 1e-6 && Math.abs(rgba[2] - 0.5) < 1e-6 && rgba[3] === 1;
  if (mat >= 0 && isDefault) rgba = [model.mat_rgba[mat * 4], model.mat_rgba[mat * 4 + 1], model.mat_rgba[mat * 4 + 2], model.mat_rgba[mat * 4 + 3]];
  if (rgba[3] <= 0.01) continue;                                // invisible helper geoms
  const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color: new THREE.Color(rgba[0], rgba[1], rgba[2]), transparent: rgba[3] < 1, opacity: rgba[3], roughness: 0.6, side: THREE.DoubleSide }));
  mesh.castShadow = true; mesh.receiveShadow = true;
  setPos(model.geom_pos, g, mesh.position); setQuat(model.geom_quat, g, mesh.quaternion);
  if (t === GEOM.ELLIPSOID) mesh.scale.set(s[0], s[2], s[1]);
  bodies[b].add(mesh);
}
function syncBodies() {
  for (const b in bodies) { setPos(data.xpos, +b, bodies[b].position); setQuat(data.xquat, +b, bodies[b].quaternion); }
}

// ------------------------------------------------------------------ Loop
let simTime = 0, substep = 0, paused = false, speed = 0.2, wall0 = performance.now(), simAtWall0 = 0, follow = true;
try { const sv = localStorage.getItem('flybiped.speed'); if (sv) { speed = parseFloat(sv); $('speed').value = sv; } } catch (e) {}
const dt = model.opt.timestep;
reset('stance');
$('reset').onclick = () => reset('stance');
$('resetBiped').onclick = () => reset('biped');
$('resetDrop').onclick = () => reset('drop');
$('resetFlip').onclick = () => reset('flip');
$('pause').onclick = () => { paused = !paused; $('pause').textContent = paused ? '▶ 재생' : '⏸ 일시정지'; wall0 = performance.now(); simAtWall0 = simTime; };
$('speed').onchange = (e) => { speed = parseFloat(e.target.value); wall0 = performance.now(); simAtWall0 = simTime; try { localStorage.setItem('flybiped.speed', e.target.value); } catch (err) {} };
$('follow').onchange = (e) => { follow = e.target.checked; };
$('reload').onclick = () => loadPolicy();
$('assist').onchange = (e) => { assistOn = e.target.checked; applyAssist(); };
$('noise').onchange = (e) => { noiseOn = e.target.checked; };
setInterval(loadPolicy, 30000);    // pick up new checkpoints while training runs
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });

let last = performance.now(), lastHud = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const elapsed = Math.min((now - last) / 1000, 0.05); last = now;
  if (!paused) {
    let budget = elapsed * speed;
    const t0 = performance.now();
    while (budget > 0 && performance.now() - t0 < 25) {
      if (substep === 0) {
        const a = mlp(observe());
        let d = 0; for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - state.lastAction[i]);
        state.actionChange = 0.9 * (state.actionChange || 0) + 0.1 * d / a.length;
        applyAction(a);
      }
      mujoco.mj_step(model, data);
      if (!Number.isFinite(data.qpos[0]) || !Number.isFinite(data.qpos[2]) || Math.abs(data.qpos[2]) > 50) {
        state.blowups = (state.blowups || 0) + 1;
        status(`물리 발산 감지 (${state.blowups}회) → 6족 자세로 자동 리셋했습니다.`);
        reset('stance'); break;
      }
      simTime += dt; budget -= dt;
      substep = (substep + 1) % E.n_substeps;
      if (substep === 0) {
        updateGait();
        const midFcNow = [minLow(midL) < 0.003 ? 1 : 0, minLow(midR) < 0.003 ? 1 : 0];
        if (midFcNow[0] && !state.lastMidFc[0]) dropFootprint(midL);
        if (midFcNow[1] && !state.lastMidFc[1]) dropFootprint(midR);
        state.lastMidFc = midFcNow;

        const fcNow = (minLow(hindL) < 0.003 ? 1 : 0) + (minLow(hindR) < 0.003 ? 1 : 0);
        if (goalDist() < E.goal.reach_radius && contacts().bipedal && fcNow >= 1) { state.reached++; sampleGoal(); flash(); }
        if (state.trail.length === 0 || simTime - state.trail[state.trail.length - 1][0] > 0.02) state.trail.push([simTime, ...thoraxPos()]);
      }
    }
    // fade footprints
    for (const fp of footprints) {
      if (fp.mesh.visible) {
        fp.time -= elapsed;
        if (fp.time <= 0) fp.mesh.visible = false;
        else fp.mesh.material.opacity = 0.8 * (fp.time / 5.0);
      }
    }
  }
  syncBodies();
  const n = Math.min(state.trail.length, 2000);
  for (let i = 0; i < n; i++) { const p = state.trail[state.trail.length - n + i]; trailPos[3 * i] = p[1]; trailPos[3 * i + 1] = p[3] + 0.005; trailPos[3 * i + 2] = -p[2]; }
  trailGeom.setDrawRange(0, n); trailGeom.attributes.position.needsUpdate = true;
  const p = thoraxPos(); const target = new THREE.Vector3(p[0], p[2], -p[1]);
  if (follow) controls.target.lerp(target, 0.1);
  controls.update();
  sunTarget.position.copy(target); sun.position.set(target.x + 1.5, target.y + 3, target.z + 1);
  // HUD (updated 4x per second so the numbers are readable)
  if (now - lastHud < 250) { renderer.render(scene, camera); return; }
  lastHud = now;
  $('time').textContent = simTime.toFixed(2) + ' s';
  $('dist').textContent = goalDist().toFixed(2) + ' cm'; $('goals').textContent = state.reached;
  $('height').textContent = p[2].toFixed(3) + ' cm';
  { const g = state.gait, td = g.alt + g.same + g.both;
    $('gait').textContent = td ? `교대 ${(100 * g.alt / td).toFixed(0)}% · 동시착지 ${g.both} · 공중 ${(100 * g.flight / Math.max(1, g.walk)).toFixed(0)}%` : '-'; }
  $('achg').textContent = (state.actionChange || 0).toFixed(3) + (state.actionChange < 0.01 ? ' (정책이 정지 출력)' : '');
  $('see').textContent = seeGoal()[0] ? '보임' : '시야 밖';
  if (!paused) { const r = (simTime - simAtWall0) / ((now - wall0) / 1000); $('rt').textContent = isFinite(r) && r > 0 ? r.toFixed(2) + 'x' : '-'; }
  renderer.render(scene, camera);
}
function flash() { $('goals').classList.add('flash'); setTimeout(() => $('goals').classList.remove('flash'), 600); }
status('실행 중. 마우스 드래그로 회전, 휠로 확대. 노란 선은 이동 궤적, 빨간 공이 목표.');
requestAnimationFrame(frame);
