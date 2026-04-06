"""Three.js-based 3D mesh viewer for Jupyter notebooks.

Embeds an interactive WebGL viewer directly in notebook output cells using
Three.js (loaded from CDN).  Zero Python dependencies beyond the standard
library and IPython — no VTK, no trame, no meshio required.

Features:
    - Parses Gmsh ``.msh`` files (v2.x and v4.x) client-side in JavaScript
    - Tetrahedral → surface extraction with correct outward normals
    - BVH spatial chunking with per-frame frustum & screen-size culling
    - Per-physical-group toggle buttons
    - CAD-standard camera presets (Iso, Top, Bottom, Front, Back, Right, Left)

Based on a custom standalone Three.js mesh viewer.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

__all__ = ["plot_mesh_threejs"]


def plot_mesh_threejs(
    msh_path: str | Path,
    *,
    height: int = 600,
) -> object:
    """Display a ``.msh`` file in an interactive Three.js viewer.

    The viewer is rendered inline as an HTML widget — works in JupyterLab,
    VS Code notebooks, and Google Colab with zero server-side dependencies.

    Args:
        msh_path: Path to a Gmsh ``.msh`` file (ASCII, v2.x or v4.x).
        height: Pixel height of the viewer widget.

    Returns:
        An ``IPython.display.HTML`` object.  In a notebook the viewer is
        displayed automatically; in a script call ``display(result)``.
    """
    from IPython.display import HTML

    msh_text = Path(msh_path).read_text()
    msh_json = json.dumps(msh_text)
    cid = f"gsim-mesh-{uuid.uuid4().hex[:8]}"

    html = _TEMPLATE.format(cid=cid, height=height, msh_json=msh_json)
    return HTML(html)


# ---------------------------------------------------------------------------
# HTML / JS template
# ---------------------------------------------------------------------------
# Uses {{/}} for literal braces inside the f-string, and {cid}/{height}/{msh_json}
# for Python-interpolated values.

_TEMPLATE = """\
<div id="{cid}" style="width:100%;height:{height}px;position:relative;background:#111;border-radius:8px;overflow:hidden;">
  <div id="{cid}-panel" style="position:absolute;top:10px;left:10px;background:rgba(0,0,0,.7);backdrop-filter:blur(10px);padding:12px 16px;border-radius:8px;color:#fff;border:1px solid rgba(255,255,255,.1);z-index:10;font:12px/1.5 system-ui,sans-serif;max-width:280px;">
    <button id="{cid}-foldBtn" style="background:none;border:none;cursor:pointer;padding:0;line-height:1;margin-bottom:6px;" title="Collapse panel"><svg width="20" height="18" viewBox="0 0 20 18" fill="none" stroke="#9ca3af" stroke-width="1.2" stroke-linejoin="round"><path d="M1 17L10 1L19 17Z"/><path d="M5.5 9L14.5 9"/><path d="M10 1L5.5 9L10 17"/><path d="M10 1L14.5 9L10 17"/></svg></button>
    <div id="{cid}-body">
      <div id="{cid}-views" style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid #333;"></div>
      <div id="{cid}-vertCount">Vertices: &mdash;</div>
      <div id="{cid}-faceCount">Triangles: &mdash;</div>
      <div id="{cid}-chunkCount">BVH Chunks: &mdash;</div>
      <div style="margin-top:6px;padding-top:6px;border-top:1px solid #333;">
        <div id="{cid}-drawCalls">Draw Calls: &mdash;</div>
        <div id="{cid}-liveTris">Rendered: &mdash;</div>
      </div>
      <div id="{cid}-options" style="margin-top:6px;padding-top:6px;border-top:1px solid #333;">
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:11px;color:#d1d5db;">
          <input type="checkbox" id="{cid}-transpCb" style="accent-color:#3b82f6;cursor:pointer;">
          Transparent
        </label>
      </div>
      <div id="{cid}-groups" style="margin-top:6px;padding-top:6px;border-top:1px solid #333;display:none;">
        <div style="color:#aaa;margin-bottom:4px;font-weight:600;">Physical Groups</div>
        <div id="{cid}-groupList" style="display:flex;flex-direction:column;gap:3px;"></div>
      </div>
      <div style="margin-top:6px;color:#60a5fa;font-size:11px;">Left-click rotate &middot; scroll zoom &middot; right-click pan</div>
    </div>
  </div>
  <div id="{cid}-canvas" style="width:100%;height:100%;"></div>
</div>

<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import {{ OrbitControls }} from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

/* ── constants ─────────────────────────────────────────────────────── */
const MAX_TRIS_PER_CHUNK = 2048;
const MIN_PROJ_PX        = 1.0;
const _cv                = new THREE.Vector3();

/* ── DOM refs ──────────────────────────────────────────────────────── */
const C   = "{cid}";
const el  = id => document.getElementById(C + '-' + id);
const box = el('canvas');

/* ── state ─────────────────────────────────────────────────────────── */
let model = null, totTris = 0, radius = 1;
const matGroups = new Map();
const bvh       = [];

/* ── helpers ───────────────────────────────────────────────────────── */
function faceKey(a, b, c) {{
  if (a > b) {{ let t = a; a = b; b = t; }}
  if (b > c) {{ let t = b; b = c; c = t; }}
  if (a > b) {{ let t = a; a = b; b = t; }}
  return `${{a}},${{b}},${{c}}`;
}}

/* ── scene ─────────────────────────────────────────────────────────── */
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const W = box.clientWidth, H = box.clientHeight;
const cam = new THREE.PerspectiveCamera(60, W / H, .01, 10000);
cam.position.set(5, 5, 5);

const ren = new THREE.WebGLRenderer({{ antialias: true }});
ren.setSize(W, H);
ren.setPixelRatio(devicePixelRatio);
box.appendChild(ren.domElement);

const ctrl = new OrbitControls(cam, ren.domElement);
ctrl.enableDamping  = true;
ctrl.dampingFactor  = 0.05;
ctrl.enableZoom     = false;  /* we handle zoom ourselves for cursor-targeted zoom */

/* ── zoom toward mouse cursor ──────────────────────────────────────── */
const _mouse = new THREE.Vector2();
const _zoomDir = new THREE.Vector3();

ren.domElement.addEventListener('wheel', e => {{
  e.preventDefault();
  if (!model) return;

  const rect = ren.domElement.getBoundingClientRect();
  _mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  _mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;

  /* build a ray from camera through the mouse position */
  _zoomDir.set(_mouse.x, _mouse.y, 0.5).unproject(cam).sub(cam.position).normalize();

  /* scale step by current distance to target — feels proportional at any zoom */
  const dist = cam.position.distanceTo(ctrl.target);
  const step = dist * (e.deltaY > 0 ? -0.015 : 0.015);

  cam.position.addScaledVector(_zoomDir, step);
  ctrl.target.addScaledVector(_zoomDir, step);
  ctrl.update();
}}, {{ passive: false }});

/* ── CAD view presets ──────────────────────────────────────────────── */
const VIEWS = {{
  Iso:    {{ d: [ 1,  1,  1], u: [0, 0, 1] }},
  Top:    {{ d: [ 0,  0,  1], u: [0, 1, 0] }},
  Bottom: {{ d: [ 0,  0, -1], u: [0, 1, 0] }},
  Front:  {{ d: [ 0, -1,  0], u: [0, 0, 1] }},
  Back:   {{ d: [ 0,  1,  0], u: [0, 0, 1] }},
  Right:  {{ d: [ 1,  0,  0], u: [0, 0, 1] }},
  Left:   {{ d: [-1,  0,  0], u: [0, 0, 1] }},
}};

function setView(name) {{
  const v = VIEWS[name];
  if (!v) return;
  const dir = new THREE.Vector3(...v.d).normalize().multiplyScalar(radius * 2.5);
  cam.position.copy(dir);
  cam.up.set(...v.u);
  cam.lookAt(0, 0, 0);
  ctrl.target.set(0, 0, 0);
  ctrl.update();
}}

// build view buttons
const vBar = el('views');
for (const n of Object.keys(VIEWS)) {{
  const b = document.createElement('button');
  b.textContent = n;
  b.style.cssText = 'padding:2px 7px;border-radius:3px;border:none;font-size:10px;font-weight:600;cursor:pointer;background:#374151;color:#d1d5db;';
  b.onmouseenter = () => {{ b.style.background = '#4b5563'; b.style.color = '#fff'; }};
  b.onmouseleave = () => {{ b.style.background = '#374151'; b.style.color = '#d1d5db'; }};
  b.onclick      = () => setView(n);
  vBar.appendChild(b);
}}

/* ── fold/unfold panel ─────────────────────────────────────────────── */
const foldBtn  = el('foldBtn');
const bodyDiv  = el('body');
foldBtn.onclick = () => {{
  const open = bodyDiv.style.display !== 'none';
  bodyDiv.style.display  = open ? 'none' : 'block';
  foldBtn.style.opacity  = open ? '0.5' : '1';
  foldBtn.title          = open ? 'Expand panel' : 'Collapse panel';
}};

/* ── Gmsh parser ───────────────────────────────────────────────────── */
function parseGmsh(text) {{
  const lines = text.split(/\\r?\\n/);
  let idx = 0;
  const skip = tag => {{ while (idx < lines.length && lines[idx].trim() !== tag) idx++; idx++; }};
  const next = ()  => {{ while (idx < lines.length && lines[idx].trim() === '') idx++; return lines[idx++].trim(); }};

  idx = 0; skip('$MeshFormat');
  const ver = parseInt(next().split(/\\s+/)[0]);
  if (ver !== 2 && ver !== 4) throw new Error('Unsupported Gmsh v' + ver);

  /* physical names */
  const phys = new Map();
  if (text.includes('$PhysicalNames')) {{
    idx = 0; skip('$PhysicalNames');
    let n = parseInt(next());
    while (n-- > 0) {{
      const p = next().split(/\\s+/);
      phys.set(`${{p[0]}}_${{p[1]}}`, p.slice(2).join(' ').replace(/^"|"$/g, ''));
    }}
  }}

  /* entities (v4) */
  const entMap = new Map();
  if (ver === 4 && text.includes('$Entities')) {{
    idx = 0; skip('$Entities');
    const [nP, nC, nS, nV] = next().split(/\\s+/).map(Number);
    for (let i = 0; i < nP; i++) next();
    for (let i = 0; i < nC; i++) next();
    for (let i = 0; i < nS; i++) {{ const p = next().split(/\\s+/).map(Number); if (p[7] > 0) entMap.set('2_' + p[0], p[8]); }}
    for (let i = 0; i < nV; i++) {{ const p = next().split(/\\s+/).map(Number); if (p[7] > 0) entMap.set('3_' + p[0], p[8]); }}
  }}

  /* nodes */
  idx = 0; skip('$Nodes');
  const nodes = new Map();
  if (ver === 2) {{
    let n = parseInt(next());
    while (n-- > 0) {{ const p = next().split(/\\s+/); nodes.set(+p[0], [+p[1], +p[2], +p[3]]); }}
  }} else {{
    let nB = parseInt(next().split(/\\s+/)[0]);
    while (nB-- > 0) {{
      const bh = next().split(/\\s+/); let c = +bh[3]; const tags = [];
      for (let i = 0; i < c; i++) tags.push(+next());
      for (let i = 0; i < c; i++) {{ const p = next().split(/\\s+/); nodes.set(tags[i], [+p[0], +p[1], +p[2]]); }}
    }}
  }}

  /* elements */
  idx = 0; skip('$Elements');
  const tris = [], tG = [], tets = [], tetG = [];
  if (ver === 2) {{
    let n = parseInt(next());
    while (n-- > 0) {{
      const p = next().split(/\\s+/).map(Number); const ty = p[1], ns = 3 + p[2];
      if (ty === 2)      {{ tris.push(p[ns], p[ns+1], p[ns+2]); tG.push('2_' + (p[2] > 0 ? p[3] : 0)); }}
      else if (ty === 4) {{ tets.push(p[ns], p[ns+1], p[ns+2], p[ns+3]); tetG.push('3_' + (p[2] > 0 ? p[3] : 0)); }}
    }}
  }} else {{
    let nB = parseInt(next().split(/\\s+/)[0]);
    while (nB-- > 0) {{
      const bh = next().split(/\\s+/); const eDim = +bh[0], eTag = +bh[1], ty = +bh[2]; let c = +bh[3];
      const pt = entMap.get(eDim + '_' + eTag) ?? 0, gk = eDim + '_' + pt;
      while (c-- > 0) {{
        const p = next().split(/\\s+/).map(Number);
        if (ty === 2)      {{ tris.push(p[1], p[2], p[3]); tG.push(gk); }}
        else if (ty === 4) {{ tets.push(p[1], p[2], p[3], p[4]); tetG.push(gk); }}
      }}
    }}
  }}

  /* build per-group face lists */
  const gf = new Map();
  const add = (g, a, b, c) => {{ if (!gf.has(g)) gf.set(g, []); gf.get(g).push(a, b, c); }};

  if (tets.length) {{
    const {{ groupFaces: tf, orientMap: om }} = tetSurface(tets, tetG, nodes);
    const seen = new Set();
    for (let i = 0; i < tris.length; i += 3) {{
      const k = faceKey(tris[i], tris[i+1], tris[i+2]); seen.add(k);
      const o = om.get(k); const [a,b,c] = o ?? [tris[i], tris[i+1], tris[i+2]];
      add(tG[i/3], a, b, c);
    }}
    for (const [g, f] of tf) for (let i = 0; i < f.length; i += 3)
      if (!seen.has(faceKey(f[i], f[i+1], f[i+2]))) add(g, f[i], f[i+1], f[i+2]);
  }} else {{
    for (let i = 0; i < tris.length; i += 3) add(tG[i/3] ?? '2_0', tris[i], tris[i+1], tris[i+2]);
  }}

  if (!gf.size) throw new Error('No surface elements found.');

  /* geometry per group */
  const geoms = new Map();
  for (const [g, faces] of gf) {{
    const nT = faces.length / 3, pos = new Float32Array(nT * 9);
    for (let t = 0; t < nT; t++) for (let v = 0; v < 3; v++) {{
      const node = nodes.get(faces[t*3+v]);
      pos[t*9+v*3] = node[0]; pos[t*9+v*3+1] = node[1]; pos[t*9+v*3+2] = node[2];
    }}
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geoms.set(g, geo);
  }}
  return {{ geoms, phys }};
}}

/* ── tet surface extraction ────────────────────────────────────────── */
function tetSurface(tets, tetG, nodes) {{
  const fd = new Map();
  const TF = [[0,1,2],[0,1,3],[0,2,3],[1,2,3]], OPP = [3,2,1,0];
  for (let t = 0; t < tets.length / 4; t++) {{
    const b = t * 4, ns = [tets[b], tets[b+1], tets[b+2], tets[b+3]], g = tetG[t];
    for (let fi = 0; fi < 4; fi++) {{
      const [a, bb, c] = TF[fi], k = faceKey(ns[a], ns[bb], ns[c]);
      if (!fd.has(k)) {{
        const pa = nodes.get(ns[a]), pb = nodes.get(ns[bb]), pc = nodes.get(ns[c]), pd = nodes.get(ns[OPP[fi]]);
        const ex = pb[0]-pa[0], ey = pb[1]-pa[1], ez = pb[2]-pa[2];
        const fx = pc[0]-pa[0], fy = pc[1]-pa[1], fz = pc[2]-pa[2];
        const nx = ey*fz-ez*fy, ny = ez*fx-ex*fz, nz = ex*fy-ey*fx;
        const dot = nx*(pd[0]-pa[0])+ny*(pd[1]-pa[1])+nz*(pd[2]-pa[2]);
        fd.set(k, {{ cnt: 0, n: dot > 0 ? [ns[a], ns[c], ns[bb]] : [ns[a], ns[bb], ns[c]], g }});
      }} else {{
        const d = fd.get(k); if (d.g2 === undefined) d.g2 = g;
      }}
      fd.get(k).cnt++;
    }}
  }}
  const groupFaces = new Map(), orientMap = new Map();
  const push = (g, n) => {{ if (!groupFaces.has(g)) groupFaces.set(g, []); groupFaces.get(g).push(...n); }};
  for (const [k, {{ cnt, n, g, g2 }}] of fd) {{
    if (cnt === 1)                    {{ push(g, n); orientMap.set(k, n); }}
    else if (cnt === 2 && g !== g2) {{ push(g, n); push(g2, n); orientMap.set(k, n); }}
  }}
  return {{ groupFaces, orientMap }};
}}

/* ── BVH chunker ───────────────────────────────────────────────────── */
function buildBVH(geo, fMat, wMat) {{
  const g = geo.index ? geo.toNonIndexed() : geo;
  const pos = g.attributes.position.array;
  const nT = (pos.length / 9) | 0;
  if (!nT) return [];

  const idx = new Int32Array(nT);
  const cx = new Float32Array(nT), cy = new Float32Array(nT), cz = new Float32Array(nT);
  for (let i = 0; i < nT; i++) {{
    idx[i] = i; const b = i * 9;
    cx[i] = (pos[b]+pos[b+3]+pos[b+6])/3;
    cy[i] = (pos[b+1]+pos[b+4]+pos[b+7])/3;
    cz[i] = (pos[b+2]+pos[b+5]+pos[b+8])/3;
  }}

  const out = [];
  function split(s, e, depth) {{
    if (e - s <= MAX_TRIS_PER_CHUNK || depth > 24) {{ out.push(leaf(s, e)); return; }}
    let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity,z0=Infinity,z1=-Infinity;
    for (let i = s; i < e; i++) {{
      const t = idx[i];
      if (cx[t]<x0) x0=cx[t]; if (cx[t]>x1) x1=cx[t];
      if (cy[t]<y0) y0=cy[t]; if (cy[t]>y1) y1=cy[t];
      if (cz[t]<z0) z0=cz[t]; if (cz[t]>z1) z1=cz[t];
    }}
    const dx=x1-x0, dy=y1-y0, dz=z1-z0;
    let ax = 0;
    if (dy>dx && dy>=dz) ax=1; else if (dz>dx && dz>dy) ax=2;
    const c = ax===0?cx:ax===1?cy:cz;
    idx.subarray(s, e).sort((a, b) => c[a] - c[b]);
    const m = s + ((e-s)>>1);
    split(s, m, depth+1); split(m, e, depth+1);
  }}

  function leaf(s, e) {{
    const n = e - s, v = new Float32Array(n * 9);
    for (let i = 0; i < n; i++) {{ const src = idx[s+i]*9, dst = i*9; for (let j=0;j<9;j++) v[dst+j]=pos[src+j]; }}
    const cg = new THREE.BufferGeometry();
    cg.setAttribute('position', new THREE.BufferAttribute(v, 3));
    cg.computeVertexNormals();
    cg.computeBoundingSphere();
    const mesh = new THREE.Mesh(cg, fMat);
    mesh.add(new THREE.LineSegments(new THREE.WireframeGeometry(cg), wMat));
    mesh.userData.bsc = cg.boundingSphere.center.clone();
    return mesh;
  }}

  split(0, nT, 0);
  return out;
}}

/* ── display ───────────────────────────────────────────────────────── */
const VOL_OPACITY = 0.3;

function display(geoms, phys) {{
  /* surface groups (2D) — own material so transparency can be toggled independently */
  const fMat  = new THREE.MeshNormalMaterial({{ side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }});
  /* volume groups (3D) — separate material instance */
  const fMatT = new THREE.MeshNormalMaterial({{ side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }});
  const wMat  = new THREE.LineBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.4 }});
  const wMatT = new THREE.LineBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.4 }});

  const root = new THREE.Group();
  let verts = 0, tris = 0, chunks = 0;
  for (const [g, geo] of geoms) {{
    geo.computeVertexNormals();
    verts += geo.attributes.position.count;
    tris  += (geo.attributes.position.count / 3) | 0;
    const isVol = g.startsWith('3_');
    const gr = new THREE.Group();
    buildBVH(geo, isVol ? fMatT : fMat, isVol ? wMatT : wMat).forEach(c => {{ gr.add(c); bvh.push(c); }});
    chunks += gr.children.length;
    root.add(gr);
    matGroups.set(g, gr);
    geo.dispose();
  }}

  new THREE.Box3().setFromObject(root).getCenter(root.position).negate();
  scene.add(root);
  model = root;

  const sp = new THREE.Box3().setFromObject(root).getBoundingSphere(new THREE.Sphere());
  radius = sp.radius;
  ctrl.reset();
  setView('Iso');
  cam.near = radius / 100; cam.far = radius * 100;
  cam.updateProjectionMatrix();
  totTris = tris;

  el('vertCount').textContent  = 'Vertices: '   + verts.toLocaleString();
  el('faceCount').textContent  = 'Triangles: '  + tris.toLocaleString();
  el('chunkCount').textContent = 'BVH Chunks: ' + chunks.toLocaleString();

  /* transparency checkbox — applies to ALL groups */
  el('transpCb').addEventListener('change', e => {{
    const on = e.target.checked;
    fMat.opacity      = on ? VOL_OPACITY : 1.0;
    fMat.transparent  = on;
    fMat.depthWrite   = !on;
    fMat.needsUpdate  = true;
    fMatT.opacity     = on ? VOL_OPACITY : 1.0;
    fMatT.transparent = on;
    fMatT.depthWrite  = !on;
    fMatT.needsUpdate = true;
    wMat.opacity      = on ? 0.15 : 0.4;
    wMat.needsUpdate  = true;
    wMatT.opacity     = on ? 0.15 : 0.4;
    wMatT.needsUpdate = true;
  }});

  /* should this group be hidden by default? */
  function hideByDefault(name) {{
    const lo = name.toLowerCase();
    return /sio2|oxide|clad|box|passive|silicon|nitride|air|vacuum/i.test(lo)
        || (lo.includes('__'));
  }}

  /* group toggles */
  if (matGroups.size) {{
    el('groups').style.display = 'block';
    for (const [g, gr] of matGroups) {{
      const name = phys.get(g) ?? ('Group ' + (g.split('_')[1] ?? g));
      const hide = hideByDefault(name);
      if (hide) gr.visible = false;
      const b = document.createElement('button');
      b.textContent = name;
      b.style.cssText = 'padding:3px 8px;border-radius:4px;border:none;font-size:11px;font-weight:600;cursor:pointer;text-align:left;background:' + (hide ? '#374151' : '#1d4ed8') + ';color:#fff;';
      b.onclick = () => {{ gr.visible = !gr.visible; b.style.background = gr.visible ? '#1d4ed8' : '#374151'; }};
      el('groupList').appendChild(b);
    }}
  }}
}}

/* ── render loop ───────────────────────────────────────────────────── */
function cull() {{
  if (!model) return;
  const fov = cam.fov * Math.PI / 180;
  const hH  = ren.domElement.height / (2 * Math.tan(fov / 2));
  for (const c of bvh) {{
    _cv.copy(c.userData.bsc).add(model.position);
    const d = cam.position.distanceTo(_cv);
    c.visible = (c.geometry.boundingSphere.radius / d) * hH >= MIN_PROJ_PX;
  }}
}}

(function loop() {{
  requestAnimationFrame(loop);
  ctrl.update();
  /* keep near plane proportional to camera distance so close-up zoom works */
  if (radius > 0) {{
    const dist = cam.position.length();
    cam.near = Math.max(dist * 0.001, radius * 0.0001);
    cam.far  = radius * 100;
    cam.updateProjectionMatrix();
  }}
  cull();
  ren.render(scene, cam);
  if (model) {{
    const r = ren.info.render, drawn = r.triangles, pct = totTris ? Math.round(Math.max(0, totTris - drawn) / totTris * 100) : 0;
    el('drawCalls').textContent = 'Draw Calls: ' + r.calls.toLocaleString();
    el('liveTris').textContent  = 'Rendered: '   + drawn.toLocaleString() + ' (' + pct + '% culled)';
  }}
}})();

/* ── load ──────────────────────────────────────────────────────────── */
try {{
  const {{ geoms, phys }} = parseGmsh({msh_json});
  display(geoms, phys);
}} catch (err) {{
  console.error('[gsim mesh viewer]', err);
  el('vertCount').textContent = 'Error: ' + err.message;
  el('vertCount').style.color = '#f87171';
}}

/* ── resize ────────────────────────────────────────────────────────── */
new ResizeObserver(() => {{
  const w = box.clientWidth, h = box.clientHeight;
  cam.aspect = w / h; cam.updateProjectionMatrix();
  ren.setSize(w, h);
}}).observe(box);
</script>
"""
