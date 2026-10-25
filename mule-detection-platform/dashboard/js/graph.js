'use strict';
/* Graph rendering: force-directed layout + zoom/pan SVG */
const NCOL = { HIGH: '#DC2626', MEDIUM: '#D97706', LOW: '#15803D', suspicious: '#DC2626', selected: '#D97706', normal: '#2563EB' };

function layout(nodes, edges, W, H) {
  const n = nodes.length, P = {}, k = Math.sqrt(W * H / Math.max(n, 1)) * .8;
  nodes.forEach((v, i) => { const a = 2 * Math.PI * i / n; P[v.id] = { x: W / 2 + Math.cos(a) * W * .35, y: H / 2 + Math.sin(a) * H * .35, vx: 0, vy: 0 }; });
  for (let it = 0; it < 260; it++) {
    const c = 1 - it / 260;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const a = P[nodes[i].id], b = P[nodes[j].id]; if (!a || !b) continue;
      let dx = a.x - b.x, dy = a.y - b.y; const d = Math.hypot(dx, dy) || .01, f = k * k / d; dx /= d; dy /= d;
      a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    }
    edges.forEach(e => {
      const a = P[e.from], b = P[e.to]; if (!a || !b) return;
      let dx = b.x - a.x, dy = b.y - a.y; const d = Math.hypot(dx, dy) || .01, f = d * d / k * .6; dx /= d; dy /= d;
      a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    });
    nodes.forEach(v => {
      const p = P[v.id]; if (!p) return;
      p.vx += (W / 2 - p.x) * .05; p.vy += (H / 2 - p.y) * .05;
      const m = Math.hypot(p.vx, p.vy) || 1, s = Math.min(m, 30 * c + 2);
      p.x += p.vx / m * s; p.y += p.vy / m * s; p.vx = p.vy = 0;
    });
  }
  return P;
}

function graphSvg(nodes, edges) {
  if (!nodes || !nodes.length) return '<p class="sub">No network data. Enter an account ID and load network.</p>';
  const W = 900, H = 560, P = layout(nodes, edges, W, H);
  const mx = Math.max(...edges.map(e => e.value || 1), 1);
  const deg = {};
  edges.forEach(e => { deg[e.from] = (deg[e.from] || 0) + 1; deg[e.to] = (deg[e.to] || 0) + 1; });
  const R = Object.fromEntries(nodes.map(v => [v.id, (v.size || 15) * .8 + Math.min(10, (deg[v.id] || 0) * 1.3)]));

  const es = edges.map(e => {
    const a = P[e.from], b = P[e.to]; if (!a || !b) return '';
    const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1;
    const x1 = a.x + dx / d * R[e.from], y1 = a.y + dy / d * R[e.from];
    const x2 = b.x - dx / d * (R[e.to] + 4), y2 = b.y - dy / d * (R[e.to] + 4);
    const w = 1 + 3 * (e.value || 1) / mx;
    return `<g class="edge"><line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke-width="${w}" marker-end="url(#ar)"><title>${esc(e.from)} → ${esc(e.to)}: ${fmt(e.value)}</title></line><line class="hit" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/></g>`;
  }).join('');

  const ns = nodes.map(v => {
    const p = P[v.id]; if (!p) return '';
    const color = v.color || NCOL[v.type] || NCOL.normal;
    return `<g class="node" data-act="gnode" data-id="${esc(v.id)}" tabindex="0"><circle cx="${p.x}" cy="${p.y}" r="${R[v.id]}" fill="${color}"><title>${esc(v.id)}</title></circle><text x="${p.x}" y="${p.y + 3}" text-anchor="middle">${esc(String(v.id).slice(-5))}</text></g>`;
  }).join('');

  return `<div class="gv"><div class="gtools"><button data-g="in">＋</button><button data-g="out">－</button><button data-g="fit">Fit</button></div><svg viewBox="0 0 ${W} ${H}"><defs><marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0L8,4L0,8z" fill="#64748B"/></marker></defs>${es}${ns}</svg></div>
  <div class="legend"><span><i style="background:#DC2626"></i>Suspicious</span><span><i style="background:#D97706"></i>Selected</span><span><i style="background:#2563EB"></i>Connected</span><span>Scroll to zoom · Drag to pan · Click node</span></div>`;
}

function initGraphs() {
  $$('.gv').forEach(g => {
    const svg = ('svg', g); if (!svg) return;
    const o = svg.getAttribute('viewBox').split(' ').map(Number); let vb = [...o], drag = null, moved = false;
    const set = () => svg.setAttribute('viewBox', vb.join(' '));
    const zoom = (f, cx = vb[0] + vb[2] / 2, cy = vb[1] + vb[3] / 2) => { vb = [cx - (cx - vb[0]) * f, cy - (cy - vb[1]) * f, vb[2] * f, vb[3] * f]; set(); };
    svg.addEventListener('wheel', e => { e.preventDefault(); zoom(e.deltaY > 0 ? 1.15 : .87); }, { passive: false });
    svg.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, vb: [...vb] }; moved = false; });
    svg.addEventListener('pointermove', e => {
      if (!drag) return;
      const k = 1 / svg.getScreenCTM().a, dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) moved = true;
      if (moved) { vb = [drag.vb[0] - dx * k, drag.vb[1] - dy * k, vb[2], vb[3]]; set(); }
    });
    const up = () => { drag = null; }; svg.addEventListener('pointerup', up); svg.addEventListener('pointerleave', up);
    svg.addEventListener('click', e => { if (moved) { e.stopPropagation(); e.preventDefault(); moved = false; } }, true);
    $$$$('[data-g]', g).forEach(b => b.onclick = () => {
      const a = b.dataset.g;
      if (a === 'in') zoom(.8); else if (a === 'out') zoom(1.25); else { vb = [...o]; set(); }
    });
  });
}