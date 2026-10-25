'use strict';
/* ============================================================
   CORE: API client, shared state, formatting, UI primitives
   ============================================================ */
const API = '';
let S = { stats: {}, accounts: [], hotspots: [] };
let D = { acct: null, trail: null, net: null, report: null, cmpReport: null, trailReport: null };
const st = {
  tab: 'Overview',
  acc: { q: '', band: '', page: 1 },
  cmp: { q: '', page: 1 },
  mt: { acct: '', depth: 3 },
  net: { acct: '', depth: 2 },
  repAcct: '', repCmp: '', repTrailAcct: ''
};

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = n => Math.round(n || 0).toLocaleString('en-IN');
const pct = n => ((n || 0) * 100).toFixed(1) + '%';
const dtm = s => s ? String(s).replace('T', ' ').slice(0, 16) : '—';
const inr = n => '₹' + Math.round(n || 0).toLocaleString('en-IN');

async function apiGet(p) {
  try {
    const r = await fetch(API + p);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return await r.json();
  } catch (e) {
    console.error('[API]', p, e);
    return null;
  }
}

function toast(m) {
  const t = $('#toast'); t.textContent = m; t.style.display = 'block';
  clearTimeout(toast.h); toast.h = setTimeout(() => t.style.display = 'none', 3200);
}

function modal(html) { const m = $('#modal'); m.innerHTML = '<div>' + html + '</div>'; m.style.display = 'flex'; }
const closeModal = () => { $('#modal').style.display = 'none'; };

function download(name, text, type = 'text/plain') {
  const u = URL.createObjectURL(new Blob([text], { type }));
  const l = document.createElement('a'); l.href = u; l.download = name;
  document.body.append(l); l.click(); l.remove();
  setTimeout(() => URL.revokeObjectURL(u), 1000);
}

/* UI fragments */
const LBL = { HIGH: '▲', MEDIUM: '◆', LOW: '▼' };
const pb = l => `<span class="b ${esc(l)}">${LBL[l] || ''} ${esc(l)}</span>`;
const sbar = s => s == null ? '<span class="sub">—</span>' : `<span class="sb"><i class="${s >= 70 ? 'Highbg' : s >= 40 ? 'Mediumbg' : 'Lowbg'}" style="width:${Math.min(s, 100)}%"></i></span> <b>${Math.round(s)}</b>`;

function tbl(h, rows, none = 'No records match the current filters.') {
  return rows.length
    ? `<div class="tw"><table><thead><tr>${h.map(x => `<th scope="col">${x}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`
    : `<p class="sub">${none}</p>`;
}

function pager(g, total, per = 20) {
  const n = Math.max(1, Math.ceil(total / per));
  if (st[g].page > n) st[g].page = n;
  const p = st[g].page;
  return `<div class="pager"><button class="btn g" data-act="pg" data-id="${g}|${p - 1}" ${p <= 1 ? 'disabled' : ''}>‹ Prev</button><span>Page ${p} of ${n} · ${fmt(total)} records</span><button class="btn g" data-act="pg" data-id="${g}|${p + 1}" ${p >= n ? 'disabled' : ''}>Next ›</button></div>`;
}

function pageOf(g, a, per = 20) { return a.slice((st[g].page - 1) * per, st[g].page * per); }

function donut(items) {
  const tot = items.reduce((a, i) => a + i[1], 0) || 1;
  const C = 2 * Math.PI * 40; let off = 0;
  const arcs = items.map(([l, n, c]) => {
    const len = n / tot * C;
    const s = `<circle r="40" cx="60" cy="60" fill="none" stroke="${c}" stroke-width="18" stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-off}"><title>${l}: ${n}</title></circle>`;
    off += len; return s;
  }).join('');
  return `<div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap"><svg viewBox="0 0 120 120" width="150"><g transform="rotate(-90 60 60)">${arcs}</g><text x="60" y="64" text-anchor="middle" style="font-size:14px">${fmt(tot)}</text></svg><div>${items.map(([l, n]) => `<p style="margin:3px 0">${pb(l)} ${fmt(n)}</p>`).join('')}</div></div>`;
}

/* ============================================================
   REPORT GENERATORS (TXT + HTML)
   ============================================================ */
function buildAccountReportTXT(data, acct) {
  const rs = data.risk_score || 0;
  let t = '';
  t += '══════════════════════════════════════════════════════════\n';
  t += '        FINANCIAL FRAUD INVESTIGATION REPORT\n';
  t += '           ACCOUNT RISK ASSESSMENT\n';
  t += '══════════════════════════════════════════════════════════\n\n';
  t += `Report ID:      RPT-${acct}-${Date.now().toString().slice(-6)}\n`;
  t += `Generated:      ${new Date().toLocaleString()}\n`;
  t += `Account ID:     ${acct}\n`;
  t += `Risk Score:     ${rs.toFixed(1)} / 100\n`;
  t += `Risk Band:      ${data.band || '—'}\n`;
  t += `Global Rank:    #${data.rank || '—'}\n\n`;

  t += '──────────────────────────────────────────────────────────\n';
  t += 'RISK BREAKDOWN\n';
  t += '──────────────────────────────────────────────────────────\n';
  t += `  Behavioural Risk: ${pct(data.p_behavioural)}\n`;
  t += `  Graph Risk:       ${pct(data.graph_risk)}\n`;
  t += `  Identity Risk:    ${pct(data.p_identity)}\n`;
  t += `  Fusion Formula:   R = 0.50×Behavioural + 0.30×Graph + 0.20×Identity\n\n`;

  if (data.reasons && data.reasons.length) {
    t += '──────────────────────────────────────────────────────────\n';
    t += 'RISK INDICATORS (SHAP ANALYSIS)\n';
    t += '──────────────────────────────────────────────────────────\n';
    data.reasons.forEach((r, i) => {
      t += `  ${i + 1}. ${r.description || r.feature}\n`;
      t += `     Impact: ${r.impact > 0 ? '+' : ''}${(r.impact || 0).toFixed(3)} | ${r.direction || ''}\n\n`;
    });
  }

  if (data.complaints && data.complaints.length) {
    t += '──────────────────────────────────────────────────────────\n';
    t += `LINKED COMPLAINTS (${data.complaints.length})\n`;
    t += '──────────────────────────────────────────────────────────\n';
    data.complaints.forEach(c => {
      t += `  [${c.complaint_id}] ${c.category} | Loss: ${inr(c.amount_lost)} | ${c.complaint_date || '—'}\n`;
    });
    t += '\n';
  }

  if (data.trail && data.trail.length) {
    t += '──────────────────────────────────────────────────────────\n';
    t += `FUND TRAIL (${data.trail.length} transactions)\n`;
    t += '──────────────────────────────────────────────────────────\n';
    const s = data.trail_summary || {};
    t += `  Total Outflow:       ${inr(s.total_outflow)}\n`;
    t += `  Unique Destinations: ${s.unique_destinations || 0}\n`;
    t += `  Transactions:        ${s.total_transactions || 0}\n\n`;
    data.trail.slice(0, 20).forEach(x => {
      t += `  ${x.from} → ${x.to} | ${inr(x.amount)} | ${x.time || '—'}\n`;
    });
    if (data.trail.length > 20) t += `  ... and ${data.trail.length - 20} more\n`;
    t += '\n';
  }

  t += '──────────────────────────────────────────────────────────\n';
  t += 'RECOMMENDATION\n';
  t += '──────────────────────────────────────────────────────────\n';
  t += rs >= 70 ? '  ESCALATE — High Priority Investigation Required\n'
    : rs >= 40 ? '  REVIEW — Monitor and Gather Additional Data\n'
      : '  LOW PRIORITY — Routine Monitoring Sufficient\n';

  t += '\n══════════════════════════════════════════════════════════\n';
  t += 'DISCLAIMER: AI-generated. Requires human review.\n';
  t += 'Risk indicators do not establish fraud or guilt.\n';
  t += `Generated: Mule Detection Platform | ${new Date().toLocaleDateString()}\n`;
  t += '══════════════════════════════════════════════════════════\n';
  return t;
}

function buildAccountReportHTML(data, acct) {
  const rs = data.risk_score || 0;
  const bc = rs >= 70 ? '#DC2626' : rs >= 40 ? '#D97706' : '#15803D';
  return `<!DOCTYPE html><html><head><meta charset="utf-8"><title>SAR Report - ${esc(acct)}</title>
<style>body{font:13px/1.6 Arial,sans-serif;max-width:900px;margin:20px auto;padding:0 20px;color:#172033}
h1{font-size:22px;margin:0;border-bottom:4px solid #14233B;padding-bottom:10px}
h2{font-size:16px;color:#14233B;border-bottom:2px solid #E2E8F0;padding-bottom:4px;margin-top:24px}
table{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0}th,td{border:1px solid #E2E8F0;padding:8px;text-align:left}th{background:#F4F7FB}
.m{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}
.mc{background:#F4F7FB;border:1px solid #E2E8F0;border-radius:8px;padding:14px;text-align:center}
.mc .v{font-size:28px;font-weight:700}.mc .l{font-size:11px;color:#64748B}
.ind{padding:10px 14px;margin:6px 0;border-radius:8px;border-left:4px solid}
.ind.p{background:#FEF2F2;border-color:#DC2626}.ind.n{background:#F0FDF4;border-color:#15803D}
.rec{border:2px solid ${bc};border-radius:12px;padding:20px;margin:20px 0;background:${rs >= 70 ? '#FEF2F2' : rs >= 40 ? '#FFFBEB' : '#F0FDF4'}}
.ft{border-top:2px solid #E2E8F0;padding-top:12px;margin-top:24px;font-size:11px;color:#64748B;text-align:center}
@media print{button{display:none}}</style></head><body>
<button onclick="window.print()" style="background:#2563EB;color:#fff;border:0;border-radius:6px;padding:10px 20px;font-size:14px;cursor:pointer;margin:10px 5px">🖨 Print / Save as PDF</button>
<h1>FINANCIAL FRAUD INVESTIGATION REPORT</h1>
<p style="color:#64748B">Mule Detection Platform · AI/ML + Graph Analytics · <b>CONFIDENTIAL</b></p>
<div class="m">
  <div class="mc"><div class="v" style="color:${bc}">${rs.toFixed(1)}</div><div class="l">Risk Score</div></div>
  <div class="mc"><div class="v">${esc(data.band)}</div><div class="l">Risk Band</div></div>
  <div class="mc"><div class="v">${data.complaints?.length || 0}</div><div class="l">Complaints</div></div>
  <div class="mc"><div class="v">#${data.rank || '—'}</div><div class="l">Global Rank</div></div>
</div>
<p><b>Report ID:</b> RPT-${esc(acct)}-${Date.now().toString().slice(-6)} · <b>Generated:</b> ${new Date().toLocaleString()} · <b>Account:</b> ${esc(acct)}</p>
<h2>1. Risk Breakdown</h2>
<table><tr><th>Component</th><th>Score</th></tr>
<tr><td>Behavioural Risk</td><td>${pct(data.p_behavioural)}</td></tr>
<tr><td>Graph Risk</td><td>${pct(data.graph_risk)}</td></tr>
<tr><td>Identity Risk</td><td>${pct(data.p_identity)}</td></tr>
<tr><td><b>Fusion Formula</b></td><td>R = 0.50×B + 0.30×G + 0.20×I</td></tr></table>
${data.reasons?.length ? `<h2>2. Risk Indicators (SHAP)</h2>${data.reasons.map(r => `<div class="ind ${r.impact > 0 ? 'p' : 'n'}"><b>${r.impact > 0 ? '⚠️' : '✅'} ${esc(r.description || r.feature)}</b><br><small>Feature: ${esc(r.feature)} | Impact: ${(r.impact || 0).toFixed(3)} | ${esc(r.direction || '')}</small></div>`).join('')}` : ''}
${data.complaints?.length ? `<h2>3. Linked Complaints (${data.complaints.length})</h2><table><tr><th>ID</th><th>Category</th><th>Amount Lost</th><th>Date</th></tr>${data.complaints.map(c => `<tr><td>${esc(c.complaint_id)}</td><td>${esc(c.category)}</td><td>${inr(c.amount_lost)}</td><td>${esc(c.complaint_date || '—')}</td></tr>`).join('')}</table>` : ''}
${data.trail?.length ? `<h2>4. Fund Trail (${data.trail.length} transactions)</h2><p><b>Total Outflow:</b> ${inr(data.trail_summary?.total_outflow)} · <b>Destinations:</b> ${data.trail_summary?.unique_destinations || 0}</p><table><tr><th>From</th><th>To</th><th>Amount</th><th>Time</th></tr>${data.trail.slice(0, 20).map(t => `<tr><td>${esc(t.from)}</td><td>${esc(t.to)}</td><td>${inr(t.amount)}</td><td>${esc(t.time || '—')}</td></tr>`).join('')}</table>` : ''}
<div class="rec"><b>5. RECOMMENDATION:</b><br>${rs >= 70 ? '🔴 ESCALATE — High Priority Investigation Required' : rs >= 40 ? '🟡 REVIEW — Monitor and Gather Additional Data' : '🟢 LOW PRIORITY — Routine Monitoring Sufficient'}</div>
<div class="ft"><p><b>DISCLAIMER:</b> AI-generated report. Requires human review. Risk indicators do not establish fraud or guilt.</p><p>Mule Detection Platform · ${new Date().toLocaleDateString()}</p></div>
</body></html>`;
}

function buildComplaintReportTXT(cmp, acctData) {
  let t = '';
  t += '══════════════════════════════════════════════════════════\n';
  t += '          COMPLAINT INVESTIGATION REPORT\n';
  t += '══════════════════════════════════════════════════════════\n\n';
  t += `Report ID:      RPT-CMP-${cmp.complaint_id}-${Date.now().toString().slice(-6)}\n`;
  t += `Generated:      ${new Date().toLocaleString()}\n\n`;
  t += '──────────────────────────────────────────────────────────\n';
  t += 'COMPLAINT DETAILS\n';
  t += '──────────────────────────────────────────────────────────\n';
  t += `  Complaint ID:   ${cmp.complaint_id}\n`;
  t += `  Category:       ${cmp.category}\n`;
  t += `  Reported Acct:  ${cmp.reported_acct}\n`;
  t += `  Victim Acct:    ${cmp.victim_acct}\n`;
  t += `  Amount Lost:    ${inr(cmp.amount_lost)}\n`;
  t += `  Date:           ${cmp.complaint_date}\n\n`;

  if (acctData && acctData.account) {
    const acc = acctData.account;
    t += '──────────────────────────────────────────────────────────\n';
    t += 'REPORTED ACCOUNT RISK PROFILE\n';
    t += '──────────────────────────────────────────────────────────\n';
    t += `  Account:        ${acc.acct}\n`;
    t += `  Risk Score:     ${(acc.risk_score || 0).toFixed(1)} / 100\n`;
    t += `  Risk Band:      ${acc.band}\n`;
    t += `  Rank:           #${acc.rank || '—'}\n`;
    t += `  Behavioural:    ${pct(acc.p_behavioural)}\n`;
    t += `  Graph Risk:     ${pct(acc.graph_risk)}\n\n`;

    if (acctData.reasons?.length) {
      t += '  SHAP INDICATORS:\n';
      acctData.reasons.forEach((r, i) => {
        t += `    ${i + 1}. ${r.description || r.feature} (${(r.impact || 0).toFixed(3)})\n`;
      });
      t += '\n';
    }

    if (acctData.complaints?.length > 1) {
      t += `  OTHER COMPLAINTS (${acctData.complaints.length - 1} more):\n`;
      acctData.complaints.filter(c => c.complaint_id !== cmp.complaint_id).forEach(c => {
        t += `    [${c.complaint_id}] ${c.category} | ${inr(c.amount_lost)}\n`;
      });
      t += '\n';
    }
  }

  t += '──────────────────────────────────────────────────────────\n';
  t += 'RECOMMENDATION\n';
  t += '──────────────────────────────────────────────────────────\n';
  t += '  Investigate reported account fund flows and network.\n';
  t += '  Correlate with other complaints against same account.\n';
  t += '  Trace fund movements for recovery.\n\n';
  t += '══════════════════════════════════════════════════════════\n';
  t += 'DISCLAIMER: AI-generated. Requires human review.\n';
  t += `Generated: Mule Detection Platform | ${new Date().toLocaleDateString()}\n`;
  t += '══════════════════════════════════════════════════════════\n';
  return t;
}

function buildComplaintReportHTML(cmp, acctData) {
  const acc = acctData?.account || {};
  const rs = acc.risk_score || 0;
  const bc = rs >= 70 ? '#DC2626' : rs >= 40 ? '#D97706' : '#15803D';
  return `<!DOCTYPE html><html><head><meta charset="utf-8"><title>Complaint Report - ${esc(cmp.complaint_id)}</title>
<style>body{font:13px/1.6 Arial,sans-serif;max-width:900px;margin:20px auto;padding:0 20px;color:#172033}
h1{font-size:22px;margin:0;border-bottom:4px solid #14233B;padding-bottom:10px}
h2{font-size:16px;color:#14233B;border-bottom:2px solid #E2E8F0;padding-bottom:4px;margin-top:24px}
table{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0}th,td{border:1px solid #E2E8F0;padding:8px;text-align:left}th{background:#F4F7FB}
.ind{padding:10px 14px;margin:6px 0;border-radius:8px;border-left:4px solid}
.ind.p{background:#FEF2F2;border-color:#DC2626}.ind.n{background:#F0FDF4;border-color:#15803D}
.ft{border-top:2px solid #E2E8F0;padding-top:12px;margin-top:24px;font-size:11px;color:#64748B;text-align:center}
@media print{button{display:none}}</style></head><body>
<button onclick="window.print()" style="background:#2563EB;color:#fff;border:0;border-radius:6px;padding:10px 20px;cursor:pointer;margin:10px 5px">🖨 Print / PDF</button>
<h1>COMPLAINT INVESTIGATION REPORT</h1>
<p style="color:#64748B">Mule Detection Platform · <b>CONFIDENTIAL</b></p>
<h2>1. Complaint Details</h2>
<table><tr><th>Field</th><th>Value</th></tr>
<tr><td>Complaint ID</td><td>${esc(cmp.complaint_id)}</td></tr>
<tr><td>Category</td><td>${esc(cmp.category)}</td></tr>
<tr><td>Reported Account</td><td>${esc(cmp.reported_acct)}</td></tr>
<tr><td>Victim Account</td><td>${esc(cmp.victim_acct)}</td></tr>
<tr><td>Amount Lost</td><td>${inr(cmp.amount_lost)}</td></tr>
<tr><td>Date</td><td>${esc(cmp.complaint_date)}</td></tr></table>
<h2>2. Reported Account Risk Profile</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Account</td><td>${esc(acc.acct || cmp.reported_acct)}</td></tr>
<tr><td>Risk Score</td><td style="color:${bc};font-weight:700">${rs.toFixed(1)} / 100</td></tr>
<tr><td>Risk Band</td><td>${pb(acc.band || '—')}</td></tr>
<tr><td>Rank</td><td>#${acc.rank || '—'}</td></tr>
<tr><td>Behavioural</td><td>${pct(acc.p_behavioural)}</td></tr>
<tr><td>Graph Risk</td><td>${pct(acc.graph_risk)}</td></tr></table>
${acctData?.reasons?.length ? `<h2>3. SHAP Risk Indicators</h2>${acctData.reasons.map(r => `<div class="ind ${r.impact > 0 ? 'p' : 'n'}"><b>${esc(r.description || r.feature)}</b> — Impact: ${(r.impact || 0).toFixed(3)}</div>`).join('')}` : ''}
${acctData?.complaints?.length > 1 ? `<h2>4. Other Complaints Against This Account (${acctData.complaints.length - 1})</h2><table><tr><th>ID</th><th>Category</th><th>Amount</th></tr>${acctData.complaints.filter(c => c.complaint_id !== cmp.complaint_id).map(c => `<tr><td>${esc(c.complaint_id)}</td><td>${esc(c.category)}</td><td>${inr(c.amount_lost)}</td></tr>`).join('')}</table>` : ''}
<h2>5. Recommendation</h2><p>Investigate reported account fund flows and network. Correlate with other complaints. Trace fund movements for recovery.</p>
<div class="ft"><p>DISCLAIMER: AI-generated. Requires human review.</p><p>Mule Detection Platform · ${new Date().toLocaleDateString()}</p></div>
</body></html>`;
}

function buildTrailReportTXT(trailData, acct) {
  const s = trailData.summary || {};
  let t = '';
  t += '══════════════════════════════════════════════════════════\n';
  t += '           MONEY TRAIL INVESTIGATION REPORT\n';
  t += '══════════════════════════════════════════════════════════\n\n';
  t += `Report ID:      RPT-TRL-${acct}-${Date.now().toString().slice(-6)}\n`;
  t += `Generated:      ${new Date().toLocaleString()}\n`;
  t += `Account:        ${acct}\n\n`;
  t += '──────────────────────────────────────────────────────────\n';
  t += 'TRAIL SUMMARY\n';
  t += '──────────────────────────────────────────────────────────\n';
  t += `  Total Transactions:  ${s.total_transactions || 0}\n`;
  t += `  Total Outflow:       ${inr(s.total_outflow)}\n`;
  t += `  Unique Destinations: ${s.unique_destinations || 0}\n\n`;
  t += '──────────────────────────────────────────────────────────\n';
  t += 'FUND FLOW DETAILS\n';
  t += '──────────────────────────────────────────────────────────\n';
  (trailData.trail || []).forEach((x, i) => {
    t += `  ${i + 1}. ${x.from} → ${x.to} | ${inr(x.amount)} | ${x.time || '—'}\n`;
  });
  t += '\n══════════════════════════════════════════════════════════\n';
  t += 'DISCLAIMER: Paths show recorded transactions only.\n';
  t += `Generated: Mule Detection Platform | ${new Date().toLocaleDateString()}\n`;
  t += '══════════════════════════════════════════════════════════\n';
  return t;
}

function buildTrailReportHTML(trailData, acct) {
  const s = trailData.summary || {};
  return `<!DOCTYPE html><html><head><meta charset="utf-8"><title>Money Trail Report - ${esc(acct)}</title>
<style>body{font:13px/1.6 Arial,sans-serif;max-width:900px;margin:20px auto;padding:0 20px;color:#172033}
h1{font-size:22px;margin:0;border-bottom:4px solid #14233B;padding-bottom:10px}
h2{font-size:16px;color:#14233B;border-bottom:2px solid #E2E8F0;padding-bottom:4px;margin-top:24px}
table{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0}th,td{border:1px solid #E2E8F0;padding:8px;text-align:left}th{background:#F4F7FB}
.stat{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:16px 0}
.sc{background:#F4F7FB;border:1px solid #E2E8F0;border-radius:8px;padding:14px;text-align:center}
.sc .v{font-size:28px;font-weight:700}.sc .l{font-size:11px;color:#64748B}
.ft{border-top:2px solid #E2E8F0;padding-top:12px;margin-top:24px;font-size:11px;color:#64748B;text-align:center}
@media print{button{display:none}}</style></head><body>
<button onclick="window.print()" style="background:#2563EB;color:#fff;border:0;border-radius:6px;padding:10px 20px;cursor:pointer;margin:10px 5px">🖨 Print / PDF</button>
<h1>MONEY TRAIL INVESTIGATION REPORT</h1>
<p style="color:#64748B">Mule Detection Platform · <b>CONFIDENTIAL</b></p>
<p><b>Account:</b> ${esc(acct)} · <b>Generated:</b> ${new Date().toLocaleString()}</p>
<div class="stat">
  <div class="sc"><div class="v">${s.total_transactions || 0}</div><div class="l">Transactions</div></div>
  <div class="sc"><div class="v">${inr(s.total_outflow)}</div><div class="l">Total Outflow</div></div>
  <div class="sc"><div class="v">${s.unique_destinations || 0}</div><div class="l">Destinations</div></div>
</div>
<h2>Fund Flow Details</h2>
<table><tr><th>#</th><th>From</th><th>To</th><th>Amount</th><th>Time</th></tr>
${(trailData.trail || []).map((t, i) => `<tr><td>${i + 1}</td><td>${esc(t.from)}</td><td>${esc(t.to)}</td><td>${inr(t.amount)}</td><td>${esc(t.time || '—')}</td></tr>`).join('')}</table>
<div class="ft"><p>Paths show recorded transactions only. They do not prove specific funds moved onward.</p><p>Mule Detection Platform · ${new Date().toLocaleDateString()}</p></div>
</body></html>`;
}