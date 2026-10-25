'use strict';
/* ============================================================
   VIEWS: Every page returns HTML from the state S and D
   ============================================================ */
const V = {};
const DISC = '<p class="disc">Risk scores are investigation-priority indicators. They do not establish fraud or guilt; every lead needs human review.</p>';
const hd = (t, s, r = '') => `<div class="ph"><div><h1>${t}</h1><p class="sub">${s}</p></div><div class="bar">${r}</div></div>`;
const A_ = id => `<a href="#/account/${esc(id)}">${esc(id)}</a>`;
const tabs = list => `<div class="tabs" role="tablist">${list.map(t => `<button role="tab" class="${st.tab === t ? 'on' : ''}" data-act="tab" data-id="${t}">${t}</button>`).join('')}</div>`;
const opts = (list, cur, all) => `<option value="">${all}</option>` + list.map(o => `<option ${cur === o ? 'selected' : ''}>${esc(o)}</option>`).join('');

/* ===== OVERVIEW ===== */
V.overview = () => {
  const s = S.stats || {};
  const K = [
    ['Accounts monitored', fmt(s.total_accounts), '#/accounts', 'All analysed'],
    ['High-risk accounts', fmt(s.high_risk), '#/accounts?band=HIGH', 'Priority review'],
    ['Medium-risk', fmt(s.medium_risk), '#/accounts?band=MEDIUM', 'Monitor'],
    ['Low-risk', fmt(s.low_risk), '#/accounts?band=LOW', 'Routine'],
    ['Complaints', fmt(s.total_complaints), '#/complaints', fmt(s.unique_reported) + ' hotspots']
  ];
  const leads = (S.accounts || []).slice(0, 10);
  return `${hd('Financial Intelligence Overview', 'Risk-ranked leads, suspicious patterns and investigation activity.',
    '<button class="btn g" data-act="refresh">Refresh data</button>')}
<div class="grid">${K.map(k => `<div class="card kpi" tabindex="0" data-act="go" data-id="${k[2]}"><span>${k[0]}</span><big>${k[1]}</big><span>${k[3]}</span></div>`).join('')}</div>
<div class="two"><div class="card"><h2>Account priority distribution</h2>${donut([
    ['HIGH', s.high_risk || 0, '#DC2626'], ['MEDIUM', s.medium_risk || 0, '#D97706'], ['LOW', s.low_risk || 0, '#15803D']
  ])}<p class="sub">Investigation priority only.</p></div>
<div class="card"><h2>Quick actions</h2>
<p><a href="#/accounts">→ Open Risk Queue</a></p>
<p><a href="#/trails">→ Trace a Money Trail</a></p>
<p><a href="#/networks">→ View Networks</a></p>
<p><a href="#/complaints">→ Complaint Correlation</a></p>
<p><a href="#/reports">→ Generate Reports</a></p></div></div>
<div class="card"><h2>Prioritised investigative leads</h2>
${tbl(['#', 'Account', 'Score', 'Band', 'Behavioural', 'Graph', 'Actions'],
  leads.map((a, i) => [i + 1, A_(a.acct), sbar(a.risk_score), pb(a.band), pct(a.p_behavioural), pct(a.graph_risk),
    `<a href="#/account/${esc(a.acct)}">Profile</a> · <a href="#/reports?acct=${esc(a.acct)}">Report</a>`]))}
</div>${DISC}`;
};

/* ===== ACCOUNTS (Risk Queue) ===== */
V.accounts = () => {
  const f = st.acc;
  let rows = (S.accounts || []).slice();
  if (f.q) rows = rows.filter(a => String(a.acct).toLowerCase().includes(f.q.toLowerCase()));
  if (f.band) rows = rows.filter(a => a.band === f.band);
  return `${hd('Risk Queue', 'Rank suspicious accounts by AI risk score.',
    '<button class="btn g" data-act="exportacc">Export CSV</button>')}
<div class="bar">
  <input id="f_acc_q" data-f="acc.q" placeholder="Search account ID" value="${esc(f.q)}">
  <select id="f_acc_b" data-f="acc.band">${opts(['HIGH', 'MEDIUM', 'LOW'], f.band, 'All bands')}</select>
  <button class="btn g" data-act="clr" data-id="acc">Clear</button>
</div>
<div class="card">
${tbl(['Rank', 'Account', 'Score', 'Band', 'Behavioural', 'Graph', 'Actions'],
  pageOf('acc', rows).map(a => [
    '#' + (a.rank || '—'), A_(a.acct), sbar(a.risk_score), pb(a.band),
    pct(a.p_behavioural), pct(a.graph_risk),
    `<a href="#/account/${esc(a.acct)}">Profile</a> · <a href="#/trails?from=${esc(a.acct)}">Trace</a> · <a href="#/reports?acct=${esc(a.acct)}">Report</a>`
  ]))}
${pager('acc', rows.length)}
</div>${DISC}`;
};

/* ===== ACCOUNT PROFILE ===== */
V.account = id => {
  const d = D.acct;
  if (!d || d.error) return `<h1>Loading...</h1><p class="sub">${esc(d?.error || 'Fetching data for ' + id)}</p><a href="#/accounts">← Back to Risk Queue</a>`;
  const acc = d.account, r = d.reasons || [], feats = d.features || {}, cmps = d.complaints || [];
  const T = st.tab;
  let body = '';

  if (T === 'Overview') {
    body = `<div class="two"><div class="card"><h2>Risk Explanation</h2>
      <p>${pb(acc.band)} Score <b>${(acc.risk_score || 0).toFixed(1)}</b> / 100 · Rank #${acc.rank || '—'}</p>
      <div class="sg"><span>Behavioural risk</span><span class="sb"><i style="width:${(acc.p_behavioural || 0) * 100}%"></i></span><b>${pct(acc.p_behavioural)}</b></div>
      <div class="sg"><span>Graph anomaly risk</span><span class="sb"><i style="width:${(acc.graph_risk || 0) * 100}%"></i></span><b>${pct(acc.graph_risk)}</b></div>
      <div class="sg"><span>Identity risk</span><span class="sb"><i style="width:${(acc.p_identity || 0) * 100}%"></i></span><b>${pct(acc.p_identity)}</b></div>
      <p class="sub">R = 0.50×Behavioural + 0.30×Graph + 0.20×Identity</p></div>
      <div class="card"><h2>SHAP Reason Codes</h2>
      ${r.length ? `<ul class="ind">${r.map(i => `<li><b>${esc(i.description || i.feature)}</b> ${pb(i.impact > 0 ? 'HIGH' : 'LOW')}<br><span class="sub">Impact: ${i.impact > 0 ? '+' : ''}${(i.impact || 0).toFixed(3)} · ${esc(i.feature)}</span></li>`).join('')}</ul>` : '<p class="sub">No SHAP explanations available.</p>'}
      </div></div>`;
  }

  if (T === 'Features') {
    body = `<div class="card"><h2>Behavioural Features</h2>${tbl(['Feature', 'Value'],
      Object.entries(feats).map(([k, v]) => [esc(k.replace(/_/g, ' ')), typeof v === 'number' ? v.toFixed(4) : esc(v)]))}</div>`;
  }

  if (T === 'Complaints') {
    body = `<div class="card"><h2>Linked Complaints (${cmps.length})</h2>
      ${tbl(['ID', 'Category', 'Amount Lost', 'Date', 'Report'],
        cmps.map(c => [esc(c.complaint_id), esc(c.category), inr(c.amount_lost), esc(c.complaint_date),
          `<a data-act="gen_cmp_report" data-id="${esc(c.complaint_id)}|${esc(id)}">📋 Report</a>`]))}</div>`;
  }

  return `<h1>${esc(acc.acct || id)} ${pb(acc.band)}</h1>
<p class="sub">Risk Score: <b>${(acc.risk_score || 0).toFixed(1)}</b>/100 · Rank #${acc.rank || '—'} · Behavioural ${pct(acc.p_behavioural)} · Graph ${pct(acc.graph_risk)}</p>
<div class="bar">
  <button class="btn" data-act="go" data-id="#/trails?from=${esc(acc.acct || id)}">Trace Money Trail</button>
  <button class="btn g" data-act="go" data-id="#/networks?sel=${esc(acc.acct || id)}">View Network</button>
  <button class="btn g" data-act="go" data-id="#/reports?acct=${esc(acc.acct || id)}">📋 Full Report</button>
</div>
${tabs(['Overview', 'Features', 'Complaints'])}
${body}${DISC}`;
};

/* ===== MONEY TRAILS ===== */
V.trails = () => {
  const m = st.mt, d = D.trail;
  let trailHtml = '';
  if (d && d.trail && d.trail.length) {
    trailHtml = `<div class="stat">
      <div><b>${fmt(d.summary.total_transactions)}</b><span>Transactions</span></div>
      <div><b>${inr(d.summary.total_outflow)}</b><span>Total Outflow</span></div>
      <div><b>${fmt(d.summary.unique_destinations)}</b><span>Destinations</span></div>
    </div>
    <div class="card"><h2>Fund Flow Trail</h2>
      <div class="bar">
        <button class="btn g" data-act="dl_trail_txt">📥 Download TXT</button>
        <button class="btn g" data-act="dl_trail_html">📥 Download HTML</button>
        <button class="btn g" data-act="print_trail">🖨 Print / PDF</button>
      </div>
      ${tbl(['#', 'From', 'To', 'Amount', 'Time'],
        d.trail.slice(0, 50).map((t, i) => [i + 1, esc(t.from), esc(t.to), inr(t.amount), esc(t.time || '—')]))}
      ${d.trail.length > 50 ? `<p class="sub">Showing 50 of ${fmt(d.trail.length)}.</p>` : ''}
    </div>`;
  } else if (d) {
    trailHtml = '<p class="sub">No fund trail found.</p>';
  }

  return `${hd('Money Trail Explorer', 'Trace fund movements across accounts.')}
<div class="card"><div class="bar">
  <input id="mf" value="${esc(m.acct)}" placeholder="Account ID">
  <select id="mh">${[1, 2, 3, 4, 5].map(n => `<option ${m.depth == n ? 'selected' : ''}>${n}</option>`).join('')}</select>
  <button class="btn" data-act="trace">Trace Funds</button>
  <button class="btn g" data-act="mtreset">Reset</button>
</div></div>
${trailHtml}${DISC}`;
};

/* ===== NETWORK VIEW ===== */
V.networks = () => {
  const d = D.net;
  let netHtml = '<p class="sub">Enter an account ID and click "Load Network".</p>';
  if (d && d.nodes && d.nodes.length) netHtml = graphSvg(d.nodes, d.edges || []);
  return `${hd('Suspicious Network Visualization', 'Connected accounts and fund-flow relationships.')}
<div class="card"><div class="bar">
  <input id="nf" value="${esc(st.net.acct)}" placeholder="Account ID">
  <select id="nd">${[1, 2, 3].map(n => `<option ${st.net.depth == n ? 'selected' : ''}>${n}</option>`).join('')}</select>
  <button class="btn" data-act="loadnet">Load Network</button>
</div>
${d?.nodes ? `<p class="sub">${d.nodes.length} nodes · ${(d.edges || []).length} edges · ${d.nodes.filter(n => n.is_suspicious).length} suspicious</p>` : ''}
${netHtml}</div>
<div class="card" id="det"><p class="sub">Click a node for details.</p></div>${DISC}`;
};

/* ===== COMPLAINTS ===== */
V.complaints = () => {
  const f = st.cmp;
  let rows = (S.hotspots || []).slice();
  if (f.q) rows = rows.filter(x => String(x.reported_acct).toLowerCase().includes(f.q.toLowerCase()));
  return `${hd('Complaint Correlation', 'Hotspot accounts and linked complaints.')}
<div class="bar">
  <input id="f_cmp_q" data-f="cmp.q" placeholder="Search account" value="${esc(f.q)}">
  <button class="btn g" data-act="clr" data-id="cmp">Clear</button>
</div>
<div class="card"><h2>Most-complained accounts (Hotspots)</h2>
${tbl(['Account', 'Complaints', 'Total Loss', 'Actions'],
  pageOf('cmp', rows).map(x => [A_(x.reported_acct), fmt(x.n_complaints), inr(x.total_loss),
    `<a href="#/account/${esc(x.reported_acct)}">Profile</a> · <a href="#/trails?from=${esc(x.reported_acct)}">Trace</a> · <a data-act="gen_cmp_report" data-id="ALL|${esc(x.reported_acct)}">📋 Report</a>`]))}
${pager('cmp', rows.length)}
</div>
<div class="card"><h2>Summary</h2>
<div class="stat">
  <div><b>${fmt((S.hotspots || []).length)}</b><span>Hotspot accounts</span></div>
  <div><b>${fmt((S.hotspots || []).reduce((a, x) => a + (x.n_complaints || 0), 0))}</b><span>Total complaints</span></div>
  <div><b>${inr((S.hotspots || []).reduce((a, x) => a + (x.total_loss || 0), 0))}</b><span>Total loss</span></div>
</div></div>${DISC}`;
};

/* ===== REPORTS ===== */
V.reports = () => {
  const acct = st.repAcct || '';
  const d = D.report;

  if (d && d.report_text) {
    const rs = d.risk_score || 0;
    const bc = rs >= 70 ? '#DC2626' : rs >= 40 ? '#D97706' : '#15803D';
    return `${hd('Investigation Report', `SAR-style report for account ${esc(acct)}`,
      `<button class="btn g" data-act="genreport">🔄 Regenerate</button>
       <button class="btn" data-act="dl_txt">📥 TXT</button>
       <button class="btn" data-act="dl_html">📥 HTML</button>
       <button class="btn g" data-act="print_report">🖨 Print / PDF</button>`)}

<div class="bar">
  <input id="rf" value="${esc(acct)}" placeholder="Enter account ID...">
  <button class="btn" data-act="genreport">Generate</button>
  <button class="btn g" data-act="go" data-id="#/account/${esc(acct)}">View Profile</button>
</div>

<div class="rep-card">
  <div style="border-bottom:4px solid #14233B;padding-bottom:12px;margin-bottom:16px">
    <h1 style="margin:0;font-size:22px">FINANCIAL FRAUD INVESTIGATION REPORT</h1>
    <p style="color:#64748B;margin:4px 0 0;font-size:13px">Mule Detection Platform · CONFIDENTIAL</p>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px;font-size:13px">
    <div><b>Report ID:</b> RPT-${esc(acct)}-${Date.now().toString().slice(-6)}</div>
    <div><b>Generated:</b> ${new Date().toLocaleString()}</div>
    <div><b>Account:</b> ${esc(acct)}</div>
    <div><b>Classification:</b> ${esc(d.band)}</div>
  </div>

  <div class="rep-metrics">
    <div class="rep-metric"><div class="val" style="color:${bc}">${rs.toFixed(1)}</div><div class="lbl">Risk Score</div></div>
    <div class="rep-metric"><div class="val">${esc(d.band)}</div><div class="lbl">Risk Band</div></div>
    <div class="rep-metric"><div class="val">${d.n_complaints || 0}</div><div class="lbl">Complaints</div></div>
    <div class="rep-metric"><div class="val">#${d.rank || '—'}</div><div class="lbl">Rank</div></div>
  </div>

  <div style="background:#F8FAFC;border-radius:8px;padding:20px;border:1px solid #E2E8F0;margin:16px 0">
    <h3 style="margin:0 0 12px;font-size:15px;color:#14233B">📋 Detailed Findings</h3>
    <pre style="white-space:pre-wrap;font-family:'Courier New',monospace;font-size:13px;line-height:1.7;margin:0">${esc(d.report_text)}</pre>
  </div>

  ${d.reasons?.length ? `<div style="margin:16px 0"><h3 style="font-size:15px;color:#14233B">🔍 SHAP Risk Indicators</h3>
    ${d.reasons.map(r => `<div class="rep-ind ${r.impact > 0 ? 'pos' : 'neg'}">
      <span style="font-size:18px">${r.impact > 0 ? '⚠️' : '✅'}</span>
      <div style="flex:1"><div style="font-weight:600;font-size:13px">${esc(r.description || r.feature)}</div>
      <div style="font-size:11px;color:#64748B">${esc(r.feature)}</div></div>
      <div style="font-family:monospace;font-weight:700;color:${r.impact > 0 ? '#DC2626' : '#15803D'}">${r.impact > 0 ? '+' : ''}${(r.impact || 0).toFixed(3)}</div>
    </div>`).join('')}</div>` : ''}

  ${d.complaints?.length ? `<div style="margin:16px 0"><h3 style="font-size:15px;color:#14233B">📢 Complaints (${d.complaints.length})</h3>
    <table class="rep-table"><thead><tr><th>ID</th><th>Category</th><th>Loss</th><th>Date</th></tr></thead>
    <tbody>${d.complaints.map(c => `<tr><td>${esc(c.complaint_id)}</td><td>${esc(c.category)}</td><td>${inr(c.amount_lost)}</td><td>${esc(c.complaint_date || '—')}</td></tr>`).join('')}</tbody></table></div>` : ''}

  ${d.trail?.length ? `<div style="margin:16px 0"><h3 style="font-size:15px;color:#14233B">💸 Fund Trail (${d.trail.length} transactions)</h3>
    <div class="rep-metrics" style="grid-template-columns:repeat(3,1fr)">
      <div class="rep-metric"><div class="val" style="font-size:20px">${fmt(d.trail_summary?.total_transactions || 0)}</div><div class="lbl">Transactions</div></div>
      <div class="rep-metric"><div class="val" style="font-size:20px">${inr(d.trail_summary?.total_outflow || 0)}</div><div class="lbl">Outflow</div></div>
      <div class="rep-metric"><div class="val" style="font-size:20px">${fmt(d.trail_summary?.unique_destinations || 0)}</div><div class="lbl">Destinations</div></div>
    </div>
    <table class="rep-table"><thead><tr><th>From</th><th>To</th><th>Amount</th><th>Time</th></tr></thead>
    <tbody>${d.trail.slice(0, 15).map(t => `<tr><td>${esc(t.from)}</td><td>${esc(t.to)}</td><td>${inr(t.amount)}</td><td>${esc(t.time || '—')}</td></tr>`).join('')}</tbody></table>
    ${d.trail.length > 15 ? `<p style="font-size:11px;color:#64748B">Showing 15 of ${d.trail.length}</p>` : ''}</div>` : ''}

  <div class="rep-rec" style="border-color:${bc};background:${rs >= 70 ? '#FEF2F2' : rs >= 40 ? '#FFFBEB' : '#F0FDF4'}">
    <h3 style="margin:0 0 8px;font-size:16px">📌 Recommendation</h3>
    <p style="margin:0;font-size:15px;font-weight:600">
      ${rs >= 70 ? '🔴 ESCALATE — High Priority Investigation Required'
        : rs >= 40 ? '🟡 REVIEW — Monitor and Gather Additional Data'
        : '🟢 LOW PRIORITY — Routine Monitoring Sufficient'}
    </p>
  </div>

  <div class="rep-footer">
    <p><b>DISCLAIMER:</b> AI-generated. Requires human review. Does not establish fraud or guilt.</p>
    <p>Generated by: Mule Detection Platform · ${new Date().toLocaleDateString()}</p>
  </div>
</div>

<div class="bar" style="justify-content:center;margin-top:20px">
  <button class="btn" data-act="dl_txt">📥 Download TXT</button>
  <button class="btn" data-act="dl_html">📥 Download HTML (Print-ready)</button>
  <button class="btn g" data-act="print_report">🖨 Print / Save as PDF</button>
  <button class="btn g" data-act="genreport">🔄 Regenerate</button>
</div>
${DISC}`;
  }

  // No report generated
  return `${hd('Investigation Reports', 'Generate SAR-style reports for accounts and complaints.')}
<div class="two">
<div class="card" style="max-width:550px">
  <h2>📋 Account Report</h2>
  <p class="sub">Generate full risk report with SHAP, complaints and fund trails.</p>
  <div class="bar">
    <input id="rf" value="${esc(acct)}" placeholder="Account ID" style="flex:1">
    <button class="btn" data-act="genreport">Generate</button>
  </div>
  <p class="sub"><b>Quick generate:</b></p>
  ${(S.accounts || []).slice(0, 5).map(a => `<p><a data-act="quickreport" data-id="${esc(a.acct)}">📋 ${esc(a.acct)} (Score: ${(a.risk_score || 0).toFixed(1)})</a></p>`).join('')}
</div>
<div class="card" style="max-width:550px">
  <h2>📢 Complaint Report</h2>
  <p class="sub">Generate complaint investigation report with account risk profile.</p>
  <div class="bar">
    <input id="rc" value="${esc(st.repCmp || '')}" placeholder="Complaint ID (e.g. C00001)" style="flex:1">
    <button class="btn" data-act="gen_cmp_report_direct">Generate</button>
  </div>
  <p class="sub"><b>Or pick a hotspot account:</b></p>
  ${(S.hotspots || []).slice(0, 5).map(x => `<p><a data-act="gen_cmp_report" data-id="ALL|${esc(x.reported_acct)}">📋 ${esc(x.reported_acct)} (${x.n_complaints} complaints)</a></p>`).join('')}
</div>
</div>
<div class="card">
  <h2>Report Contents</h2>
  <ul style="font-size:13px;line-height:2;padding-left:20px">
    <li>Account risk score, band and global rank</li>
    <li>SHAP-based explainable risk indicators</li>
    <li>Linked cybercrime complaints with amounts</li>
    <li>Fund flow trail with transactions</li>
    <li>Recommendation (Escalate / Review / Monitor)</li>
    <li>Download as TXT, HTML (print-ready) or PDF</li>
  </ul>
</div>${DISC}`;
};