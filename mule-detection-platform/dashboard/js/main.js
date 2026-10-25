'use strict';
/* ============================================================
   MAIN: Router, event handling, page shell
   ============================================================ */
const NAV = [
  ['overview', 'Overview'],
  ['accounts', 'Risk Queue'],
  ['trails', 'Money Trails'],
  ['networks', 'Network View'],
  ['complaints', 'Complaints'],
  ['reports', 'Reports']
];

const SHIELD = '<svg width="34" height="40" viewBox="0 0 100 120"><path d="M50 4 92 20v38c0 28-18 49-42 58C26 107 8 86 8 58V20z" fill="#0f1f3a" stroke="#3b82f6" stroke-width="5"/><path d="m50 34 7 15 16 2-12 11 3 16-14-8-14 8 3-16-12-11 16-2z" fill="#f5c542"/></svg>';

async function route() {
  const hash = location.hash.slice(2) || 'overview';
  const [p, qs] = hash.split('?');
  const [seg, arg] = (p || 'overview').split('/');
  const q = new URLSearchParams(qs || '');
  const id = arg ? decodeURIComponent(arg) : '';

  closeModal();

  // Pre-fetch data
  if ((seg === 'overview' || seg === 'accounts') && !S.stats.total_accounts) {
    S.stats = await apiGet('/api/dashboard/stats') || {};
    const rd = await apiGet('/api/risk-queue?limit=100');
    S.accounts = rd?.accounts || [];
  }

  if (seg === 'account' && id) {
    D.acct = await apiGet('/api/account/' + encodeURIComponent(id));
  }

  if (seg === 'trails') {
    const from = q.get('from');
    if (from) {
      st.mt.acct = from;
      if (!D.trail || st.mt.lastAcct !== from) {
        st.mt.lastAcct = from;
        D.trail = await apiGet(`/api/account/${encodeURIComponent(from)}/trail?depth=${st.mt.depth}`);
      }
    }
  }

  if (seg === 'networks') {
    const sel = q.get('sel');
    if (sel) {
      st.net.acct = sel;
      if (!D.net || st.net.lastAcct !== sel) {
        st.net.lastAcct = sel;
        D.net = await apiGet(`/api/account/${encodeURIComponent(sel)}/network?depth=${st.net.depth}`);
      }
    }
  }

  if (seg === 'complaints' && !S.hotspots.length) {
    const cd = await apiGet('/api/complaints/hotspots');
    S.hotspots = cd?.hotspots || [];
  }

  if (seg === 'reports') {
    const ra = q.get('acct');
    if (ra && st.repAcct !== ra) {
      st.repAcct = ra;
      D.report = null; // reset so it generates fresh
    }
  }

  // Build page
  let h;
  if (seg === 'account' && id) h = V.account(id);
  else if (NAV.some(n => n[0] === seg)) h = V[seg]();
  else h = '<h1>Page not found</h1><p class="sub"><a href="#/overview">Go to overview</a></p>';

  $('#app').innerHTML = `<nav aria-label="Primary">
    <div class="brand">${SHIELD}<div><b>Mule<span>Detect</span></b><small>FINANCIAL FRAUD INVESTIGATION</small></div></div>
    ${NAV.map(n => `<a href="#/${n[0]}" class="${seg === n[0] ? 'on' : ''}">${n[1]}</a>`).join('')}
    <small>v1.0 · AI/ML + Graph Analytics<br>Powered by XGBoost + SHAP</small>
  </nav>
  <main>
    <div class="top">
      <span class="unit">MULE DETECTION PLATFORM · Cyber Crime &amp; Financial Fraud</span>
      <input id="gs" placeholder="Go to account ID..." aria-label="Global search">
      <span class="sub" style="margin:0">Analyst</span>
    </div>
    ${h}
  </main>`;

  initGraphs();
}

function rerender() { route(); }

/* ============================================================
   EVENT HANDLERS
   ============================================================ */
const H = {
  go: (e, id) => { location.hash = id; },
  refresh: async () => {
    S.stats = await apiGet('/api/dashboard/stats') || {};
    const rd = await apiGet('/api/risk-queue?limit=100');
    S.accounts = rd?.accounts || [];
    rerender(); toast('Data refreshed');
  },
  close: closeModal,
  tab: (e, id) => { st.tab = id; rerender(); },
  clr: (e, g) => {
    if (g === 'acc') st.acc = { q: '', band: '', page: 1 };
    if (g === 'cmp') st.cmp = { q: '', page: 1 };
    rerender();
  },
  pg: (e, id) => { const [g, n] = id.split('|'); st[g].page = +n; rerender(); },
  exportacc: () => {
    const rows = (S.accounts || []).map(a => [a.rank, a.acct, a.risk_score, a.band, a.p_behavioural, a.graph_risk].join(','));
    download('risk_queue.csv', 'Rank,Account,Risk Score,Band,Behavioural,Graph\n' + rows.join('\n'), 'text/csv');
    toast('CSV exported');
  },

  // ===== MONEY TRAIL =====
  trace: async () => {
    const acct = $('#mf')?.value.trim();
    const depth = $('#mh')?.value || 3;
    if (!acct) { toast('Enter an account ID'); return; }
    st.mt.acct = acct; st.mt.depth = depth; st.mt.lastAcct = acct;
    toast('Tracing funds...');
    D.trail = await apiGet(`/api/account/${encodeURIComponent(acct)}/trail?depth=${depth}`);
    rerender();
  },
  mtreset: () => { st.mt = { acct: '', depth: 3, lastAcct: '' }; D.trail = null; rerender(); },

  // ===== NETWORK =====
  loadnet: async () => {
    const acct = $('#nf')?.value.trim();
    const depth = $('#nd')?.value || 2;
    if (!acct) { toast('Enter an account ID'); return; }
    st.net.acct = acct; st.net.depth = depth; st.net.lastAcct = acct;
    toast('Loading network...');
    D.net = await apiGet(`/api/account/${encodeURIComponent(acct)}/network?depth=${depth}`);
    rerender();
  },
  gnode: (el, id) => {
    $$('.node.sel').forEach(n => n.classList.remove('sel'));
    el.classList.add('sel');
    const det = $('#det');
    if (det && D.net) {
      const n = D.net.nodes?.find(x => x.id === id);
      if (n) {
        det.innerHTML = `<h2>${esc(n.id)}</h2>
          <p>${pb(n.is_suspicious ? 'HIGH' : n.is_selected ? 'MEDIUM' : 'LOW')} · ${n.is_suspicious ? 'Complaint-linked' : n.is_selected ? 'Selected' : 'Connected'}</p>
          <p><a href="#/account/${esc(n.id)}">Profile</a> · <a href="#/trails?from=${esc(n.id)}">Trace</a> · <a href="#/reports?acct=${esc(n.id)}">Report</a></p>`;
      }
    }
  },

  // ===== REPORTS =====
  genreport: async () => {
    const acct = $('#rf')?.value.trim();
    if (!acct) { toast('Enter an account ID'); return; }
    st.repAcct = acct;
    toast('Generating report...');
    const repData = await apiGet('/api/report/' + encodeURIComponent(acct));
    const accData = await apiGet('/api/account/' + encodeURIComponent(acct));
    const trailData = await apiGet(`/api/account/${encodeURIComponent(acct)}/trail?depth=3`);
    if (!repData) { toast('Report generation failed'); return; }
    D.report = {
      report_text: repData.report_text || '',
      risk_score: repData.risk_score || accData?.account?.risk_score || 0,
      band: repData.band || accData?.account?.band || '—',
      rank: accData?.account?.rank || repData.rank || '—',
      n_complaints: repData.n_complaints || 0,
      p_behavioural: accData?.account?.p_behavioural || 0,
      graph_risk: accData?.account?.graph_risk || 0,
      p_identity: accData?.account?.p_identity || 0,
      reasons: accData?.reasons || [],
      complaints: accData?.complaints || [],
      trail: trailData?.trail || [],
      trail_summary: trailData?.summary || {},
    };
    rerender();
    toast('Report generated!');
  },

  quickreport: async (el, id) => {
    const acct = id;
    st.repAcct = acct;
    toast(`Generating report for ${acct}...`);
    const repData = await apiGet('/api/report/' + encodeURIComponent(acct));
    const accData = await apiGet('/api/account/' + encodeURIComponent(acct));
    const trailData = await apiGet(`/api/account/${encodeURIComponent(acct)}/trail?depth=3`);
    if (!repData) { toast('Report generation failed'); return; }
    D.report = {
      report_text: repData.report_text || '',
      risk_score: repData.risk_score || accData?.account?.risk_score || 0,
      band: repData.band || accData?.account?.band || '—',
      rank: accData?.account?.rank || repData.rank || '—',
      n_complaints: repData.n_complaints || 0,
      p_behavioural: accData?.account?.p_behavioural || 0,
      graph_risk: accData?.account?.graph_risk || 0,
      p_identity: accData?.account?.p_identity || 0,
      reasons: accData?.reasons || [],
      complaints: accData?.complaints || [],
      trail: trailData?.trail || [],
      trail_summary: trailData?.summary || {},
    };
    rerender();
    toast('Report generated!');
  },

  dl_txt: () => {
    if (!D.report) { toast('Generate a report first'); return; }
    const txt = buildAccountReportTXT(D.report, st.repAcct);
    download(`SAR_Report_${st.repAcct}_${new Date().toISOString().slice(0, 10)}.txt`, txt);
    toast('TXT downloaded!');
  },
  dl_html: () => {
    if (!D.report) { toast('Generate a report first'); return; }
    const html = buildAccountReportHTML(D.report, st.repAcct);
    download(`SAR_Report_${st.repAcct}_${new Date().toISOString().slice(0, 10)}.html`, html, 'text/html');
    toast('HTML downloaded! Open in browser to print.');
  },
  print_report: () => {
    if (!D.report) { toast('Generate a report first'); return; }
    const html = buildAccountReportHTML(D.report, st.repAcct);
    const w = window.open('', '_blank');
    if (!w) { toast('Pop-up blocked. Allow pop-ups or use Download.'); return; }
    w.document.write(html); w.document.close();
    setTimeout(() => w.print(), 500);
    toast('Print dialog opened — Save as PDF!');
  },

  // ===== COMPLAINT REPORTS =====
  gen_cmp_report: async (el, id) => {
    const [cmpId, acctId] = id.split('|');
    toast('Generating complaint report...');
    const acctData = await apiGet('/api/account/' + encodeURIComponent(acctId));
    if (!acctData) { toast('Account not found'); return; }

    // Find complaint data
    let cmp;
    if (cmpId === 'ALL') {
      // Use first complaint for this account
      cmp = acctData.complaints?.[0] || {
        complaint_id: 'MULTIPLE', category: 'Multiple complaints',
        reported_acct: acctId, victim_acct: 'Multiple',
        amount_lost: (S.hotspots.find(x => x.reported_acct == acctId)?.total_loss) || 0,
        complaint_date: new Date().toISOString().slice(0, 10)
      };
    } else {
      cmp = acctData.complaints?.find(c => c.complaint_id === cmpId) || {
        complaint_id: cmpId, category: 'Unknown', reported_acct: acctId,
        victim_acct: '—', amount_lost: 0, complaint_date: '—'
      };
    }

    // Generate and show in modal
    const txt = buildComplaintReportTXT(cmp, acctData);
    const html = buildComplaintReportHTML(cmp, acctData);

    D.cmpReport = { txt, html, cmp };

    modal(`<h2>📋 Complaint Report — ${esc(cmp.complaint_id)}</h2>
      <div style="background:#F8FAFC;padding:16px;border-radius:8px;max-height:400px;overflow:auto">
        <pre style="white-space:pre-wrap;font-family:monospace;font-size:12px;margin:0">${esc(txt)}</pre>
      </div>
      <div class="bar" style="margin-top:12px">
        <button class="btn" data-act="dl_cmp_txt">📥 TXT</button>
        <button class="btn" data-act="dl_cmp_html">📥 HTML</button>
        <button class="btn g" data-act="print_cmp">🖨 Print</button>
        <button class="btn g" data-act="close">Close</button>
      </div>`);
    toast('Complaint report generated!');
  },

  gen_cmp_report_direct: async () => {
    const cmpId = $('#rc')?.value.trim();
    if (!cmpId) { toast('Enter a complaint ID'); return; }
    // Try to find in hotspots
    const hotspot = S.hotspots.find(x => String(x.reported_acct) === cmpId);
    const acctId = hotspot ? hotspot.reported_acct : cmpId;
    H.gen_cmp_report(null, `ALL|${acctId}`);
  },

  dl_cmp_txt: () => {
    if (!D.cmpReport) return;
    download(`Complaint_Report_${D.cmpReport.cmp.complaint_id}.txt`, D.cmpReport.txt);
    toast('Complaint TXT downloaded!');
  },
  dl_cmp_html: () => {
    if (!D.cmpReport) return;
    download(`Complaint_Report_${D.cmpReport.cmp.complaint_id}.html`, D.cmpReport.html, 'text/html');
    toast('Complaint HTML downloaded!');
  },
  print_cmp: () => {
    if (!D.cmpReport) return;
    const w = window.open('', '_blank');
    if (!w) { toast('Pop-up blocked'); return; }
    w.document.write(D.cmpReport.html); w.document.close();
    setTimeout(() => w.print(), 500);
    toast('Print dialog opened!');
  },

  // ===== TRAIL REPORTS =====
  dl_trail_txt: () => {
    if (!D.trail) { toast('Trace funds first'); return; }
    const txt = buildTrailReportTXT(D.trail, st.mt.acct);
    download(`Trail_Report_${st.mt.acct}.txt`, txt);
    toast('Trail TXT downloaded!');
  },
  dl_trail_html: () => {
    if (!D.trail) { toast('Trace funds first'); return; }
    const html = buildTrailReportHTML(D.trail, st.mt.acct);
    download(`Trail_Report_${st.mt.acct}.html`, html, 'text/html');
    toast('Trail HTML downloaded!');
  },
  print_trail: () => {
    if (!D.trail) { toast('Trace funds first'); return; }
    const html = buildTrailReportHTML(D.trail, st.mt.acct);
    const w = window.open('', '_blank');
    if (!w) { toast('Pop-up blocked'); return; }
    w.document.write(html); w.document.close();
    setTimeout(() => w.print(), 500);
    toast('Print dialog opened!');
  }
};

/* ============================================================
   EVENT DELEGATION
   ============================================================ */
document.addEventListener('click', e => {
  const el = e.target.closest('[data-act]');
  if (!el) return;
  const fn = H[el.dataset.act];
  if (fn) { e.preventDefault(); fn(el, el.dataset.id); }
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
  if (e.key === 'Enter' && e.target.id === 'gs') {
    const v = e.target.value.trim();
    if (v) location.hash = '#/account/' + encodeURIComponent(v);
  }
});

document.addEventListener('input', e => {
  const f = e.target.dataset?.f;
  if (f) {
    const [g, k] = f.split('.');
    st[g][k] = e.target.value;
    if ('page' in st[g]) st[g].page = 1;
    rerender();
  }
});

/* ============================================================
   INIT
   ============================================================ */
addEventListener('hashchange', () => { route(); scrollTo(0, 0); });

(async () => {
  if (!location.hash) location.hash = '#/overview';
  await route();
  console.log('[MuleDetect] Dashboard ready!');
})();