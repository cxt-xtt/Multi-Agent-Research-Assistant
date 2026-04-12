/**
 * ResearchOS — Multi-Agent Research Dashboard
 * Frontend application logic
 */

// ── State ──────────────────────────────────────────────────────────────────

const state = {
  currentReport: null,
  isRunning: false,
  searchDepth: 'advanced',
  history: JSON.parse(localStorage.getItem('research_history') || '[]'),
};

const API_BASE = window.location.origin;

// ── DOM References ─────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

// ── Init ───────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  setupQueryInput();
  setupDepthToggle();
  setupKeyboardShortcuts();
  checkAPIHealth();
  renderHistory();
  setInterval(checkAPIHealth, 30000);
});

// ── Health Check ───────────────────────────────────────────────────────────

async function checkAPIHealth() {
  try {
    const resp = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    const dot = $('statusDot');
    const label = $('statusLabel');
    if (resp.ok) {
      dot.className = 'status-dot ok';
      label.textContent = 'API online';
    } else {
      dot.className = 'status-dot err';
      label.textContent = 'API error';
    }
  } catch {
    $('statusDot').className = 'status-dot err';
    $('statusLabel').textContent = 'offline';
  }
}

// ── Input Setup ────────────────────────────────────────────────────────────

function setupQueryInput() {
  const input = $('queryInput');
  input.addEventListener('input', () => {
    $('charCount').textContent = input.value.length;
  });
}

function setupDepthToggle() {
  document.querySelectorAll('.depth-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.depth-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.searchDepth = btn.dataset.depth;
    });
  });
}

function setupKeyboardShortcuts() {
  $('queryInput').addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      runResearch();
    }
  });
}

// ── Pipeline ───────────────────────────────────────────────────────────────

async function runResearch() {
  const query = $('queryInput').value.trim();
  if (!query || query.length < 3) {
    showToast('Please enter a research question (min. 3 characters).');
    return;
  }
  if (state.isRunning) return;

  state.isRunning = true;
  setRunButton(true);
  hideResults();
  hideError();
  resetPipeline();
  activateStep('search');

  try {
    const resp = await fetch(`${API_BASE}/api/research/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        search_depth: state.searchDepth,
        use_cache: true,
        notify_n8n: true,
      }),
      signal: AbortSignal.timeout(150000),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
      throw new Error(err.detail || `Request failed: ${resp.status}`);
    }

    const report = await resp.json();
    state.currentReport = report;

    // Animate pipeline completion
    await animatePipelineCompletion(report);

    renderReport(report);
    saveToHistory(query, report);
    renderHistory();

  } catch (err) {
    markPipelineError();
    showError(err.message || 'Research pipeline failed. Please try again.');
    console.error('Research error:', err);
  } finally {
    state.isRunning = false;
    setRunButton(false);
  }
}

// ── Pipeline Animation ─────────────────────────────────────────────────────

function resetPipeline() {
  ['search', 'summarize', 'factcheck', 'report'].forEach(s => {
    const el = $(`step-${s}`);
    el.classList.remove('active', 'done', 'error');
  });
  ['line-1', 'line-2', 'line-3'].forEach(id => {
    const el = $(id);
    if (el) el.classList.remove('active', 'done');
  });
}

function activateStep(stepName, detail = null) {
  const stepId = { search: 'step-search', summarize: 'step-summarize', factcheck: 'step-factcheck', report: 'step-report' }[stepName];
  const el = $(stepId);
  if (!el) return;
  el.classList.add('active');
  el.classList.remove('done', 'error');

  if (detail) {
    const detailEl = $(`step-${stepName}-detail`);
    if (detailEl) detailEl.textContent = detail;
  }
}

function completeStep(stepName, detail = null) {
  const stepId = `step-${stepName}`;
  const el = $(stepId);
  if (!el) return;
  el.classList.remove('active');
  el.classList.add('done');
  if (detail) {
    const detailEl = $(`step-${stepName}-detail`);
    if (detailEl) detailEl.textContent = detail;
  }
}

function activateLine(lineId) {
  const el = $(lineId);
  if (el) el.classList.add('active');
}

function completeLine(lineId) {
  const el = $(lineId);
  if (el) { el.classList.remove('active'); el.classList.add('done'); }
}

async function animatePipelineCompletion(report) {
  const timings = report.node_timings_ms || {};

  // Search → done
  completeStep('search', `${Math.round((timings.search || 2000) / 1000)}s — ${report.sources?.length || 0} sources`);
  activateLine('line-1');
  await delay(300);
  completeLine('line-1');

  // Summarize
  activateStep('summarize', 'Synthesizing...');
  await delay(400);
  completeStep('summarize', `${Math.round((timings.summarize || 5000) / 1000)}s — GPT-4o`);
  activateLine('line-2');
  await delay(300);
  completeLine('line-2');

  // Fact-check
  activateStep('factcheck', 'Verifying...');
  await delay(400);
  completeStep('factcheck', `${Math.round((timings.fact_check || 4000) / 1000)}s — ${report.fact_check?.verdict || ''}`);
  activateLine('line-3');
  await delay(300);
  completeLine('line-3');

  // Report
  activateStep('report');
  await delay(200);
  completeStep('report', 'Complete ✓');
}

function markPipelineError() {
  ['search', 'summarize', 'factcheck', 'report'].forEach(s => {
    const el = $(`step-${s}`);
    if (el && el.classList.contains('active')) {
      el.classList.remove('active');
      el.classList.add('error');
    }
  });
}

// ── Render Report ──────────────────────────────────────────────────────────

function renderReport(report) {
  // Header metadata
  const verdict = report.fact_check?.verdict || 'UNCERTAIN';
  const confidence = Math.round((report.overall_confidence || 0) * 100);
  const latency = Math.round((report.total_latency_ms || 0) / 1000);
  const emoji = report.fact_check?.verdict_emoji || '❓';

  const verdictEl = $('reportVerdict');
  verdictEl.textContent = `${emoji} ${verdict.replace('_', ' ')}`;
  verdictEl.className = `report-verdict ${verdict}`;

  $('reportConfidence').textContent = `Confidence: ${confidence}%`;
  $('reportLatency').textContent = `${latency}s • ${report.sources?.length || 0} sources${report._from_cache ? ' • cached' : ''}`;

  // Render all tabs
  renderSummaryTab(report);
  renderSourcesTab(report);
  renderFactCheckTab(report);
  $('rawJson').textContent = JSON.stringify(report, null, 2);

  // Show results
  $('sourcesBadge').textContent = report.sources?.length || 0;
  showResults();
  switchTab('summary');
}

function renderSummaryTab(report) {
  // Direct answer
  if (report.direct_answer) {
    $('directAnswer').hidden = false;
    $('directAnswerText').textContent = report.direct_answer;
  } else {
    $('directAnswer').hidden = true;
  }

  // Summary text
  $('summaryText').textContent = report.summary?.summary || 'No summary available.';

  // Key points
  const points = report.summary?.key_points || [];
  const kpEl = $('keyPointsList');
  if (points.length > 0) {
    kpEl.innerHTML = '<h4>Key Findings</h4>';
    points.forEach((pt, i) => {
      const item = document.createElement('div');
      item.className = 'key-point-item';
      item.style.animationDelay = `${i * 60}ms`;
      item.textContent = pt;
      kpEl.appendChild(item);
    });
  } else {
    kpEl.innerHTML = '';
  }

  // Entities
  const entities = report.summary?.key_entities || {};
  const egEl = $('entitiesGrid');
  egEl.innerHTML = '';

  const groups = [
    { key: 'people', label: 'People' },
    { key: 'organizations', label: 'Organizations' },
    { key: 'dates', label: 'Dates' },
    { key: 'statistics', label: 'Statistics' },
  ];

  groups.forEach(({ key, label }) => {
    const items = entities[key];
    if (items && items.length > 0) {
      const group = document.createElement('div');
      group.className = 'entity-group';
      group.innerHTML = `<span class="entity-group-label">${label}</span>`;
      items.forEach(item => {
        const tag = document.createElement('span');
        tag.className = 'entity-tag';
        tag.textContent = item;
        group.appendChild(tag);
      });
      egEl.appendChild(group);
    }
  });
}

function renderSourcesTab(report) {
  const list = $('sourcesList');
  list.innerHTML = '';

  (report.sources || []).forEach((src, i) => {
    const score = Math.round((src.score || 0) * 100);
    const item = document.createElement('div');
    item.className = 'source-item';
    item.style.animationDelay = `${i * 70}ms`;
    item.innerHTML = `
      <div class="source-header">
        <a href="${escapeHtml(src.url)}" target="_blank" rel="noopener" class="source-title">
          ${escapeHtml(src.title || 'Untitled')}
        </a>
        <span class="source-score">${score}%</span>
      </div>
      <div class="source-url">${escapeHtml(src.url)}</div>
      <div class="source-content">${escapeHtml(src.content || '')}</div>
      ${src.published_date ? `<div class="source-date">${escapeHtml(src.published_date)}</div>` : ''}
    `;
    list.appendChild(item);
  });

  if (!report.sources?.length) {
    list.innerHTML = '<p style="color:var(--text-muted);font-size:0.85rem;">No sources available.</p>';
  }
}

function renderFactCheckTab(report) {
  const fc = report.fact_check;
  if (!fc) {
    $('factCheckHeader').innerHTML = '<p style="color:var(--text-muted);font-size:0.85rem;">Fact-check data unavailable.</p>';
    return;
  }

  const confidence = Math.round((fc.overall_confidence || 0) * 100);

  $('factCheckHeader').innerHTML = `
    <div class="fc-verdict-row">
      <span class="fc-verdict-badge">${fc.verdict_emoji || ''} ${(fc.verdict || '').replace('_', ' ')}</span>
      <div class="fc-confidence-bar">
        <div class="fc-confidence-fill" style="width: 0%" id="fcConfidenceFill"></div>
      </div>
      <span class="fc-confidence-value">${confidence}%</span>
    </div>
    <p class="fc-summary">${escapeHtml(fc.fact_check_summary || '')}</p>
  `;

  // Animate bar after DOM update
  requestAnimationFrame(() => {
    setTimeout(() => {
      const fill = $('fcConfidenceFill');
      if (fill) fill.style.width = `${confidence}%`;
    }, 100);
  });

  // Claims
  const claimsList = $('claimsList');
  claimsList.innerHTML = '';
  (fc.claims || []).forEach((claim, i) => {
    const claimScore = Math.round((claim.confidence || 0) * 100);
    const item = document.createElement('div');
    item.className = `claim-item ${claim.status || 'PARTIAL'}`;
    item.style.animationDelay = `${i * 60}ms`;
    item.innerHTML = `
      <div class="claim-header">
        <span class="claim-status">${claim.status || 'PARTIAL'}</span>
        <span class="claim-confidence">${claimScore}%</span>
      </div>
      <div class="claim-text">${escapeHtml(claim.claim || '')}</div>
      ${claim.note ? `<div class="claim-note">${escapeHtml(claim.note)}</div>` : ''}
    `;
    claimsList.appendChild(item);
  });

  // Contradictions
  const contradictions = fc.contradictions || [];
  if (contradictions.length > 0) {
    $('contradictionsBlock').hidden = false;
    const ul = $('contradictionsList');
    ul.innerHTML = contradictions.map(c => `<li>${escapeHtml(c)}</li>`).join('');
  } else {
    $('contradictionsBlock').hidden = true;
  }

  // Unverified
  const unverified = fc.unverified_claims || [];
  if (unverified.length > 0) {
    $('unverifiedBlock').hidden = false;
    $('unverifiedList').innerHTML = unverified.map(c => `<li>${escapeHtml(c)}</li>`).join('');
  } else {
    $('unverifiedBlock').hidden = true;
  }
}

// ── Tab Navigation ─────────────────────────────────────────────────────────

function switchTab(tabName) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => { p.classList.remove('active'); p.hidden = true; });

  const activeTab = document.querySelector(`.tab[data-tab="${tabName}"]`);
  if (activeTab) activeTab.classList.add('active');

  const activePanel = $(`tab-${tabName}`);
  if (activePanel) { activePanel.classList.add('active'); activePanel.hidden = false; }
}

// ── History ────────────────────────────────────────────────────────────────

function saveToHistory(query, report) {
  const entry = {
    query: query.slice(0, 80),
    verdict: report.fact_check?.verdict || 'UNCERTAIN',
    emoji: report.fact_check?.verdict_emoji || '❓',
    timestamp: Date.now(),
  };
  state.history.unshift(entry);
  state.history = state.history.slice(0, 10);
  localStorage.setItem('research_history', JSON.stringify(state.history));
}

function renderHistory() {
  const list = $('historyList');
  if (!state.history.length) {
    list.innerHTML = '<li class="history-empty">No queries yet.</li>';
    return;
  }
  list.innerHTML = state.history.map((h, i) => `
    <li class="history-item" onclick="rerunQuery(${i})">
      <span class="history-query">${escapeHtml(h.query)}</span>
      <span class="history-verdict">${h.emoji}</span>
    </li>
  `).join('');
}

function rerunQuery(index) {
  const entry = state.history[index];
  if (!entry) return;
  $('queryInput').value = entry.query;
  $('charCount').textContent = entry.query.length;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Actions ────────────────────────────────────────────────────────────────

function copyReport() {
  if (!state.currentReport) return;
  const r = state.currentReport;
  const text = [
    `# Research Report: ${r.query}`,
    `Confidence: ${Math.round((r.overall_confidence || 0) * 100)}% | Verdict: ${r.fact_check?.verdict || 'N/A'}`,
    '',
    '## Summary',
    r.summary?.summary || '',
    '',
    '## Key Points',
    (r.summary?.key_points || []).map(p => `• ${p}`).join('\n'),
    '',
    '## Sources',
    (r.sources || []).slice(0, 5).map((s, i) => `${i+1}. ${s.title} — ${s.url}`).join('\n'),
  ].join('\n');

  navigator.clipboard.writeText(text)
    .then(() => showToast('Report copied to clipboard!'))
    .catch(() => showToast('Copy failed. Use browser copy.'));
}

function downloadReport() {
  if (!state.currentReport) return;
  const blob = new Blob([JSON.stringify(state.currentReport, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `research-report-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Report downloaded!');
}

function clearResults() {
  hideResults();
  hideError();
  $('queryInput').value = '';
  $('charCount').textContent = '0';
  resetPipeline();
  state.currentReport = null;
  $('queryInput').focus();
}

// ── UI Helpers ─────────────────────────────────────────────────────────────

function showResults() { $('resultsPanel').hidden = false; }
function hideResults() { $('resultsPanel').hidden = true; }
function showError(msg) { $('errorPanel').hidden = false; $('errorMessage').textContent = msg; }
function hideError() { $('errorPanel').hidden = true; }

function setRunButton(loading) {
  const btn = $('runBtn');
  btn.disabled = loading;
  btn.classList.toggle('loading', loading);
  btn.querySelector('.run-btn-text').textContent = loading ? 'Running...' : 'Run Research';
  btn.querySelector('.run-btn-icon').textContent = loading ? '⟳' : '→';
}

function showToast(message, duration = 2800) {
  const toast = $('toast');
  toast.textContent = message;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), duration);
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function escapeHtml(str) {
  if (typeof str !== 'string') return str ?? '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}