// Phase 16 Real-Time Demonstration & Observability Console JavaScript

(function () {
  'use strict';

  // State
  let isWorkloadRunning = false;
  let isDemoRunning = false;
  let lastSeenRecordId = 0;
  let streamRecords = [];
  let pollInterval = null;

  // DOM Elements
  const clusterStatusDot = document.getElementById('cluster-status-dot');
  const clusterStatusLabel = document.getElementById('cluster-status-label');
  const headerActiveAlgo = document.getElementById('header-active-algo');
  const algoSelect = document.getElementById('algo-select');
  const algoTypeBadge = document.getElementById('algo-type-badge');
  const btnApplyAlgo = document.getElementById('btn-apply-algo');
  const algoStatusMsg = document.getElementById('algo-status-msg');

  const scenarioSelect = document.getElementById('scenario-select');
  const inpNumRequests = document.getElementById('inp-num-requests');
  const inpConcurrency = document.getElementById('inp-concurrency');
  const inpDuration = document.getElementById('inp-duration');
  const inpPriorityProfile = document.getElementById('inp-priority-profile');
  const btnStartWorkload = document.getElementById('btn-start-workload');
  const btnStopWorkload = document.getElementById('btn-stop-workload');
  const workloadStatusPill = document.getElementById('workload-status-pill');

  const faultNodesContainer = document.getElementById('fault-nodes-container');
  const topoLbAlgo = document.getElementById('topo-lb-algo');
  const topoTrafficRate = document.getElementById('topo-traffic-rate');
  const topoBackendsContainer = document.getElementById('topo-backends-container');

  const kpiTotalRequests = document.getElementById('kpi-total-requests');
  const kpiSuccessSub = document.getElementById('kpi-success-sub');
  const kpiThroughput = document.getElementById('kpi-throughput');
  const kpiAvgLatency = document.getElementById('kpi-avg-latency');
  const kpiP95Latency = document.getElementById('kpi-p95-latency');
  const kpiRoutingOverhead = document.getElementById('kpi-routing-overhead');

  const chartBackendDistribution = document.getElementById('chart-backend-distribution');
  const chartAdaptiveDistribution = document.getElementById('chart-adaptive-distribution');
  const streamTableBody = document.getElementById('stream-table-body');
  const btnClearStream = document.getElementById('btn-clear-stream');

  const btnRunGuidedDemo = document.getElementById('btn-run-guided-demo');
  const guidedDemoProgressBox = document.getElementById('guided-demo-progress-box');
  const demoStageTitle = document.getElementById('demo-stage-title');
  const demoProgressBar = document.getElementById('demo-progress-bar');
  const demoStageDesc = document.getElementById('demo-stage-desc');
  const demoLogsConsole = document.getElementById('demo-logs-console');

  const btnExportJson = document.getElementById('btn-export-json');
  const btnExportCsv = document.getElementById('btn-export-csv');
  const btnRefreshCluster = document.getElementById('btn-refresh-cluster');

  // Algorithm Type Classifications
  const ALGO_TYPES = {
    round_robin: 'Traditional',
    least_connections: 'Traditional',
    ip_hash: 'Traditional',
    logistic_regression: 'Machine Learning',
    random_forest: 'Machine Learning',
    decision_tree: 'Machine Learning',
    svm: 'Machine Learning',
    xgboost: 'Machine Learning',
    adaptive_meta: 'Adaptive ML',
    priority_adaptive: 'Priority Adaptive',
  };

  // Preset Scenario Definitions
  const PRESET_MAP = {
    // Calibrated Phase 16 presets
    stable_normal: { num: 30, conc: 2, dur: 0.015, priority: 'equal' },
    dynamic_moderate: { num: 40, conc: 5, dur: 0.035, priority: 'equal' },
    burst_spike: { num: 50, conc: 10, dur: 0.045, priority: 'equal' },
    sustained_stress: { num: 60, conc: 14, dur: 0.080, priority: 'equal' },
    priority_conflict: { num: 45, conc: 8, dur: 0.040, priority: 'conflict' },
    adaptive_multiphase: { num: 75, conc: 8, dur: 0.040, priority: 'mixed' },
    // Legacy Presets
    steady_state: { num: 30, conc: 5, dur: 0.03, priority: 'equal' },
    burst_traffic: { num: 50, conc: 10, dur: 0.04, priority: 'equal' },
    stress_overload: { num: 60, conc: 12, dur: 0.06, priority: 'equal' },
    mixed_priority: { num: 40, conc: 6, dur: 0.03, priority: 'mixed' },
    high_contention_conflict: { num: 40, conc: 8, dur: 0.04, priority: 'conflict' },
  };

  // -------------------------------------------------------------------------
  // Initialization & Event Listeners
  // -------------------------------------------------------------------------

  function init() {
    algoSelect.addEventListener('change', updateAlgoTypeBadge);
    btnApplyAlgo.addEventListener('click', onApplyAlgorithm);
    scenarioSelect.addEventListener('change', onScenarioSelectChange);
    btnStartWorkload.addEventListener('click', onStartWorkload);
    btnStopWorkload.addEventListener('click', onStopWorkload);
    btnRunGuidedDemo.addEventListener('click', onRunGuidedDemo);
    btnClearStream.addEventListener('click', onClearStream);
    btnRefreshCluster.addEventListener('click', fetchClusterStatus);
    btnExportJson.addEventListener('click', () => window.open('/api/export?format=json', '_blank'));
    btnExportCsv.addEventListener('click', () => window.open('/api/export?format=csv', '_blank'));

    // Start background polling
    fetchClusterStatus();
    fetchTelemetry();
    pollInterval = setInterval(pollLoop, 1000);
  }

  function updateAlgoTypeBadge() {
    const selected = algoSelect.value;
    algoTypeBadge.textContent = ALGO_TYPES[selected] || 'Algorithm';
  }

  function onScenarioSelectChange() {
    const val = scenarioSelect.value;
    if (PRESET_MAP[val]) {
      const p = PRESET_MAP[val];
      inpNumRequests.value = p.num;
      inpConcurrency.value = p.conc;
      inpDuration.value = p.dur;
      inpPriorityProfile.value = p.priority;
    }
  }

  // -------------------------------------------------------------------------
  // Polling Loop
  // -------------------------------------------------------------------------

  function pollLoop() {
    fetchClusterStatus();
    fetchTelemetry();
    if (isDemoRunning) {
      fetchDemoStatus();
    }
  }

  // -------------------------------------------------------------------------
  // API Calls & Rendering: Cluster Status
  // -------------------------------------------------------------------------

  async function fetchClusterStatus() {
    try {
      const resp = await fetch('/api/status');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      renderClusterStatus(data);
    } catch (err) {
      clusterStatusDot.className = 'status-dot offline';
      clusterStatusLabel.textContent = 'Cluster Offline / Unreachable';
    }
  }

  function renderClusterStatus(data) {
    const lb = data.load_balancer || {};
    const backends = data.backends || [];

    // Header LB Status
    if (lb.status === 'running') {
      clusterStatusDot.className = 'status-dot online';
      clusterStatusLabel.textContent = 'Cluster Online (3 Backends)';
      headerActiveAlgo.textContent = lb.algorithm || 'unknown';
      topoLbAlgo.textContent = lb.algorithm || 'unknown';
    } else {
      clusterStatusDot.className = 'status-dot offline';
      clusterStatusLabel.textContent = 'Load Balancer Down';
    }

    // Workload running pill
    isWorkloadRunning = !!data.workload_running;
    if (isWorkloadRunning) {
      workloadStatusPill.className = 'status-pill running';
      workloadStatusPill.textContent = 'RUNNING';
      btnStartWorkload.disabled = true;
      btnStopWorkload.disabled = false;
    } else {
      workloadStatusPill.className = 'status-pill';
      workloadStatusPill.textContent = 'IDLE';
      btnStartWorkload.disabled = false;
      btnStopWorkload.disabled = true;
    }

    // Render Fault Injection Controls
    renderFaultControls(backends);

    // Render Topology Backends
    renderTopologyBackends(backends);
  }

  function renderFaultControls(backends) {
    faultNodesContainer.innerHTML = '';
    backends.forEach((b) => {
      const row = document.createElement('div');
      row.className = 'node-toggle-row';

      const nameSpan = document.createElement('span');
      nameSpan.className = 'node-name';
      nameSpan.textContent = `Node ${b.port} (${b.healthy ? 'Online' : 'Stopped'})`;

      const btn = document.createElement('button');
      if (b.healthy) {
        btn.className = 'btn btn-sm btn-outline-danger';
        btn.textContent = 'Simulate Crash';
        btn.onclick = () => toggleBackendFault(b.url, 'stop');
      } else {
        btn.className = 'btn btn-sm btn-outline-success';
        btn.textContent = 'Restore Node';
        btn.onclick = () => toggleBackendFault(b.url, 'start');
      }

      row.appendChild(nameSpan);
      row.appendChild(btn);
      faultNodesContainer.appendChild(row);
    });
  }

  function renderTopologyBackends(backends) {
    topoBackendsContainer.innerHTML = '';
    backends.forEach((b) => {
      const card = document.createElement('div');
      card.className = `topology-node backend-node ${b.healthy ? 'healthy' : 'unreachable'}`;

      const icon = document.createElement('div');
      icon.className = 'node-icon';
      icon.textContent = b.healthy ? '🖥️' : '⚠️';

      const title = document.createElement('div');
      title.className = 'node-title';
      title.textContent = `Node ${b.port}`;

      const badge = document.createElement('div');
      badge.className = `node-status-badge ${b.healthy ? 'healthy' : 'down'}`;
      badge.textContent = b.status_label || (b.healthy ? 'HEALTHY' : 'DOWN');

      const stats = document.createElement('div');
      stats.className = 'node-stats';
      stats.textContent = `CPU: ${Math.round(b.cpu_percent)}% | Conns: ${b.active_connections}`;

      card.appendChild(icon);
      card.appendChild(title);
      card.appendChild(badge);
      card.appendChild(stats);
      topoBackendsContainer.appendChild(card);
    });
  }

  async function toggleBackendFault(backendUrl, action) {
    try {
      const resp = await fetch('/api/backend/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend: backendUrl, action }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      await fetchClusterStatus();
    } catch (err) {
      alert(`Fault injection error: ${err.message}`);
    }
  }

  // -------------------------------------------------------------------------
  // Algorithm Switching
  // -------------------------------------------------------------------------

  async function onApplyAlgorithm() {
    const selected = algoSelect.value;
    btnApplyAlgo.disabled = true;
    algoStatusMsg.className = 'msg-box';
    algoStatusMsg.textContent = 'Applying...';

    try {
      const resp = await fetch('/api/algorithm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithm: selected }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'Failed to switch algorithm');

      algoStatusMsg.className = 'msg-box success';
      algoStatusMsg.textContent = `Successfully active: ${data.algorithm} (${data.router_class})`;
      headerActiveAlgo.textContent = data.algorithm;
      topoLbAlgo.textContent = data.algorithm;
    } catch (err) {
      algoStatusMsg.className = 'msg-box error';
      algoStatusMsg.textContent = `Error: ${err.message}`;
    } finally {
      btnApplyAlgo.disabled = false;
      setTimeout(() => {
        algoStatusMsg.style.display = 'none';
      }, 4000);
    }
  }

  // -------------------------------------------------------------------------
  // Workload Execution
  // -------------------------------------------------------------------------

  async function onStartWorkload() {
    const scenario = scenarioSelect.value;
    const num_requests = parseInt(inpNumRequests.value, 10);
    const concurrency = parseInt(inpConcurrency.value, 10);
    const request_duration = parseFloat(inpDuration.value);
    const priority_profile = inpPriorityProfile.value;

    btnStartWorkload.disabled = true;
    try {
      const resp = await fetch('/api/workload/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario,
          num_requests,
          concurrency,
          request_duration,
          priority_profile,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'Failed to launch workload');
      btnStopWorkload.disabled = false;
    } catch (err) {
      alert(`Workload launch error: ${err.message}`);
      btnStartWorkload.disabled = false;
    }
  }

  async function onStopWorkload() {
    btnStopWorkload.disabled = true;
    try {
      await fetch('/api/workload/stop', { method: 'POST' });
    } catch (err) {
      console.error(err);
    }
  }

  // -------------------------------------------------------------------------
  // Telemetry & Charts
  // -------------------------------------------------------------------------

  async function fetchTelemetry() {
    try {
      const resp = await fetch('/api/telemetry');
      if (!resp.ok) return;
      const data = await resp.json();
      renderTelemetry(data);
    } catch (err) {
      console.warn('Telemetry fetch error:', err);
    }
  }

  function renderTelemetry(data) {
    // KPIs
    kpiTotalRequests.textContent = data.total_requests || 0;
    kpiSuccessSub.textContent = `${data.successful_requests || 0} succ / ${data.failed_requests || 0} fail`;
    kpiThroughput.textContent = (data.throughput_rps || 0.0).toFixed(1);
    topoTrafficRate.textContent = `${(data.throughput_rps || 0.0).toFixed(1)} req/s`;
    kpiAvgLatency.textContent = (data.avg_latency_ms || 0.0).toFixed(2);
    kpiP95Latency.textContent = (data.p95_latency_ms || 0.0).toFixed(2);
    kpiRoutingOverhead.textContent = (data.avg_overhead_ms || 0.0).toFixed(3);

    // Backend Distribution Chart
    renderBackendChart(data.backend_distribution || {}, data.total_requests || 0);

    // Adaptive Distribution Chart
    renderAdaptiveChart(data.adaptive_distribution || {});

    // Live Request Stream Table
    renderStreamTable(data.latest_records || []);
  }

  function renderBackendChart(dist, total) {
    chartBackendDistribution.innerHTML = '';
    const entries = Object.entries(dist);
    if (entries.length === 0 || total === 0) {
      chartBackendDistribution.innerHTML = '<div class="empty-state-text">No traffic processed yet</div>';
      return;
    }

    entries.forEach(([backend, count]) => {
      const pct = total > 0 ? Math.round((count / total) * 100) : 0;
      const row = document.createElement('div');
      row.className = 'bar-row';

      const port = backend.split(':').pop().replace('/', '');
      row.innerHTML = `
        <div class="bar-row-header">
          <span><strong>Node ${port}</strong> (${backend})</span>
          <span>${count} reqs (${pct}%)</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width: ${pct}%"></div>
        </div>
      `;
      chartBackendDistribution.appendChild(row);
    });
  }

  function renderAdaptiveChart(dist) {
    chartAdaptiveDistribution.innerHTML = '';
    const entries = Object.entries(dist);
    if (entries.length === 0) {
      chartAdaptiveDistribution.innerHTML = '<div class="empty-state-text">Active when using Adaptive Meta-Selector or Priority Adaptive Router</div>';
      return;
    }

    const total = entries.reduce((acc, [, c]) => acc + c, 0);
    entries.forEach(([model, count]) => {
      const pct = total > 0 ? Math.round((count / total) * 100) : 0;
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML = `
        <div class="bar-row-header">
          <span><strong>${model}</strong></span>
          <span>${count} picks (${pct}%)</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width: ${pct}%; background: linear-gradient(90deg, #8b5cf6, #c084fc);"></div>
        </div>
      `;
      chartAdaptiveDistribution.appendChild(row);
    });
  }

  function renderStreamTable(records) {
    if (!records || records.length === 0) return;

    // Filter to only new or recent records
    const reversed = [...records].reverse();
    streamTableBody.innerHTML = '';

    reversed.forEach((r) => {
      const tr = document.createElement('tr');

      const pVal = r.request_priority || 'NORMAL';
      const statusClass = r.success ? 'ok' : 'fail';
      const statusText = r.status_code ? `${r.status_code} ${r.success ? 'OK' : 'ERR'}` : 'FAILED';
      const overhead = r.routing_overhead_ms !== undefined ? `${r.routing_overhead_ms.toFixed(3)} ms` : '—';
      const latency = r.response_time !== undefined ? `${r.response_time.toFixed(2)} ms` : '—';

      let mlInfo = '—';
      if (r.ml_predicted_server) {
        const conf = r.ml_confidence ? ` (${Math.round(r.ml_confidence * 100)}%)` : '';
        const fb = r.ml_fallback ? ' [FALLBACK]' : '';
        mlInfo = `${r.ml_predicted_server}${conf}${fb}`;
      }

      const selModel = r.selected_model || '—';

      tr.innerHTML = `
        <td>#${r.request_id}</td>
        <td><span class="priority-tag ${pVal}">${pVal}</span></td>
        <td>${r.backend_server || '—'}</td>
        <td><span class="status-tag ${statusClass}">${statusText}</span></td>
        <td>${latency}</td>
        <td>${overhead}</td>
        <td>${mlInfo}</td>
        <td><strong>${selModel}</strong></td>
      `;
      streamTableBody.appendChild(tr);
    });
  }

  function onClearStream() {
    streamTableBody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">Stream cleared. Launch a workload to observe live routing decisions.</td></tr>';
  }

  // -------------------------------------------------------------------------
  // Guided Professor Demo Runner
  // -------------------------------------------------------------------------

  async function onRunGuidedDemo() {
    btnRunGuidedDemo.disabled = true;
    guidedDemoProgressBox.style.display = 'block';
    isDemoRunning = true;

    try {
      const resp = await fetch('/api/demo/run', { method: 'POST' });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'Failed to start guided demo');
    } catch (err) {
      alert(`Demo start error: ${err.message}`);
      btnRunGuidedDemo.disabled = false;
      isDemoRunning = false;
    }
  }

  async function fetchDemoStatus() {
    try {
      const resp = await fetch('/api/demo/status');
      if (!resp.ok) return;
      const st = await resp.json();

      demoStageTitle.textContent = st.step_title || 'Guided Demo';
      demoStageDesc.textContent = st.step_description || '';
      const pct = st.total_steps > 0 ? Math.round((st.current_step / st.total_steps) * 100) : 0;
      demoProgressBar.style.width = `${pct}%`;

      if (st.logs && st.logs.length > 0) {
        demoLogsConsole.innerHTML = st.logs.map((l) => `<div>&gt; ${l}</div>`).join('');
        demoLogsConsole.scrollTop = demoLogsConsole.scrollHeight;
      }

      if (!st.is_running) {
        isDemoRunning = false;
        btnRunGuidedDemo.disabled = false;
      }
    } catch (err) {
      console.warn('Demo status fetch error:', err);
    }
  }

  // Kick off on page load
  document.addEventListener('DOMContentLoaded', init);
})();

