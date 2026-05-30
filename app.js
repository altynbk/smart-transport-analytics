

const ASSETS = {
  data: {
    aps:          'assets/data/scania_aps_results.json',
    ved:          'assets/data/ved_anomaly_results.json',
    obd:          'assets/data/obd_stress_results.json',
    can:          'assets/data/can_intrusion_results.json',
    obdScenarios: 'assets/data/obd_stress_scenarios.json',
  },
  models: {
    obd: 'assets/models/obd_stress_xgb.onnx',
  },
};

const state = {
  apsResults: null,
  vedResults: null,
  obdResults: null,
  canResults: null,
  obdScenarios: null,
  obdSession: null,
  sliderValues: {},
  pendingInference: null,
  canAnimationInterval: null,
};

const VERDICT_THRESHOLDS = {
  safe: 0.30,   
  warn: 0.65,   

};

document.addEventListener('DOMContentLoaded', async () => {
  if (typeof lucide !== 'undefined') lucide.createIcons();

  try {
    await Promise.all([
      loadJson('aps'),
      loadJson('ved'),
      loadJson('obd'),
      loadJson('can'),
      loadJson('obdScenarios'),
    ]);
  } catch (err) {
    console.warn('Some artifacts failed to load:', err);
  }

  populateApsStats();
  populateObdStats();
  populateVedTable();

  loadFieldValidation();

  initializeSliders();
  setupPresets();
  setupFAQ();

  loadObdModel().then(() => {
    setTryItStatus('ok', '✓ Model loaded · ready to predict');
    if (state.obdSession) runObdInference();
  }).catch((err) => {
    console.warn('OBD ONNX load failed:', err);
    setTryItStatus('error', '✗ Model failed to load');
  });

  if (typeof lucide !== 'undefined') lucide.createIcons();
});

async function loadJson(key) {
  const path = ASSETS.data[key];
  if (!path) return;
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (key === 'aps')              state.apsResults  = data;
    else if (key === 'ved')         state.vedResults  = data;
    else if (key === 'obd')         state.obdResults  = data;
    else if (key === 'can')         state.canResults  = data;
    else if (key === 'obdScenarios') state.obdScenarios = data;
  } catch (err) {
    console.warn(`Failed to load ${key} from ${path}:`, err);
  }
}

function populateApsStats() {
  if (!state.apsResults) return;
  const r = state.apsResults;
  const totalCost = safeNum(r?.results?.total_cost);
  const auc = safeNum(r?.model?.roc_auc);
  const rank = r?.baselines_comparison?.our_rank;
  const total = r?.baselines_comparison?.total_baselines_compared;
  const defaultCost = safeNum(r?.results?.default_threshold_cost);

  setText('aps-total-cost', totalCost != null ? totalCost.toLocaleString() : '8,910');
  setText('aps-auc', auc != null ? auc.toFixed(4) : '0.9958');

  if (rank != null && total != null) {
    setText('aps-rank', `${rank}/${total}`);
  }

  if (defaultCost && totalCost) {
    const improvement = (defaultCost - totalCost) / defaultCost * 100;
    setText('aps-improvement', `${improvement.toFixed(1)}%`);
  }
}

function populateObdStats() {
  if (!state.obdResults) return;
  const acc = safeNum(state.obdResults?.model?.test_accuracy);
  if (acc != null) {
    const pct = (acc * 100).toFixed(2) + '%';
    setText('obd-accuracy', pct);
    setText('nav-obd-acc', pct);
  }
}

function populateVedTable() {
  const tbody = document.querySelector('#ved-top-table tbody');
  if (!tbody) return;

  let trips = state.vedResults?.top20_trips;
  if (!Array.isArray(trips) || trips.length === 0) {
    trips = HARDCODED_VED_TOP20;
  }

  const top10 = trips.slice(0, 10);
  tbody.innerHTML = '';

  top10.forEach((trip) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="rank-cell">${trip.rank}</td>
      <td>${trip.vehicle_id}</td>
      <td>${trip.trip_id}</td>
      <td class="score-cell">${trip.anomaly_score.toFixed(3)}</td>
      <td>${trip.speed_mean.toFixed(1)} km/h</td>
      <td>${trip.rpm_max.toFixed(0)} rpm</td>
      <td>${(trip.idle_fraction * 100).toFixed(1)}%</td>
      <td>${formatDuration(trip.duration_sec)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function formatDuration(seconds) {
  if (!seconds) return '—';
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs.toString().padStart(2, '0')}s`;
}

async function loadObdModel() {
  if (typeof ort === 'undefined') {
    throw new Error('ONNX Runtime Web not loaded');
  }

  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/';

  setTryItStatus('loading', '⏳ Loading model…');

  const t0 = performance.now();
  const session = await ort.InferenceSession.create(ASSETS.models.obd, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  const loadTime = (performance.now() - t0).toFixed(0);

  state.obdSession = session;
  console.log(`OBD ONNX loaded in ${loadTime} ms`);
  console.log(`  Input names: ${session.inputNames.join(', ')}`);
  console.log(`  Output names: ${session.outputNames.join(', ')}`);
}

function setTryItStatus(status, text) {
  const el = document.getElementById('obd-model-status');
  if (!el) return;
  el.className = 'tryit-status';
  if (status === 'ok') el.classList.add('status-ok');
  if (status === 'error') el.classList.add('status-error');
  const iconName = status === 'ok' ? 'check-circle' : status === 'error' ? 'alert-triangle' : 'loader';
  el.innerHTML = `<i data-lucide="${iconName}"></i><span>${text}</span>`;
  if (typeof lucide !== 'undefined') lucide.createIcons();
}

async function loadFieldValidation() {
  try {
    const r = await fetch('assets/data/field_validation_results.json');
    if (!r.ok) {
      console.log('Field validation JSON not yet available — section will show placeholders.');
      return;
    }
    const d = await r.json();

    const setText = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    const ds = d.dataset || {};
    const nTrips  = ds.n_trips || 0;
    const hours   = (ds.total_duration_min || 0) / 60;
    const samples = ds.total_samples_1hz || 0;
    const drivers = ds.drivers || [];
    const vehicles = ds.vehicles || [];

    setText('field-trips',   nTrips);
    setText('field-hours',   hours.toFixed(1));
    setText('field-samples', samples.toLocaleString());
    setText('field-drivers', `${drivers.length} · ${vehicles.length}`);

    setText('nav-field-trips', nTrips);
    setText('nav-field-hours', hours.toFixed(1));

    console.log(`Field validation loaded: ${nTrips} trips, ${hours.toFixed(1)} h, ${samples.toLocaleString()} samples`);
  } catch (e) {
    console.warn('loadFieldValidation failed:', e);
  }
}

const FALLBACK_RANGES = {
  'ENGINE_RPM':                {'min': 600, 'max': 7000, 'step': 50, 'default': 1500, 'unit': 'rpm'},
  'THROTTLE_POS':              {'min': 0,   'max': 100,  'step': 1,  'default': 15,   'unit': '%'},
  'ENGINE_COOLANT_TEMP':       {'min': 40,  'max': 130,  'step': 1,  'default': 90,   'unit': '°C'},
  'ENGINE_LOAD':               {'min': 0,   'max': 100,  'step': 1,  'default': 30,   'unit': '%'},
  'SPEED':                     {'min': 0,   'max': 200,  'step': 1,  'default': 60,   'unit': 'km/h'},
  'INTAKE_MANIFOLD_PRESSURE':  {'min': 20,  'max': 250,  'step': 1,  'default': 60,   'unit': 'kPa'},
};

const DISPLAY_NAMES = {
  'ENGINE_RPM':                'Engine RPM',
  'THROTTLE_POS':              'Throttle position',
  'ENGINE_COOLANT_TEMP':       'Coolant temperature',
  'ENGINE_LOAD':               'Engine load',
  'SPEED':                     'Vehicle speed',
  'INTAKE_MANIFOLD_PRESSURE':  'Manifold pressure',
};

const NORMAL_HINTS = {
  'ENGINE_RPM':                '800–2500 rpm',
  'THROTTLE_POS':              '5–30 %',
  'ENGINE_COOLANT_TEMP':       '75–95 °C',
  'ENGINE_LOAD':               '15–50 %',
  'SPEED':                     '0–120 km/h',
  'INTAKE_MANIFOLD_PRESSURE':  '30–70 kPa',
};

function getSliderConfig() {
  const ranges = state.obdScenarios?.slider_ranges
              || state.obdResults?.slider_ranges
              || FALLBACK_RANGES;
  const features = state.obdScenarios?.feature_cols
                || state.obdResults?.dataset?.feature_cols
                || Object.keys(FALLBACK_RANGES);
  return { ranges, features };
}

function initializeSliders() {
  const container = document.getElementById('sliders-container');
  if (!container) return;

  const { ranges, features } = getSliderConfig();
  container.innerHTML = '';

  features.forEach((feat) => {
    const r = ranges[feat] || FALLBACK_RANGES[feat];
    if (!r) return;

    state.sliderValues[feat] = r.default;

    const row = document.createElement('div');
    row.className = 'slider-row';
    row.innerHTML = `
      <div class="slider-head">
        <span class="slider-name">${DISPLAY_NAMES[feat] || feat}</span>
        <span class="slider-value" id="val-${slugify(feat)}">
          ${r.default} <em>${r.unit}</em>
        </span>
      </div>
      <input type="range"
             class="slider-input"
             id="sl-${slugify(feat)}"
             min="${r.min}" max="${r.max}" step="${r.step}"
             value="${r.default}"
             data-feature="${feat}" data-unit="${r.unit}">
      <div class="slider-range-note">normal: ${NORMAL_HINTS[feat] || '—'}</div>
    `;
    container.appendChild(row);
  });

  document.querySelectorAll('.slider-input').forEach((slider) => {
    slider.addEventListener('input', onSliderChange);
  });
}

function slugify(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-');
}

function onSliderChange(e) {
  const feature = e.target.dataset.feature;
  const unit = e.target.dataset.unit;
  const value = Number(e.target.value);

  state.sliderValues[feature] = value;
  const valEl = document.getElementById(`val-${slugify(feature)}`);
  if (valEl) valEl.innerHTML = `${value} <em>${unit}</em>`;

  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));

  if (state.pendingInference) clearTimeout(state.pendingInference);
  state.pendingInference = setTimeout(runObdInference, 50);
}

function setupPresets() {
  document.querySelectorAll('.preset-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const scenarioName = btn.dataset.scenario;
      applyPreset(scenarioName);

      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
}

function applyPreset(scenarioName) {
  const scenarios = state.obdScenarios?.scenarios || state.obdResults?.scenarios;
  if (!scenarios) return;

  const scenario = scenarios.find(s => s.scenario === scenarioName);
  if (!scenario) return;

  const { ranges, features } = getSliderConfig();

  features.forEach((feat) => {
    const r = ranges[feat] || FALLBACK_RANGES[feat];
    if (!r) return;

    let val;
    if (scenario.features && typeof scenario.features === 'object' && !Array.isArray(scenario.features)) {
      val = scenario.features[feat];
    } else if (Array.isArray(scenario.features_array)) {
      const idx = features.indexOf(feat);
      val = scenario.features_array[idx];
    }

    if (val == null) return;

    val = Math.max(r.min, Math.min(r.max, Math.round(val / r.step) * r.step));

    state.sliderValues[feat] = val;
    const slider = document.getElementById(`sl-${slugify(feat)}`);
    if (slider) slider.value = val;
    const valEl = document.getElementById(`val-${slugify(feat)}`);
    if (valEl) valEl.innerHTML = `${val} <em>${r.unit}</em>`;
  });

  runObdInference();
}

async function runObdInference() {
  if (!state.obdSession) return;

  const { features } = getSliderConfig();
  const featureArray = new Float32Array(features.map(f => state.sliderValues[f] ?? 0));

  try {
    const inputName = state.obdSession.inputNames[0];
    const tensor = new ort.Tensor('float32', featureArray, [1, features.length]);

    const t0 = performance.now();
    const outputs = await state.obdSession.run({ [inputName]: tensor });
    const elapsedUs = ((performance.now() - t0) * 1000).toFixed(1);

    let proba = null;
    for (const name of state.obdSession.outputNames) {
      const out = outputs[name];
      if (out && out.dims.length === 2 && out.dims[1] === 3) {
        proba = Array.from(out.data);
        break;
      }
    }

    if (!proba) {
      console.error('Could not parse OBD ONNX output');
      return;
    }

    updateVerdict(proba, elapsedUs);
  } catch (err) {
    console.error('OBD inference error:', err);
  }
}

function updateVerdict(proba, elapsedUs) {

  const probaDiagnostic = proba[1];
  const probaCritical = proba[2];

  

  

  const modelRisk = (probaDiagnostic * 0.5 + probaCritical * 1.0);
  const deviationRisk = computeDeviationRisk();
  const risk = Math.max(0, Math.min(1, 0.7 * deviationRisk + 0.3 * modelRisk));

  let verdictLabel, verdictClass, verdictIcon;
  if (risk < VERDICT_THRESHOLDS.safe) {
    verdictLabel = 'SAFE TO DRIVE';
    verdictClass = 'verdict-safe';
    verdictIcon = 'check-circle';
  } else if (risk < VERDICT_THRESHOLDS.warn) {
    verdictLabel = 'DIAGNOSTIC NEEDED';
    verdictClass = 'verdict-warn';
    verdictIcon = 'alert-circle';
  } else {
    verdictLabel = 'DO NOT DRIVE';
    verdictClass = 'verdict-danger';
    verdictIcon = 'alert-triangle';
  }

  const badge = document.getElementById('verdict-badge');
  if (badge) badge.className = 'verdict-badge ' + verdictClass;

  const iconEl = document.getElementById('verdict-icon');
  if (iconEl) iconEl.innerHTML = `<i data-lucide="${verdictIcon}"></i>`;

  setText('verdict-label', verdictLabel);

  setText('gauge-value', `${(risk * 100).toFixed(0)}%`);
  const marker = document.getElementById('gauge-marker');
  if (marker) marker.style.left = `${risk * 100}%`;

  updateContributors();

  setText('verdict-latency', `${elapsedUs} µs`);

  if (typeof lucide !== 'undefined') lucide.createIcons();
}

function computeDeviationRisk() {
  const features = Object.keys(NORMAL_RANGES);
  let sum = 0;
  for (const f of features) {
    sum += deviationScore(f, state.sliderValues[f] ?? 0);
  }
  return Math.min(1, sum / 3);
}

const NORMAL_RANGES = {
  'ENGINE_RPM':                [800, 2500],
  'THROTTLE_POS':              [5, 30],
  'ENGINE_COOLANT_TEMP':       [75, 95],
  'ENGINE_LOAD':               [15, 50],
  'SPEED':                     [0, 120],
  'INTAKE_MANIFOLD_PRESSURE':  [30, 70],
};

const FEATURE_UNITS = {
  'ENGINE_RPM':                'rpm',
  'THROTTLE_POS':              '%',
  'ENGINE_COOLANT_TEMP':       '°C',
  'ENGINE_LOAD':               '%',
  'SPEED':                     'km/h',
  'INTAKE_MANIFOLD_PRESSURE':  'kPa',
};

const FEATURE_DISPLAY_NAMES = {
  'ENGINE_RPM':                'Engine RPM',
  'THROTTLE_POS':              'Throttle position',
  'ENGINE_COOLANT_TEMP':       'Coolant temperature',
  'ENGINE_LOAD':               'Engine load',
  'SPEED':                     'Vehicle speed',
  'INTAKE_MANIFOLD_PRESSURE':  'Manifold pressure',
};

const RECOMMENDATIONS = {
  'ENGINE_RPM': {
    low:    'Engine speed below idle range — possible stalling or idle-control fault.',
    high:   'Engine speed elevated — heavy throttle or downshift; sustained high RPM increases wear.',
    normal: 'Engine RPM within typical operating range.',
  },
  'THROTTLE_POS': {
    low:    'Throttle near closed — coasting or idle.',
    high:   'Throttle wide open — aggressive acceleration loads the powertrain.',
    normal: 'Throttle position is normal.',
  },
  'ENGINE_COOLANT_TEMP': {
    low:    'Coolant temperature below normal — engine not yet warm.',
    high:   'Coolant overheating — risk of engine damage; reduce load and stop if rising.',
    normal: 'Coolant temperature is in the safe operating window.',
  },
  'ENGINE_LOAD': {
    low:    'Engine load very low — typical for idle or coasting.',
    high:   'Engine under heavy load — sustained high values stress the powertrain.',
    normal: 'Engine load is typical for cruise / city driving.',
  },
  'SPEED': {
    low:    'Vehicle near stationary — typical for idle, stop, or low-speed manoeuvre.',
    high:   'Speed above the typical urban range.',
    normal: 'Speed within typical driving range.',
  },
  'INTAKE_MANIFOLD_PRESSURE': {
    low:    'Manifold pressure low — typical for idle, or possible vacuum leak.',
    high:   'Manifold pressure elevated — heavy throttle or forced induction.',
    normal: 'Manifold pressure is in the normal range.',
  },
};

function deviationScore(name, value) {
  const range = NORMAL_RANGES[name];
  if (!range) return 0;
  const [lo, hi] = range;
  const width = hi - lo;
  if (width <= 0) return 0;
  if (value < lo) return (lo - value) / width;
  if (value > hi) return (value - hi) / width;
  return 0;
}

function rangeStatus(name, value) {
  const range = NORMAL_RANGES[name];
  if (!range) return 'normal';
  const [lo, hi] = range;
  if (value < lo) return 'low';
  if (value > hi) return 'high';
  return 'normal';
}

function updateContributors() {
  const listEl = document.getElementById('contributors-list');
  if (!listEl) return;

  const v = state.sliderValues;

  const scored = Object.keys(NORMAL_RANGES).map((feat) => ({
    feat,
    value: v[feat],
    score: deviationScore(feat, v[feat]),
    status: rangeStatus(feat, v[feat]),
  }));

  const deviating = scored
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  listEl.innerHTML = '';

  if (deviating.length === 0) {
    const li = document.createElement('li');
    li.className = 'contrib-empty';
    li.textContent = 'All sensors within normal operating ranges.';
    listEl.appendChild(li);
    return;
  }

  deviating.forEach((s) => {
    const range = NORMAL_RANGES[s.feat];
    const unit = FEATURE_UNITS[s.feat];
    const name = FEATURE_DISPLAY_NAMES[s.feat];
    const advice = RECOMMENDATIONS[s.feat][s.status];

    const level = s.score > 1.5 ? 'danger' : 'warn';

    const li = document.createElement('li');
    li.className = `contrib-${level}`;
    li.innerHTML = `
      <div class="contrib-head">
        <span class="contrib-name">${name}</span>
        <span class="contrib-value mono">${Math.round(s.value)} ${unit}</span>
      </div>
      <div class="contrib-range">normal: ${range[0]}–${range[1]} ${unit}</div>
      <div class="contrib-advice">${advice}</div>
    `;
    listEl.appendChild(li);
  });
}

function setupFAQ() {
  document.querySelectorAll('.faq-q').forEach((btn) => {
    btn.addEventListener('click', () => {
      const item = btn.parentElement;
      const wasOpen = item.classList.contains('faq-open');

      document.querySelectorAll('.faq-item').forEach(i => i.classList.remove('faq-open'));

      if (!wasOpen) item.classList.add('faq-open');
    });
  });
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function safeNum(x) {
  if (x === null || x === undefined) return null;
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

const HARDCODED_VED_TOP20 = [
  { rank: 1,  vehicle_id: 189, trip_id: 2030, anomaly_score: 0.700, speed_mean: 2.3,   rpm_max: 1975, idle_fraction: 0.884, duration_sec: 6560 },
  { rank: 2,  vehicle_id: 189, trip_id: 2425, anomaly_score: 0.699, speed_mean: 1.0,   rpm_max: 2089, idle_fraction: 0.950, duration_sec: 7326 },
  { rank: 3,  vehicle_id: 169, trip_id: 1641, anomaly_score: 0.696, speed_mean: 2.5,   rpm_max: 2637, idle_fraction: 0.901, duration_sec: 10121 },
  { rank: 4,  vehicle_id: 454, trip_id: 1829, anomaly_score: 0.687, speed_mean: 4.2,   rpm_max: 2879, idle_fraction: 0.882, duration_sec: 4605 },
  { rank: 5,  vehicle_id: 492, trip_id: 1741, anomaly_score: 0.685, speed_mean: 88.2,  rpm_max: 4070, idle_fraction: 0.076, duration_sec: 1941 },
  { rank: 6,  vehicle_id: 542, trip_id: 1895, anomaly_score: 0.684, speed_mean: 57.0,  rpm_max: 3378, idle_fraction: 0.125, duration_sec: 1990 },
  { rank: 7,  vehicle_id: 536, trip_id: 485,  anomaly_score: 0.684, speed_mean: 80.1,  rpm_max: 3612, idle_fraction: 0.012, duration_sec: 4751 },
  { rank: 8,  vehicle_id: 569, trip_id: 1573, anomaly_score: 0.682, speed_mean: 21.5,  rpm_max: 2424, idle_fraction: 0.390, duration_sec: 8009 },
  { rank: 9,  vehicle_id: 569, trip_id: 750,  anomaly_score: 0.681, speed_mean: 84.8,  rpm_max: 3000, idle_fraction: 0.010, duration_sec: 4386 },
  { rank: 10, vehicle_id: 569, trip_id: 358,  anomaly_score: 0.680, speed_mean: 23.9,  rpm_max: 2407, idle_fraction: 0.272, duration_sec: 4569 },
];
