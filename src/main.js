import './style.css';
import { FidelityFXSort } from './sorting/FidelityFXSort.js';
import { DeviceRadixSort } from './sorting/DeviceRadixSort.js';
import { OneSweep } from './sorting/OneSweep.js';
import { generateTestData, validateSort, compareArrays, formatTime, formatNumber } from './utils.js';

// WebGPU device and context
let device = null;
let fidelityFXSort = null;
let deviceRadixSort = null;
let oneSweep = null;

// Initialize the application
async function init() {
  const app = document.getElementById('app');
  app.innerHTML = `
    <div class="container mx-auto px-4 py-8">
      <div class="flex items-center justify-between mb-8 px-2">
        <h1 class="text-4xl font-bold text-gray-100">WebGPU Sorting Comparison</h1>
        <a href="https://github.com/metarapi/webgpu-sorting" target="_blank" rel="noopener" class="group">
          <span class="block group-hover:hidden">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon icon-tabler icons-tabler-outline icon-tabler-brand-github text-gray-800 dark:text-gray-200 transition-colors duration-200">
              <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
              <path d="M9 19c-4.3 1.4 -4.3 -2.5 -6 -3m12 5v-3.5c0 -1 .1 -1.4 -.5 -2c2.8 -.3 5.5 -1.4 5.5 -6a4.6 4.6 0 0 0 -1.3 -3.2a4.2 4.2 0 0 0 -.1 -3.2s-1.1 -.3 -3.5 1.3a12.3 12.3 0 0 0 -6.2 0c-2.4 -1.6 -3.5 -1.3 -3.5 -1.3a4.2 4.2 0 0 0 -.1 3.2a4.6 4.6 0 0 0 -1.3 3.2c0 4.6 2.7 5.7 5.5 6c-.6 .6 -.6 1.2 -.5 2v3.5" />
            </svg>
          </span>
          <span class="hidden group-hover:block">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="currentColor" class="icon icon-tabler icons-tabler-filled icon-tabler-brand-github text-gray-800 dark:text-gray-200 transition-colors duration-200">
              <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
              <path d="M5.315 2.1c.791 -.113 1.9 .145 3.333 .966l.272 .161l.16 .1l.397 -.083a13.3 13.3 0 0 1 4.59 -.08l.456 .08l.396 .083l.161 -.1c1.385 -.84 2.487 -1.17 3.322 -1.148l.164 .008l.147 .017l.076 .014l.05 .011l.144 .047a1 1 0 0 1 .53 .514a5.2 5.2 0 0 1 .397 2.91l-.047 .267l-.046 .196l.123 .163c.574 .795 .93 1.728 1.03 2.707l.023 .295l.007 .272c0 3.855 -1.659 5.883 -4.644 6.68l-.245 .061l-.132 .029l.014 .161l.008 .157l.004 .365l-.002 .213l-.003 3.834a1 1 0 0 1 -.883 .993l-.117 .007h-6a1 1 0 0 1 -.993 -.883l-.007 -.117v-.734c-1.818 .26 -3.03 -.424 -4.11 -1.878l-.535 -.766c-.28 -.396 -.455 -.579 -.589 -.644l-.048 -.019a1 1 0 0 1 .564 -1.918c.642 .188 1.074 .568 1.57 1.239l.538 .769c.76 1.079 1.36 1.459 2.609 1.191l.001 -.678l-.018 -.168a5.03 5.03 0 0 1 -.021 -.824l.017 -.185l.019 -.12l-.108 -.024c-2.976 -.71 -4.703 -2.573 -4.875 -6.139l-.01 -.31l-.004 -.292a5.6 5.6 0 0 1 .908 -3.051l.152 -.222l.122 -.163l-.045 -.196a5.2 5.2 0 0 1 .145 -2.642l.1 -.282l.106 -.253a1 1 0 0 1 .529 -.514l.144 -.047l.154 -.03z" />
            </svg>
          </span>
        </a>
      </div>
      <p class="text-gray-400 mb-8">4-way comparison: FidelityFX vs DeviceRadixSort vs OneSweep vs JavaScript</p>
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <!-- Algorithm Selection -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 class="text-xl font-semibold mb-4">Test Mode</h2>
          <select id="algorithm-select" class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500">
            <option value="all">All Algorithms (4-way)</option>
            <option value="fidelityfx">FidelityFX Only</option>
            <option value="deviceradix">DeviceRadixSort Only</option>
            <option value="onesweep">OneSweep Only</option>
            <option value="javascript">JavaScript Only</option>
          </select>
          <p class="text-sm text-gray-400 mt-3">DeviceRadixSort and OneSweep require subgroup sizes ≥ 16 lanes.</p>
        </div>
        <!-- Array Size -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 class="text-xl font-semibold mb-4">Array Size</h2>
          <input id="array-size" type="number" value="1000000" min="1000" max="10000000" step="100000" 
                 class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500">
        </div>
        <!-- Actions -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 class="text-xl font-semibold mb-4">Actions</h2>
          <button id="run-sort" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition">
            Run Comparison
          </button>
        </div>
      </div>
      <!-- Results -->
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-6">
        <h2 class="text-xl font-semibold mb-4">Results</h2>
        <div id="results" class="space-y-2 text-gray-300">
          <p>Click \"Run Comparison\" to begin...</p>
        </div>
      </div>
      <!-- WebGPU Status -->
      <div id="webgpu-status" class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <p class="text-sm text-gray-400">Initializing WebGPU...</p>
      </div>
    </div>
  `;

  // Initialize WebGPU
  await initWebGPU();

  // Setup event listeners
  setupEventListeners();
}

async function initWebGPU() {
  const statusEl = document.getElementById('webgpu-status');
  
  if (!navigator.gpu) {
    statusEl.innerHTML = '<p class="text-sm text-red-400">❌ WebGPU is not supported in this browser</p>';
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      statusEl.innerHTML = '<p class="text-sm text-red-400">❌ Failed to get GPU adapter</p>';
      return false;
    }

    // Check for subgroups support
    if (!adapter.features.has('subgroups')) {
      statusEl.innerHTML = '<p class="text-sm text-red-400">❌ Subgroups feature not supported</p>';
      return false;
    }

    // Check for workgroup storage size support
    const limits = adapter.limits;
    const requiredWorkgroupStorage = 32768;
    if (limits.maxComputeWorkgroupStorageSize < requiredWorkgroupStorage) {
      statusEl.innerHTML = `<p class="text-sm text-red-400">❌ Insufficient workgroup storage: ${limits.maxComputeWorkgroupStorageSize} bytes (need ${requiredWorkgroupStorage} bytes)</p>`;
      return false;
    }

    // Request device with necessary features
    const features = ['subgroups'];
    if (adapter.features.has('timestamp-query')) {
      features.push('timestamp-query');
    }

    device = await adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits: {
        maxComputeWorkgroupStorageSize: requiredWorkgroupStorage
      }
    });

    // Initialize sorting algorithms
    statusEl.innerHTML = '<p class="text-sm text-blue-400">Initializing sorting algorithms...</p>';
    
    const maxKeys = 10000000; // 10M elements max
    fidelityFXSort = new FidelityFXSort(device, maxKeys);
    await fidelityFXSort.init();
    // DeviceRadixSort and OneSweep will be initialized on-demand when running tests

    let adapterInfo = null;
    try {
      if ('info' in adapter && adapter.info) {
        adapterInfo = adapter.info;
      } else if (typeof adapter.requestAdapterInfo === 'function') {
        adapterInfo = await adapter.requestAdapterInfo();
      }
    } catch (infoError) {
      console.warn('Unable to query adapter info', infoError);
    }

    const deviceLimits = device.limits;
    const formatLimit = value => (typeof value === 'number' && Number.isFinite(value)) ? formatNumber(value) : 'n/a';
    const normalizeSubgroup = value => (typeof value === 'number' && Number.isFinite(value) && value > 0) ? value : null;
    const rawMinSubgroup = normalizeSubgroup(adapterInfo?.subgroupMinSize) ?? normalizeSubgroup(deviceLimits.minSubgroupSize) ?? normalizeSubgroup(limits.minSubgroupSize);
    const rawMaxSubgroup = normalizeSubgroup(adapterInfo?.subgroupMaxSize) ?? normalizeSubgroup(deviceLimits.maxSubgroupSize) ?? normalizeSubgroup(limits.maxSubgroupSize);
    const warnings = [];
    if (rawMinSubgroup !== null && rawMinSubgroup < 16) {
      const warning = `Warning: minimum subgroup size is ${formatNumber(rawMinSubgroup)} lanes; DeviceRadixSort and OneSweep require ≥16 lanes and no SIMD8 fallback is available.`;
      warnings.push(`<p class="text-yellow-400">${warning}</p>`);
      console.warn(warning);
    }
    let subgroupRange = 'unavailable';
    if (rawMinSubgroup !== null || rawMaxSubgroup !== null) {
      const minLabel = rawMinSubgroup !== null ? formatNumber(rawMinSubgroup) : 'n/a';
      const maxLabel = rawMaxSubgroup !== null ? formatNumber(rawMaxSubgroup) : 'n/a';
      subgroupRange = (rawMinSubgroup !== null && rawMaxSubgroup !== null && rawMinSubgroup === rawMaxSubgroup)
        ? `${minLabel} lanes`
        : `${minLabel} – ${maxLabel} lanes`;
    }
    const workgroupStorageValue = formatLimit(deviceLimits.maxComputeWorkgroupStorageSize);
    const workgroupStorage = workgroupStorageValue === 'n/a'
      ? 'unavailable'
      : `${workgroupStorageValue} bytes`;

    statusEl.innerHTML = `
      <div class="text-sm space-y-1">
        <p class="text-green-400">✓ WebGPU initialized</p>
        <p class="text-gray-300"><span class="text-gray-400">Features:</span> ${features.join(', ')}</p>
        <p class="text-gray-300"><span class="text-gray-400">Subgroup size range:</span> ${subgroupRange}</p>
        <p class="text-gray-300"><span class="text-gray-400">Workgroup storage:</span> ${workgroupStorage}</p>
        ${warnings.join('')}
      </div>
    `;
    return true;
  } catch (error) {
    statusEl.innerHTML = `<p class="text-sm text-red-400">❌ Error: ${error.message}</p>`;
    console.error(error);
    return false;
  }
}

function setupEventListeners() {
  const runButton = document.getElementById('run-sort');
  const algorithmSelect = document.getElementById('algorithm-select');
  const arraySizeInput = document.getElementById('array-size');

  runButton.addEventListener('click', async () => {
    if (!device) {
      alert('WebGPU not initialized');
      return;
    }

    const mode = algorithmSelect.value;
    const arraySize = parseInt(arraySizeInput.value);

    runButton.disabled = true;
    runButton.textContent = 'Running...';

    await runSortingTest(mode, arraySize);

    runButton.disabled = false;
    runButton.textContent = 'Run Comparison';
  });
}

async function runSortingTest(mode, arraySize) {
  const resultsEl = document.getElementById('results');
  resultsEl.innerHTML = '<p class="text-blue-400">Generating test data...</p>';

  try {
    const maxKeys = Math.max(arraySize, 1000000);

    if (mode === 'all' || mode === 'deviceradix') {
      if (deviceRadixSort) deviceRadixSort.destroy();
      deviceRadixSort = new DeviceRadixSort(device, maxKeys);
      await deviceRadixSort.init();
    }
    if (mode === 'all' || mode === 'onesweep') {
      if (oneSweep) oneSweep.destroy();
      oneSweep = new OneSweep(device, maxKeys);
      await oneSweep.init();
    }

    // Generate test data
    const data = generateTestData(arraySize);
    
    resultsEl.innerHTML = '<p class="text-blue-400">Running tests...</p>';

    const results = {};

    // Run JavaScript sort for baseline
    if (mode === 'all' || mode === 'javascript') {
      const jsCopy = [...data];
      const jsStart = performance.now();
      jsCopy.sort((a, b) => a.key - b.key);
      const jsEnd = performance.now();
      
      results.javascript = {
        time: jsEnd - jsStart,
        sorted: jsCopy,
        valid: validateSort(jsCopy)
      };
    }

    // Run FidelityFX sort
    if (mode === 'all' || mode === 'fidelityfx') {
      const { sorted, gpuTime } = await fidelityFXSort.sort(data);
      results.fidelityfx = {
        time: gpuTime,
        sorted,
        valid: validateSort(sorted)
      };
    }

    // Run DeviceRadixSort
    if (mode === 'all' || mode === 'deviceradix') {
      const { sorted, gpuTime, subgroupSizes } = await deviceRadixSort.sort(data);
      results.deviceradix = {
        time: gpuTime,
        sorted,
        valid: validateSort(sorted),
        subgroupSizes
      };
    }

    // Run OneSweep
    if (mode === 'all' || mode === 'onesweep') {
      const { sorted, gpuTime, subgroupSize, shaderVariant } = await oneSweep.sort(data);
      results.onesweep = {
        time: gpuTime,
        sorted,
        valid: validateSort(sorted),
        subgroupSize,
        shaderVariant
      };
    }

    // Display results
    displayResults(results, arraySize);
  } catch (error) {
    console.error(error);
    const raw = error && error.message ? error.message : String(error);

    // Small helper to escape HTML when showing messages
    const escapeHtml = (str) => str.replace(/[&<>\"'`]/g, (ch) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
      '`': '&#96;'
    }[ch]));

    // Condense known shader/status errors into a single simple explanation
    let message = raw;
    if (/DEAD0001/i.test(raw) || /0xdead0001/i.test(raw) ||
        /DEAD0002/i.test(raw) || /0xdead0002/i.test(raw) ||
        /DEAD0004/i.test(raw) || /0xdead0004/i.test(raw) ||
        /warp hist capacity exceeded/i.test(raw)) {
      message = 'GPU sort failed: the device reported a subgroup configuration smaller than expected (subgroup < 16). Reload the page or use hardware/drivers that support subgroup ≥ 16.';
    }

    resultsEl.innerHTML = `<div class="text-red-400 font-semibold">${escapeHtml(message)}</div>`;
  }
}

function displayResults(results, arraySize) {
  const resultsEl = document.getElementById('results');
  
  let html = `<div class="space-y-4">`;
  html += `<p class="text-lg font-semibold">Array Size: ${formatNumber(arraySize)} elements</p>`;
  html += `<div class="border-t border-gray-700 pt-4">`;

  // Find fastest time for comparison
  const times = Object.values(results).map(r => r.time);
  const fastest = Math.min(...times);

  // Define baseline - use JavaScript as the reference point
  const baseline = results.javascript ? results.javascript.time : fastest;

  // Display each algorithm's results
  if (results.javascript) {
    const speedup = baseline / results.javascript.time;
    const throughput = (arraySize / (results.javascript.time / 1000)) / 1_000_000; // MKeys/s
    const zeroWarning = results.javascript.valid.zeroCount === arraySize ? 'Output is all zeros!' : null;
    html += createResultRow(
      'JavaScript Array.sort',
      results.javascript.time,
      results.javascript.valid.isSorted,
      speedup,
      throughput,
      '#60a5fa',
      fastest,
      { warning: zeroWarning }
    );
  }

  if (results.fidelityfx) {
    const speedup = baseline / results.fidelityfx.time;
    const throughput = (arraySize / (results.fidelityfx.time / 1000)) / 1_000_000; // MKeys/s
    const zeroWarning = results.fidelityfx.valid.zeroCount === arraySize ? 'Output is all zeros!' : null;
    html += createResultRow(
      'FidelityFX Radix Sort',
      results.fidelityfx.time,
      results.fidelityfx.valid.isSorted,
      speedup,
      throughput,
      '#4ade80',
      fastest,
      { warning: zeroWarning }
    );
  }

  if (results.deviceradix) {
    const speedup = baseline / results.deviceradix.time;
    const throughput = (arraySize / (results.deviceradix.time / 1000)) / 1_000_000; // MKeys/s
    const zeroWarning = results.deviceradix.valid.zeroCount === arraySize ? 'Output is all zeros!' : null;
    const subgroupExtras = (results.deviceradix.subgroupSizes || []).map(({ pass, stage, size }) => ({
      label: `Pass ${pass} ${DeviceRadixSort.STATUS_STAGE_NAMES[stage].replace(/_/g, ' ')}`,
      value: `${formatNumber(size)} lanes`
    }));
    const uniqueSizes = [...new Set((results.deviceradix.subgroupSizes || []).map(item => item.size))];
    html += createResultRow(
      'DeviceRadixSort',
      results.deviceradix.time,
      results.deviceradix.valid.isSorted,
      speedup,
      throughput,
      '#f472b6',
      fastest,
      {
        warning: zeroWarning,
        inline: uniqueSizes.length
          ? [{ label: 'Detected subgroup', value: `${uniqueSizes.map(size => `${formatNumber(size)} lanes`).join(', ')}` }]
          : [],
        collapsible: subgroupExtras.length
          ? {
              summary: 'Show per-pass subgroup lanes',
              hideSummary: 'Hide per-pass subgroup lanes',
              items: subgroupExtras
            }
          : null
      }
    );
  }

  if (results.onesweep) {
    const speedup = baseline / results.onesweep.time;
    const throughput = (arraySize / (results.onesweep.time / 1000)) / 1_000_000; // MKeys/s
    const zeroWarning = results.onesweep.valid.zeroCount === arraySize ? 'Output is all zeros!' : null;
    html += createResultRow(
      'OneSweep',
      results.onesweep.time,
      results.onesweep.valid.isSorted,
      speedup,
      throughput,
      '#fb923c',
      fastest,
      {
        warning: zeroWarning,
        inline: results.onesweep.subgroupSize
          ? [{
              label: 'Detected subgroup',
              value: `${formatNumber(results.onesweep.subgroupSize)} lanes${results.onesweep.shaderVariant ? ` (${results.onesweep.shaderVariant})` : ''}`
            }]
          : []
      }
    );
  }

  // Cross-Validation if multiple algorithms ran
  if (Object.keys(results).length > 1) {
    html += `<div class="border-t border-gray-700 pt-4 mt-4">`;
    html += `<p class="font-semibold mb-2">Cross-Validation (Keys & Values):</p>`;
    
    const algos = Object.keys(results);
    for (let i = 0; i < algos.length - 1; i++) {
      for (let j = i + 1; j < algos.length; j++) {
        const comparison = compareArrays(results[algos[i]].sorted, results[algos[j]].sorted);
        const icon = comparison.match ? '✓' : '✗';
        const color = comparison.match ? 'text-green-400' : 'text-red-400';
        html += `<p class="${color}">${icon} ${algos[i]} vs ${algos[j]}: ${comparison.match ? 'Match' : `${comparison.differences} differences`}</p>`;
      }
    }
    html += `</div>`;
  }

  html += `</div></div>`;
  resultsEl.innerHTML = html;
}

function createResultRow(name, time, valid, speedup, throughput, color, fastest, extra = {}) {
  const validIcon = valid ? '✓' : '✗';
  const validColor = valid ? 'text-green-400' : 'text-red-400';
  const isFastest = Math.abs(time - fastest) < 0.01;  // Check time, not speedup
  const inlineContent = (extra.inline || []).map(({ label, value }) => `<p><span class="text-gray-400">${label}:</span> ${value}</p>`).join('');

  const warningContent = extra.warning ? `<p class="text-red-400 font-bold">⚠️ ${extra.warning}</p>` : '';

  let collapsibleContent = '';
  if (extra.collapsible && Array.isArray(extra.collapsible.items) && extra.collapsible.items.length > 0) {
    const summaryLabel = extra.collapsible.summary || 'Show details';
    const hideLabel = extra.collapsible.hideSummary || 'Hide details';
    const detailItems = extra.collapsible.items
      .map(({ label, value }) => `<p><span class="text-gray-400">${label}:</span> ${value}</p>`)
      .join('');
    collapsibleContent = `
      <details class="group mt-2">
        <summary class="cursor-pointer text-blue-300 text-sm select-none">
          <span class="group-open:hidden">${summaryLabel}</span>
          <span class="hidden group-open:inline">${hideLabel}</span>
        </summary>
        <div class="mt-2 pl-3 border-l border-gray-700 space-y-1 text-sm">
          ${detailItems}
        </div>
      </details>
    `;
  }

  const extraContent = warningContent + inlineContent + collapsibleContent;
  
  return `
    <div class="mb-3 p-3 bg-gray-900 rounded-lg">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="w-2 h-2 rounded-full" style="background-color: ${color}"></div>
          <span class="font-semibold">${name}</span>
          ${isFastest ? '<span class="text-yellow-400 text-xs">★ FASTEST</span>' : ''}
        </div>
        <span class="${validColor}">${validIcon} ${valid ? 'Valid' : 'Invalid'}</span>
      </div>
      <div class="ml-5 mt-2 space-y-1 text-sm">
        <p><span class="text-gray-400">Time:</span> ${formatTime(time)}</p>
        <p><span class="text-gray-400">Speedup:</span> ${speedup.toFixed(2)}×</p>
        <p><span class="text-gray-400">Throughput:</span> ${throughput.toFixed(2)} MKeys/s</p>
        ${extraContent}
      </div>
    </div>
  `;
}

// Start the application
init();
