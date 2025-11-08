import './style.css';
import { FidelityFXSort } from './sorting/FidelityFXSort.js';
import { DeviceRadixSort } from './sorting/DeviceRadixSort.js';
import { generateTestData, validateSort, compareArrays, formatTime, formatNumber } from './utils.js';

// WebGPU device and context
let device = null;
let fidelityFXSort = null;
let deviceRadixSort = null;

// Initialize the application
async function init() {
  const app = document.getElementById('app');
  
  app.innerHTML = `
    <div class="container mx-auto px-4 py-8">
      <header class="mb-8">
        <h1 class="text-4xl font-bold mb-2">WebGPU Sorting Algorithms Comparison</h1>
        <p class="text-gray-400">3-way comparison: FidelityFX vs DeviceRadixSort vs JavaScript</p>
      </header>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <!-- Algorithm Selection -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 class="text-xl font-semibold mb-4">Test Mode</h2>
          <select id="algorithm-select" class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500">
            <option value="all">All Algorithms (3-way)</option>
            <option value="fidelityfx">FidelityFX Only</option>
            <option value="deviceradix">DeviceRadixSort Only</option>
            <option value="javascript">JavaScript Only</option>
          </select>
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
          <p>Click "Run Comparison" to begin...</p>
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

    // Request device with necessary features
    const features = ['subgroups'];
    if (adapter.features.has('timestamp-query')) {
      features.push('timestamp-query');
    }

    device = await adapter.requestDevice({
      requiredFeatures: features
    });

    // Initialize sorting algorithms
    statusEl.innerHTML = '<p class="text-sm text-blue-400">Initializing sorting algorithms...</p>';
    
    const maxKeys = 10000000; // 10M elements max
    fidelityFXSort = new FidelityFXSort(device, maxKeys);
    await fidelityFXSort.init();
    
    deviceRadixSort = new DeviceRadixSort(device, maxKeys);
    await deviceRadixSort.init();

    statusEl.innerHTML = `<p class="text-sm text-green-400">✓ WebGPU initialized with ${features.join(', ')}</p>`;
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
      const { sorted, gpuTime } = await deviceRadixSort.sort(data);
      results.deviceradix = {
        time: gpuTime,
        sorted,
        valid: validateSort(sorted)
      };
    }

    // Display results
    displayResults(results, arraySize);
  } catch (error) {
    resultsEl.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
    console.error(error);
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
    html += createResultRow(
      'JavaScript Array.sort',
      results.javascript.time,
      results.javascript.valid.isSorted,
      speedup,
      '#60a5fa',
      fastest
    );
  }

  if (results.fidelityfx) {
    const speedup = baseline / results.fidelityfx.time;
    html += createResultRow(
      'FidelityFX Radix Sort',
      results.fidelityfx.time,
      results.fidelityfx.valid.isSorted,
      speedup,
      '#4ade80',
      fastest
    );
  }

  if (results.deviceradix) {
    const speedup = baseline / results.deviceradix.time;
    html += createResultRow(
      'DeviceRadixSort',
      results.deviceradix.time,
      results.deviceradix.valid.isSorted,
      speedup,
      '#f472b6',
      fastest
    );
  }

  // Cross-validation if multiple algorithms ran
  if (Object.keys(results).length > 1) {
    html += `<div class="border-t border-gray-700 pt-4 mt-4">`;
    html += `<p class="font-semibold mb-2">Cross-Validation:</p>`;
    
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

function createResultRow(name, time, valid, speedup, color, fastest) {
  const validIcon = valid ? '✓' : '✗';
  const validColor = valid ? 'text-green-400' : 'text-red-400';
  const isFastest = Math.abs(time - fastest) < 0.01;  // Check time, not speedup
  
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
      </div>
    </div>
  `;
}

// Start the application
init();
