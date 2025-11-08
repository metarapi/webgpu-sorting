/**
 * Utility functions for WebGPU sorting comparison
 */

/**
 * Generate random test data for sorting
 * @param {number} count - Number of elements to generate
 * @param {number} seed - Optional seed for reproducible data
 * @returns {Array} Array of {key, value} pairs
 */
export function generateTestData(count, seed = Math.random() * 0xFFFFFFFF) {
  const data = [];
  let rng = seed;
  
  for (let i = 0; i < count; i++) {
    // Simple LCG random number generator for reproducible results
    rng = (rng * 1664525 + 1013904223) % 0x100000000;
    const key = rng >>> 0; // Ensure unsigned 32-bit
    const value = i; // Original index as payload
    
    data.push({ key, value });
  }
  
  return data;
}

/**
 * Validate that array is properly sorted
 * @param {Array} data - Array of {key, value} pairs to validate
 * @returns {{isSorted: boolean, errors: number, firstError: number}} Validation results
 */
export function validateSort(data) {
  let errors = 0;
  let firstError = -1;
  
  for (let i = 1; i < data.length; i++) {
    if (data[i - 1].key > data[i].key) {
      errors++;
      if (firstError === -1) {
        firstError = i;
      }
    }
  }
  
  return {
    isSorted: errors === 0,
    errors,
    firstError
  };
}

/**
 * Compare two sorted arrays for equality
 * @param {Array} arr1 - First sorted array
 * @param {Array} arr2 - Second sorted array  
 * @returns {{match: boolean, differences: number}} Comparison results
 */
export function compareArrays(arr1, arr2) {
  if (arr1.length !== arr2.length) {
    return { match: false, differences: Math.abs(arr1.length - arr2.length) };
  }
  
  let differences = 0;
  
  for (let i = 0; i < arr1.length; i++) {
    if (arr1[i].key !== arr2[i].key) {
      differences++;
    }
  }
  
  return {
    match: differences === 0,
    differences
  };
}

/**
 * Format time in appropriate units
 * @param {number} milliseconds - Time in milliseconds
 * @returns {string} Formatted time string
 */
export function formatTime(milliseconds) {
  if (milliseconds < 1) {
    return `${(milliseconds * 1000).toFixed(2)} μs`;
  } else if (milliseconds < 1000) {
    return `${milliseconds.toFixed(2)} ms`;
  } else {
    return `${(milliseconds / 1000).toFixed(2)} s`;
  }
}

/**
 * Format number with thousands separators
 * @param {number} num - Number to format
 * @returns {string} Formatted number
 */
export function formatNumber(num) {
  return num.toLocaleString();
}

/**
 * Create a GPU buffer and upload data
 * @param {GPUDevice} device - WebGPU device
 * @param {ArrayBuffer|TypedArray} data - Data to upload
 * @param {GPUBufferUsageFlags} usage - Buffer usage flags
 * @param {string} label - Debug label
 * @returns {GPUBuffer} Created buffer
 */
export function createBufferWithData(device, data, usage, label) {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false
  });
  
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

/**
 * Read data from GPU buffer
 * @param {GPUDevice} device - WebGPU device
 * @param {GPUBuffer} buffer - Buffer to read
 * @param {number} size - Size in bytes
 * @returns {Promise<ArrayBuffer>} Buffer contents
 */
export async function readBuffer(device, buffer, size) {
  const stagingBuffer = device.createBuffer({
    label: 'Staging Buffer',
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
  device.queue.submit([commandEncoder.finish()]);
  
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = stagingBuffer.getMappedRange().slice(0);
  stagingBuffer.unmap();
  stagingBuffer.destroy();
  
  return arrayBuffer;
}

/**
 * Load shader from file
 * @param {string} path - Path to shader file (relative to project root)
 * @returns {Promise<string>} Shader source code
 */
export async function loadShader(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load shader: ${path}`);
  }
  return await response.text();
}
