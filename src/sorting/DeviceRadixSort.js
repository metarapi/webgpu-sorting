/**
 * DeviceRadixSort - WebGPU Implementation
 * Based on Thomas Smith's GPUSorting library
 */

import shader from '../shaders/deviceradix/DeviceRadixSort.wgsl?raw';

export class DeviceRadixSort {
  static SORT_PASSES = 4;
  static BLOCK_DIM = 256;
  static RADIX = 256;
  static RADIX_LOG = 8;
  static KEYS_PER_THREAD = 15;
  static PART_SIZE = DeviceRadixSort.BLOCK_DIM * DeviceRadixSort.KEYS_PER_THREAD;
  static REDUCE_BLOCK_DIM = 128;
  static REDUCE_KEYS_PER_THREAD = 30;
  static REDUCE_PART_SIZE = DeviceRadixSort.REDUCE_BLOCK_DIM * DeviceRadixSort.REDUCE_KEYS_PER_THREAD;
  static STATUS_ERROR_COUNT = 3; // Keep in sync with STATUS_ERR_* constants in the shader
  static STATUS_LENGTH = DeviceRadixSort.STATUS_ERROR_COUNT + DeviceRadixSort.SORT_PASSES;

  constructor(device, maxKeys) {
    this.device = device;
    this.maxKeys = maxKeys;
    this.pipelines = null;
    this.buffers = null;
    this.bindGroupLayout = null;
    this.timingSupported = device.features.has('timestamp-query');
  }

  async init() {
    // Create shader module
    const shaderModule = this.device.createShaderModule({ code: shader });

    // Create bind group layout
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout]
    });

    // Create pipelines
    this.pipelines = {
      reduceHist: this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'reduce_hist' }
      }),
      scan: this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'scan' }
      }),
      dvrPass: this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'dvr_pass' }
      })
    };

    this.createBuffers();

    if (this.timingSupported) {
      this.createTimingResources();
    }
  }

  createBuffers() {
    const keySize = Math.max(16, this.maxKeys * 4); // Minimum 16 bytes
    const threadBlocks = Math.ceil(this.maxKeys / DeviceRadixSort.PART_SIZE);

    this.sortBuffer = this.device.createBuffer({
      size: keySize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    this.altBuffer = this.device.createBuffer({
      size: keySize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    this.payloadBuffer = this.device.createBuffer({
      size: keySize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    this.altPayloadBuffer = this.device.createBuffer({
      size: keySize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.bumpBuffer = this.device.createBuffer({
      size: (DeviceRadixSort.SORT_PASSES + 1) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.histBuffer = this.device.createBuffer({
      size: DeviceRadixSort.RADIX * DeviceRadixSort.SORT_PASSES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.passHistBuffer = this.device.createBuffer({
      size: threadBlocks * DeviceRadixSort.RADIX * DeviceRadixSort.SORT_PASSES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.statusBuffer = this.device.createBuffer({
      size: DeviceRadixSort.STATUS_LENGTH * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    this.infoBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Upload buffer for info data
    this.infoUploadBuffer = this.device.createBuffer({
      size: 16 * DeviceRadixSort.SORT_PASSES,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
  }

  createTimingResources() {
    this.querySet = this.device.createQuerySet({
      type: 'timestamp',
      count: 2
    });

    this.queryBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
    });

    this.readBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
  }

  async sort(data) {
    const numKeys = data.length;
    const keys = new Uint32Array(data.map(d => d.key));
    const values = new Uint32Array(data.map(d => d.value));
    const threadBlocks = Math.ceil(numKeys / DeviceRadixSort.PART_SIZE);

    // Upload data
    this.device.queue.writeBuffer(this.sortBuffer, 0, keys);
    this.device.queue.writeBuffer(this.payloadBuffer, 0, values);

    // Clear buffers
    const zeros = new Uint32Array(DeviceRadixSort.RADIX * DeviceRadixSort.SORT_PASSES).fill(0);
    this.device.queue.writeBuffer(this.histBuffer, 0, zeros);
    this.device.queue.writeBuffer(
      this.statusBuffer,
      0,
      new Uint32Array(DeviceRadixSort.STATUS_LENGTH).fill(0)
    );

    const encoder = this.device.createCommandEncoder();

    // Execute sort passes
    for (let pass = 0; pass < DeviceRadixSort.SORT_PASSES; pass++) {
      const shift = pass * DeviceRadixSort.RADIX_LOG;
      
      // Update info buffer via upload buffer
      const infoData = new Uint32Array([numKeys, shift, threadBlocks, 0]);
      const infoOffset = pass * 16;
      this.device.queue.writeBuffer(this.infoUploadBuffer, infoOffset, infoData);
      
      encoder.copyBufferToBuffer(
        this.infoUploadBuffer,
        infoOffset,
        this.infoBuffer,
        0,
        16
      );

      // Create bind group
      const bindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.infoBuffer } },
          { binding: 1, resource: { buffer: this.bumpBuffer } },
          { binding: 2, resource: { buffer: this.sortBuffer } },
          { binding: 3, resource: { buffer: this.altBuffer } },
          { binding: 4, resource: { buffer: this.payloadBuffer } },
          { binding: 5, resource: { buffer: this.altPayloadBuffer } },
          { binding: 6, resource: { buffer: this.histBuffer } },
          { binding: 7, resource: { buffer: this.passHistBuffer } },
          { binding: 8, resource: { buffer: this.statusBuffer } }
        ]
      });

      // Reduce histogram
      const reducePassDesc = this.timingSupported && pass === 0 ? {
        timestampWrites: {
          querySet: this.querySet,
          beginningOfPassWriteIndex: 0
        }
      } : {};
      const reducePass = encoder.beginComputePass(reducePassDesc);
      reducePass.setPipeline(this.pipelines.reduceHist);
      reducePass.setBindGroup(0, bindGroup);
      reducePass.dispatchWorkgroups(threadBlocks);
      reducePass.end();

      // Scan
      const scanPass = encoder.beginComputePass();
      scanPass.setPipeline(this.pipelines.scan);
      scanPass.setBindGroup(0, bindGroup);
      scanPass.dispatchWorkgroups(DeviceRadixSort.RADIX);
      scanPass.end();

      // DVR pass
      const dvrPassDesc = this.timingSupported && pass === DeviceRadixSort.SORT_PASSES - 1 ? {
        timestampWrites: {
          querySet: this.querySet,
          endOfPassWriteIndex: 1
        }
      } : {};
      const dvrPass = encoder.beginComputePass(dvrPassDesc);
      dvrPass.setPipeline(this.pipelines.dvrPass);
      dvrPass.setBindGroup(0, bindGroup);
      dvrPass.dispatchWorkgroups(threadBlocks);
      dvrPass.end();

      // Swap buffers for next pass
      [this.sortBuffer, this.altBuffer] = [this.altBuffer, this.sortBuffer];
      [this.payloadBuffer, this.altPayloadBuffer] = [this.altPayloadBuffer, this.payloadBuffer];
    }

    if (this.timingSupported) {
      encoder.resolveQuerySet(this.querySet, 0, 2, this.queryBuffer, 0);
      encoder.copyBufferToBuffer(this.queryBuffer, 0, this.readBuffer, 0, 16);
    }

    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    // Read timing
    let gpuTime = 0;
    if (this.timingSupported) {
      await this.readBuffer.mapAsync(GPUMapMode.READ);
      const times = new BigUint64Array(this.readBuffer.getMappedRange());
      const delta = Number(times[1] - times[0]);
      gpuTime = delta / 1_000_000; // Convert to milliseconds
      this.readBuffer.unmap();
    }

    // Check for errors emitted by compute passes and gather stats
    const statusData = await this.downloadBuffer(
      this.statusBuffer,
      DeviceRadixSort.STATUS_LENGTH * 4
    );
    const statusArray = new Uint32Array(statusData);

    let errorCode = 0;
    for (let i = 0; i < DeviceRadixSort.STATUS_ERROR_COUNT; i++) {
      if (statusArray[i] !== 0) {
        errorCode = statusArray[i];
        break;
      }
    }
    if (errorCode !== 0) {
      const errorMessages = {
        0xDEAD0001: 'reduce_hist: subgroup size < MIN_SUBGROUP_SIZE or alignment issue',
        0xDEAD0002: 'scan: subgroup size < MIN_SUBGROUP_SIZE or alignment issue',
        0xDEAD0004: 'dvr_pass: warp hist capacity exceeded'
      };
      throw new Error(`GPU Sort Error: ${errorMessages[errorCode] || `Unknown error code 0x${errorCode.toString(16)}`}`);
    }

    // Download results
    const resultKeys = await this.downloadBuffer(this.sortBuffer, numKeys * 4);
    const resultValues = await this.downloadBuffer(this.payloadBuffer, numKeys * 4);

    const keysArray = new Uint32Array(resultKeys);
    const valuesArray = new Uint32Array(resultValues);

    let subgroupSize = 0;
    for (let i = DeviceRadixSort.STATUS_ERROR_COUNT; i < statusArray.length; i++) {
      if (statusArray[i] !== 0) {
        subgroupSize = statusArray[i];
        break;
      }
    }

    const sorted = [];
    for (let i = 0; i < numKeys; i++) {
      sorted.push({ key: keysArray[i], value: valuesArray[i] });
    }

    return { sorted, gpuTime, subgroupSize };
  }

  async downloadBuffer(buffer, size) {
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const result = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();

    return result;
  }

  destroy() {
    this.sortBuffer?.destroy();
    this.altBuffer?.destroy();
    this.payloadBuffer?.destroy();
    this.altPayloadBuffer?.destroy();
    this.bumpBuffer?.destroy();
    this.histBuffer?.destroy();
    this.passHistBuffer?.destroy();
    this.statusBuffer?.destroy();
    this.infoBuffer?.destroy();
    this.infoUploadBuffer?.destroy();
    this.querySet?.destroy();
    this.queryBuffer?.destroy();
    this.readBuffer?.destroy();
  }
}
