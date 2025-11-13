/**
 * OneSweep - WebGPU Implementation
 * Based on Thomas Smith's GPUSorting library
 */
import shader from '../shaders/onesweep/OneSweep.wgsl?raw';

export class OneSweep {
  static SORT_PASSES = 4;
  static BLOCK_DIM = 256;
  static RADIX = 256;
  static RADIX_LOG = 8;
  static KEYS_PER_THREAD = 15;
  static PART_SIZE = OneSweep.BLOCK_DIM * OneSweep.KEYS_PER_THREAD;
  static REDUCE_BLOCK_DIM = 128;
  static REDUCE_KEYS_PER_THREAD = 30;
  static REDUCE_PART_SIZE = OneSweep.REDUCE_BLOCK_DIM * OneSweep.REDUCE_KEYS_PER_THREAD;
  static STATUS_ERROR_COUNT = 3; // Keep in sync with STATUS_ERR_* constants in the shader
  static STATUS_LENGTH = OneSweep.STATUS_ERROR_COUNT;

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
    const shaderModule = this.device.createShaderModule({
      label: 'OneSweep Shader',
      code: shader
    });

    // Check for compilation errors
    const compilationInfo = await shaderModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
      console.error('OneSweep shader compilation messages:', compilationInfo.messages);
      for (const msg of compilationInfo.messages) {
        if (msg.type === 'error') {
          throw new Error(`Shader compilation error: ${msg.message} at line ${msg.lineNum}`);
        }
      }
    }

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
      globalHist: this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'global_hist' }
      }),
      scan: this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'onesweep_scan' }
      }),
      pass: this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'onesweep_pass' }
      })
    };

    this.createBuffers();

    if (this.timingSupported) {
      this.createTimingResources();
    }
  }

  createBuffers() {
    // Destroy variable-size buffers if present to prevent memory leaks
    this.sortBuffer?.destroy();
    this.altBuffer?.destroy();
    this.payloadBuffer?.destroy();
    this.altPayloadBuffer?.destroy();
    this.passHistBuffer?.destroy();

    const keySize = Math.max(16, this.maxKeys * 4); // Minimum 16 bytes
    const threadBlocks = Math.ceil(this.maxKeys / OneSweep.PART_SIZE);

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
      size: (OneSweep.SORT_PASSES + 1) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.histBuffer = this.device.createBuffer({
      size: OneSweep.RADIX * OneSweep.SORT_PASSES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.passHistBuffer = this.device.createBuffer({
      size: threadBlocks * OneSweep.RADIX * OneSweep.SORT_PASSES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.statusBuffer = this.device.createBuffer({
      size: OneSweep.STATUS_LENGTH * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    this.infoBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Upload buffer for info data
    this.infoUploadBuffer = this.device.createBuffer({
      size: 16 * OneSweep.SORT_PASSES,
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

    // Ensure capacity for larger sorts - reallocate buffers if needed
    if (numKeys > this.maxKeys) {
      this.maxKeys = numKeys;
      this.createBuffers(); // Rebuild passHistBuffer sized for new threadBlocks
    }

    const threadBlocks = Math.ceil(numKeys / OneSweep.PART_SIZE);

    // Upload data
    this.device.queue.writeBuffer(this.sortBuffer, 0, keys);
    this.device.queue.writeBuffer(this.payloadBuffer, 0, values);

    // Clear buffers
    const zeros = new Uint32Array(OneSweep.RADIX * OneSweep.SORT_PASSES).fill(0);
    this.device.queue.writeBuffer(this.histBuffer, 0, zeros);
    this.device.queue.writeBuffer(this.bumpBuffer, 0, new Uint32Array(OneSweep.SORT_PASSES + 1).fill(0));
    this.device.queue.writeBuffer(
      this.statusBuffer,
      0,
      new Uint32Array(OneSweep.STATUS_LENGTH).fill(0)
    );
    
    // Initialize pass_hist position 0 for each pass with FLAG_INCLUSIVE (value 0)
    const FLAG_INCLUSIVE = 2;
    const passHistInit = new Uint32Array(threadBlocks * OneSweep.RADIX * OneSweep.SORT_PASSES).fill(0);
    for (let pass = 0; pass < OneSweep.SORT_PASSES; pass++) {
      const passOffset = pass * threadBlocks * OneSweep.RADIX;
      for (let bin = 0; bin < OneSweep.RADIX; bin++) {
        passHistInit[passOffset + bin] = FLAG_INCLUSIVE; // (0 << 2) | FLAG_INCLUSIVE = 2
      }
    }
    this.device.queue.writeBuffer(this.passHistBuffer, 0, passHistInit);

    const encoder = this.device.createCommandEncoder();

    // Execute OneSweep passes
    for (let pass = 0; pass < OneSweep.SORT_PASSES; pass++) {
      const shift = pass * OneSweep.RADIX_LOG;
      
      // Update info buffer
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
          { binding: 2, resource: { buffer: pass % 2 === 0 ? this.sortBuffer : this.altBuffer } },
          { binding: 3, resource: { buffer: pass % 2 === 0 ? this.altBuffer : this.sortBuffer } },
          { binding: 4, resource: { buffer: pass % 2 === 0 ? this.payloadBuffer : this.altPayloadBuffer } },
          { binding: 5, resource: { buffer: pass % 2 === 0 ? this.altPayloadBuffer : this.payloadBuffer } },
          { binding: 6, resource: { buffer: this.histBuffer } },
          { binding: 7, resource: { buffer: this.passHistBuffer } },
          { binding: 8, resource: { buffer: this.statusBuffer } }
        ]
      });

      // Global histogram
      if (pass === 0) {
        const globalHistPassDesc = this.timingSupported ? {
          timestampWrites: {
            querySet: this.querySet,
            beginningOfPassWriteIndex: 0
          }
        } : {};
        const globalHistPass = encoder.beginComputePass(globalHistPassDesc);
        globalHistPass.setPipeline(this.pipelines.globalHist);
        globalHistPass.setBindGroup(0, bindGroup);
        const globalHistThreadBlocks = Math.ceil(numKeys / OneSweep.REDUCE_PART_SIZE);
        globalHistPass.dispatchWorkgroups(globalHistThreadBlocks);
        globalHistPass.end();
      }

      // Scan
      const scanPass = encoder.beginComputePass();
      scanPass.setPipeline(this.pipelines.scan);
      scanPass.setBindGroup(0, bindGroup);
      scanPass.dispatchWorkgroups(1);
      scanPass.end();

      // OneSweep pass
      const sweepPassDesc = this.timingSupported && pass === OneSweep.SORT_PASSES - 1 ? {
        timestampWrites: {
          querySet: this.querySet,
          endOfPassWriteIndex: 1
        }
      } : {};
      const sweepPass = encoder.beginComputePass(sweepPassDesc);
      sweepPass.setPipeline(this.pipelines.pass);
      sweepPass.setBindGroup(0, bindGroup);
      sweepPass.dispatchWorkgroups(threadBlocks);
      sweepPass.end();
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
      gpuTime = Number(times[1] - times[0]) / 1000000; // Convert to milliseconds
      this.readBuffer.unmap();
    }

    // Check for errors
    const statusData = await this.downloadBuffer(
      this.statusBuffer,
      OneSweep.STATUS_LENGTH * 4
    );
    const statusArray = new Uint32Array(statusData);

    let errorCode = 0;
    for (let i = 0; i < OneSweep.STATUS_ERROR_COUNT; i++) {
      if (statusArray[i] !== 0) {
        errorCode = statusArray[i];
        break;
      }
    }
    if (errorCode !== 0) {
      const errorNames = ['global_hist', 'onesweep_scan', 'onesweep_pass'];
      const errorStage = errorCode === 0xDEAD0001 ? errorNames[0] :
                         errorCode === 0xDEAD0002 ? errorNames[1] :
                         errorCode === 0xDEAD0004 ? errorNames[2] : 'unknown';
      throw new Error(`OneSweep shader error in ${errorStage}: 0x${errorCode.toString(16)}`);
    }

    // Download results (from the final buffer)
  const finalBuffer = OneSweep.SORT_PASSES % 2 === 0 ? this.sortBuffer : this.altBuffer;
  const finalPayloadBuffer = OneSweep.SORT_PASSES % 2 === 0 ? this.payloadBuffer : this.altPayloadBuffer;
    
    const resultKeys = await this.downloadBuffer(finalBuffer, numKeys * 4);
    const resultValues = await this.downloadBuffer(finalPayloadBuffer, numKeys * 4);

    const keysArray = new Uint32Array(resultKeys);
    const valuesArray = new Uint32Array(resultValues);

    const sorted = [];
    for (let i = 0; i < numKeys; i++) {
      sorted.push({ key: keysArray[i], value: valuesArray[i] });
    }

    return { sorted, gpuTime };
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
