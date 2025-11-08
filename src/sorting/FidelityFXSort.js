/**
 * FidelityFX Radix Sort - WebGPU Implementation
 * Port of AMD FidelityFX Parallel Sort with proper GPU timing
 */

import countShader from '../shaders/fidelityfx/radix_sort_count.wgsl?raw';
import reduceShader from '../shaders/fidelityfx/radix_sort_reduce.wgsl?raw';
import scanShader from '../shaders/fidelityfx/radix_sort_scan.wgsl?raw';
import scanAddShader from '../shaders/fidelityfx/radix_sort_scan_add.wgsl?raw';
import scatterShader from '../shaders/fidelityfx/radix_sort_scatter.wgsl?raw';

export class FidelityFXSort {
  static ELEMENTS_PER_THREAD = 4;
  static THREADGROUP_SIZE = 128;
  static SORT_BITS_PER_PASS = 4;
  static SORT_BIN_COUNT = 16;
  static MAX_THREADGROUPS = 800;
  static TOTAL_PASSES = 8; // 32 bits / 4 bits per pass
  static CONSTANTS_SIZE = 32; // 8 * 4 bytes

  constructor(device, maxKeys) {
    this.device = device;
    this.maxKeys = maxKeys;
    this.pipelines = null;
    this.buffers = null;
    this.constantsBuffer = null;
    this.timingSupported = device.features.has('timestamp-query');
  }

  async init() {
    // Create shader modules
    this.shaders = {
      count: this.device.createShaderModule({ code: countShader }),
      reduce: this.device.createShaderModule({ code: reduceShader }),
      scan: this.device.createShaderModule({ code: scanShader }),
      scanAdd: this.device.createShaderModule({ code: scanAddShader }),
      scatter: this.device.createShaderModule({ code: scatterShader })
    };

    // Create pipelines
    this.pipelines = {
      count: this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: this.shaders.count, entryPoint: 'main' }
      }),
      reduce: this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: this.shaders.reduce, entryPoint: 'main' }
      }),
      scan: this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: this.shaders.scan, entryPoint: 'main' }
      }),
      scanAdd: this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: this.shaders.scanAdd, entryPoint: 'main' }
      }),
      scatter: this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: this.shaders.scatter, entryPoint: 'main' }
      })
    };

    this.createBuffers();
    
    if (this.timingSupported) {
      this.createTimingResources();
    }
  }

  createBuffers() {
    const keySize = Math.max(16, this.maxKeys * 4); // Minimum 16 bytes
    const blockSize = FidelityFXSort.ELEMENTS_PER_THREAD * FidelityFXSort.THREADGROUP_SIZE;
    const numBlocks = Math.ceil(this.maxKeys / blockSize);
    const numReducedBlocks = Math.ceil(numBlocks / blockSize);

    // Ping-pong buffers for keys and values
    this.keysBuffers = [
      this.device.createBuffer({
        size: keySize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      }),
      this.device.createBuffer({
        size: keySize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      })
    ];

    this.valuesBuffers = [
      this.device.createBuffer({
        size: keySize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      }),
      this.device.createBuffer({
        size: keySize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      })
    ];

    // Scratch buffers - need COPY_DST for initialization
    this.sumTableBuffer = this.device.createBuffer({
      size: FidelityFXSort.SORT_BIN_COUNT * numBlocks * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.reduceTableBuffer = this.device.createBuffer({
      size: FidelityFXSort.SORT_BIN_COUNT * Math.max(1, numReducedBlocks) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.scanScratchBuffer = this.device.createBuffer({
      size: FidelityFXSort.SORT_BIN_COUNT * Math.max(1, numReducedBlocks) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    // Constants buffer
    this.constantsBuffer = this.device.createBuffer({
      size: FidelityFXSort.CONSTANTS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Upload buffer for constants (one for each pass)
    this.constantsUploadBuffer = this.device.createBuffer({
      size: FidelityFXSort.CONSTANTS_SIZE * FidelityFXSort.TOTAL_PASSES,
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

  calculateConstants(numKeys, shift) {
    const blockSize = FidelityFXSort.ELEMENTS_PER_THREAD * FidelityFXSort.THREADGROUP_SIZE;
    const numBlocks = Math.ceil(numKeys / blockSize);

    let numThreadGroups = Math.min(FidelityFXSort.MAX_THREADGROUPS, numBlocks);
    let blocksPerThreadGroup = Math.floor(numBlocks / numThreadGroups);
    let numThreadGroupsWithAdditionalBlocks = numBlocks % numThreadGroups;

    if (numBlocks < numThreadGroups) {
      blocksPerThreadGroup = 1;
      numThreadGroups = numBlocks;
      numThreadGroupsWithAdditionalBlocks = 0;
    }

    const numReducedThreadGroups = FidelityFXSort.SORT_BIN_COUNT *
      (blockSize > numThreadGroups ? 1 : Math.ceil(numThreadGroups / blockSize));
    const numReduceThreadgroupPerBin = numReducedThreadGroups / FidelityFXSort.SORT_BIN_COUNT;

    return {
      numKeys,
      numBlocksPerThreadGroup: blocksPerThreadGroup,
      numThreadGroups,
      numThreadGroupsWithAdditionalBlocks,
      numReduceThreadgroupPerBin,
      numScanValues: numReducedThreadGroups,
      shift,
      padding: 0
    };
  }

  async sort(data) {
    const numKeys = data.length;
    const keys = new Uint32Array(data.map(d => d.key));
    const values = new Uint32Array(data.map(d => d.value));

    // Upload data
    this.device.queue.writeBuffer(this.keysBuffers[0], 0, keys);
    this.device.queue.writeBuffer(this.valuesBuffers[0], 0, values);

    const encoder = this.device.createCommandEncoder();
    let sourceIndex = 0;

    // Execute radix sort passes
    for (let pass = 0; pass < FidelityFXSort.TOTAL_PASSES; pass++) {
      const shift = pass * FidelityFXSort.SORT_BITS_PER_PASS;
      const isFirstPass = pass === 0;
      const isLastPass = pass === FidelityFXSort.TOTAL_PASSES - 1;
      sourceIndex = this.encodeSortPass(encoder, numKeys, shift, sourceIndex, isFirstPass, isLastPass, pass);
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

    // Download results
    const resultKeys = await this.downloadBuffer(this.keysBuffers[sourceIndex], numKeys * 4);
    const resultValues = await this.downloadBuffer(this.valuesBuffers[sourceIndex], numKeys * 4);

    const keysArray = new Uint32Array(resultKeys);
    const valuesArray = new Uint32Array(resultValues);

    const sorted = [];
    for (let i = 0; i < numKeys; i++) {
      sorted.push({ key: keysArray[i], value: valuesArray[i] });
    }

    return { sorted, gpuTime };
  }

  encodeSortPass(encoder, numKeys, shift, sourceIndex, isFirstPass = false, isLastPass = false, passIndex = 0) {
    const constants = this.calculateConstants(numKeys, shift);
    const constantsData = new Int32Array([
      constants.numKeys,
      constants.numBlocksPerThreadGroup,
      constants.numThreadGroups,
      constants.numThreadGroupsWithAdditionalBlocks,
      constants.numReduceThreadgroupPerBin,
      constants.numScanValues,
      constants.shift,
      constants.padding
    ]);

    // Write to upload buffer and copy to constants buffer via command encoder
    const constantsOffset = passIndex * FidelityFXSort.CONSTANTS_SIZE;
    this.device.queue.writeBuffer(this.constantsUploadBuffer, constantsOffset, constantsData);
    
    encoder.copyBufferToBuffer(
      this.constantsUploadBuffer,
      constantsOffset,
      this.constantsBuffer,
      0,
      FidelityFXSort.CONSTANTS_SIZE
    );

    const destIndex = 1 - sourceIndex;

    // Count pass (with optional timestamp start)
    const countPassDesc = this.timingSupported && isFirstPass ? {
      timestampWrites: {
        querySet: this.querySet,
        beginningOfPassWriteIndex: 0
      }
    } : {};
    const countPass = encoder.beginComputePass(countPassDesc);
    const countBindGroup = this.device.createBindGroup({
      layout: this.pipelines.count.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.constantsBuffer } },
        { binding: 1, resource: { buffer: this.keysBuffers[sourceIndex] } },
        { binding: 2, resource: { buffer: this.sumTableBuffer } }
      ]
    });
    countPass.setPipeline(this.pipelines.count);
    countPass.setBindGroup(0, countBindGroup);
    countPass.dispatchWorkgroups(constants.numThreadGroups);
    countPass.end();

    // Reduce pass
    const reducePass = encoder.beginComputePass();
    const reduceBindGroup = this.device.createBindGroup({
      layout: this.pipelines.reduce.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.constantsBuffer } },
        { binding: 1, resource: { buffer: this.sumTableBuffer } },
        { binding: 2, resource: { buffer: this.reduceTableBuffer } }
      ]
    });
    reducePass.setPipeline(this.pipelines.reduce);
    reducePass.setBindGroup(0, reduceBindGroup);
    reducePass.dispatchWorkgroups(constants.numScanValues);
    reducePass.end();

    // Scan pass
    const scanPass = encoder.beginComputePass();
    const scanBindGroup = this.device.createBindGroup({
      layout: this.pipelines.scan.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.constantsBuffer } },
        { binding: 1, resource: { buffer: this.reduceTableBuffer } },
        { binding: 2, resource: { buffer: this.scanScratchBuffer } }
      ]
    });
    scanPass.setPipeline(this.pipelines.scan);
    scanPass.setBindGroup(0, scanBindGroup);
    const numScanWorkgroups = Math.ceil(constants.numScanValues / 
      (FidelityFXSort.ELEMENTS_PER_THREAD * FidelityFXSort.THREADGROUP_SIZE));
    scanPass.dispatchWorkgroups(numScanWorkgroups);
    scanPass.end();

    // Scan add pass
    const scanAddPass = encoder.beginComputePass();
    const scanAddBindGroup = this.device.createBindGroup({
      layout: this.pipelines.scanAdd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.constantsBuffer } },
        { binding: 1, resource: { buffer: this.sumTableBuffer } },
        { binding: 2, resource: { buffer: this.scanScratchBuffer } }
      ]
    });
    scanAddPass.setPipeline(this.pipelines.scanAdd);
    scanAddPass.setBindGroup(0, scanAddBindGroup);
    scanAddPass.dispatchWorkgroups(constants.numScanValues);
    scanAddPass.end();

    // Scatter pass (with optional timestamp end)
    const scatterPassDesc = this.timingSupported && isLastPass ? {
      timestampWrites: {
        querySet: this.querySet,
        endOfPassWriteIndex: 1
      }
    } : {};
    const scatterPass = encoder.beginComputePass(scatterPassDesc);
    const scatterBindGroup = this.device.createBindGroup({
      layout: this.pipelines.scatter.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.constantsBuffer } },
        { binding: 1, resource: { buffer: this.keysBuffers[sourceIndex] } },
        { binding: 2, resource: { buffer: this.keysBuffers[destIndex] } },
        { binding: 3, resource: { buffer: this.sumTableBuffer } },
        { binding: 4, resource: { buffer: this.valuesBuffers[sourceIndex] } },
        { binding: 5, resource: { buffer: this.valuesBuffers[destIndex] } }
      ]
    });
    scatterPass.setPipeline(this.pipelines.scatter);
    scatterPass.setBindGroup(0, scatterBindGroup);
    scatterPass.dispatchWorkgroups(constants.numThreadGroups);
    scatterPass.end();

    return destIndex;
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
    this.keysBuffers?.forEach(b => b.destroy());
    this.valuesBuffers?.forEach(b => b.destroy());
    this.sumTableBuffer?.destroy();
    this.reduceTableBuffer?.destroy();
    this.scanScratchBuffer?.destroy();
    this.constantsBuffer?.destroy();
    this.constantsUploadBuffer?.destroy();
    this.querySet?.destroy();
    this.queryBuffer?.destroy();
    this.readBuffer?.destroy();
  }
}
