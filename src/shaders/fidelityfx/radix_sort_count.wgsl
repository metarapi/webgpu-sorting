// WebGPU/WGSL port of AMD FidelityFX Parallel Sort - Count Pass
// Based on parallelsort_sum_pass.hlsl (which calls FfxParallelSortCount)

enable subgroups;

const ELEMENTS_PER_THREAD: u32 = 4u;
const THREADGROUP_SIZE: u32 = 128u;
const SORT_BITS_PER_PASS: u32 = 4u;
const SORT_BIN_COUNT: u32 = 16u;

struct RadixSortConstants {
    numKeys: u32,
    numBlocksPerThreadGroup: i32,
    numThreadGroups: u32,
    numThreadGroupsWithAdditionalBlocks: u32,
    numReduceThreadgroupPerBin: u32,
    numScanValues: u32,
    shift: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> constants: RadixSortConstants;
@group(0) @binding(1) var<storage, read> sourceKeys: array<u32>;
@group(0) @binding(2) var<storage, read_write> sumTable: array<u32>;

var<workgroup> histogram: array<atomic<u32>, THREADGROUP_SIZE * SORT_BIN_COUNT>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(workgroup_id) groupId: vec3<u32>
) {
    let localID = localId.x;
    let groupID = groupId.x;
    
    for (var i = 0u; i < SORT_BIN_COUNT; i++) {
        atomicStore(&histogram[(i * THREADGROUP_SIZE) + localID], 0u);
    }
    
    workgroupBarrier();
    
    let blockSize = ELEMENTS_PER_THREAD * THREADGROUP_SIZE;
    var threadgroupBlockStart = blockSize * u32(constants.numBlocksPerThreadGroup) * groupID;
    var numBlocksToProcess = u32(constants.numBlocksPerThreadGroup);
    
    if (groupID >= constants.numThreadGroups - constants.numThreadGroupsWithAdditionalBlocks) {
        threadgroupBlockStart += (groupID - (constants.numThreadGroups - constants.numThreadGroupsWithAdditionalBlocks)) * blockSize;
        numBlocksToProcess++;
    }
    
    var blockIndex = threadgroupBlockStart + localID;
    
    for (var blockCount = 0u; blockCount < numBlocksToProcess; blockCount++) {
        var dataIndex = blockIndex;
        
        var srcKeys: array<u32, ELEMENTS_PER_THREAD>;
        srcKeys[0] = select(0xFFFFFFFFu, sourceKeys[dataIndex], dataIndex < constants.numKeys);
        srcKeys[1] = select(0xFFFFFFFFu, sourceKeys[dataIndex + THREADGROUP_SIZE], 
                           dataIndex + THREADGROUP_SIZE < constants.numKeys);
        srcKeys[2] = select(0xFFFFFFFFu, sourceKeys[dataIndex + THREADGROUP_SIZE * 2u], 
                           dataIndex + THREADGROUP_SIZE * 2u < constants.numKeys);
        srcKeys[3] = select(0xFFFFFFFFu, sourceKeys[dataIndex + THREADGROUP_SIZE * 3u], 
                           dataIndex + THREADGROUP_SIZE * 3u < constants.numKeys);
        
        for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
            if (dataIndex < constants.numKeys) {
                let localKey = (srcKeys[i] >> constants.shift) & 0xFu;
                atomicAdd(&histogram[(localKey * THREADGROUP_SIZE) + localID], 1u);
                dataIndex += THREADGROUP_SIZE;
            }
        }
        
        blockIndex += blockSize;
    }
    
    workgroupBarrier();
    
    if (localID < SORT_BIN_COUNT) {
        var sum = 0u;
        for (var i = 0u; i < THREADGROUP_SIZE; i++) {
            sum += atomicLoad(&histogram[localID * THREADGROUP_SIZE + i]);
        }
        sumTable[localID * constants.numThreadGroups + groupID] = sum;
    }
}