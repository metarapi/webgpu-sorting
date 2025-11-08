// WebGPU/WGSL port of AMD FidelityFX Parallel Sort - Reduce Pass
// Based on parallelsort_reduce_pass.hlsl (which calls FfxParallelSortReduce)

enable subgroups;

const ELEMENTS_PER_THREAD: u32 = 4u;
const THREADGROUP_SIZE: u32 = 128u;

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
@group(0) @binding(1) var<storage, read> sumTable: array<u32>;
@group(0) @binding(2) var<storage, read_write> reduceTable: array<u32>;

var<workgroup> subgroupSums: array<u32, THREADGROUP_SIZE>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(workgroup_id) groupId: vec3<u32>,
    @builtin(subgroup_size) subgroupSize: u32
) {
    let localID = localId.x;
    let groupID = groupId.x;
    
    let binID = groupID / constants.numReduceThreadgroupPerBin;
    let binOffset = binID * constants.numThreadGroups;
    let baseIndex = (groupID % constants.numReduceThreadgroupPerBin) * ELEMENTS_PER_THREAD * THREADGROUP_SIZE;
    
    var threadgroupSum = 0u;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let dataIndex = baseIndex + (i * THREADGROUP_SIZE) + localID;
        if (dataIndex < constants.numThreadGroups) {
            threadgroupSum += sumTable[binOffset + dataIndex];
        }
    }
    
    let subgroupReduced = subgroupAdd(threadgroupSum);
    let subgroupID = localID / subgroupSize;
    
    if (subgroupElect()) {
        subgroupSums[subgroupID] = subgroupReduced;
    }
    
    workgroupBarrier();
    
    // Compute the final sum using uniform control flow
    let numSubgroups = (THREADGROUP_SIZE + subgroupSize - 1u) / subgroupSize;
    let subgroupValue = select(0u, subgroupSums[localID], localID < numSubgroups);
    let finalSum = subgroupAdd(subgroupValue);
    
    if (localID == 0u) {
        reduceTable[groupID] = finalSum;
    }
}
