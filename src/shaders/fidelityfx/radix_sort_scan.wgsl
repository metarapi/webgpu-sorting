// WebGPU/WGSL port of AMD FidelityFX Parallel Sort - Scan Pass  
// Based on parallelsort_scan_pass.hlsl (which calls FfxParallelSortScan)

enable subgroups;

// Constants matching AMD's implementation
const ELEMENTS_PER_THREAD: u32 = 4u;
const THREADGROUP_SIZE: u32 = 128u;

// Constant buffer data structure
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

// Uniform bindings
@group(0) @binding(0) var<uniform> constants: RadixSortConstants;

// Buffer bindings for scan pass
@group(0) @binding(1) var<storage, read> scanSource: array<u32>;
@group(0) @binding(2) var<storage, read_write> scanDest: array<u32>;

// Workgroup shared memory for coalesced access transformation
var<workgroup> ldsData: array<array<u32, THREADGROUP_SIZE>, ELEMENTS_PER_THREAD>;
var<workgroup> subgroupSums: array<u32, THREADGROUP_SIZE>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(workgroup_id) groupId: vec3<u32>,
    @builtin(subgroup_size) subgroupSize: u32,
    @builtin(subgroup_invocation_id) subgroupId: u32
) {
    let localID = localId.x;
    let groupID = groupId.x;
    
    let numValuesToScan = constants.numScanValues;
    let baseIndex = ELEMENTS_PER_THREAD * THREADGROUP_SIZE * groupID;
    
    // Perform coalesced loads into workgroup memory
    // This transforms uncoalesced loads into coalesced loads and then scattered loads from LDS
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let dataIndex = baseIndex + (i * THREADGROUP_SIZE) + localID;
        
        let col = ((i * THREADGROUP_SIZE) + localID) / ELEMENTS_PER_THREAD;
        let row = ((i * THREADGROUP_SIZE) + localID) % ELEMENTS_PER_THREAD;
        
        ldsData[row][col] = select(0u, scanSource[dataIndex], dataIndex < numValuesToScan);
    }
    
    workgroupBarrier();
    
    // Calculate the local scan-prefix for current thread
    var threadgroupSum = 0u;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let tmp = ldsData[i][localID];
        ldsData[i][localID] = threadgroupSum;
        threadgroupSum += tmp;
    }
    
    // Perform block scan-prefix using subgroup operations
    let subgroupPrefixed = subgroupExclusiveAdd(threadgroupSum);
    
    // Calculate subgroup ID
    let subgroupID = localID / subgroupSize;
    let laneID = subgroupId;
    
    // Last thread in each subgroup writes partial sum to workgroup memory
    if (laneID == subgroupSize - 1u) {
        subgroupSums[subgroupID] = subgroupPrefixed + threadgroupSum;
    }
    
    workgroupBarrier();
    
    // FIX: All threads participate in scanning subgroup sums uniformly
    let numSubgroups = THREADGROUP_SIZE / subgroupSize;
    var valueToScan = 0u;
    
    // Only the first 'numSubgroups' threads have meaningful data to scan
    if (localID < numSubgroups) {
        valueToScan = subgroupSums[localID];
    }
    
    // All threads participate in this subgroup operation (uniform control flow)
    let scannedValue = subgroupExclusiveAdd(valueToScan);
    
    // Write back the scanned values (only meaningful for first numSubgroups threads)
    if (localID < numSubgroups) {
        subgroupSums[localID] = scannedValue;
    }
    
    workgroupBarrier();
    
    // Add the subgroup partial sums back to each thread's prefix sum
    let finalPrefixSum = subgroupPrefixed + subgroupSums[subgroupID];
    
    // Add the block scanned-prefixes back to workgroup memory
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        ldsData[i][localID] += finalPrefixSum;
    }
    
    workgroupBarrier();
    
    // Perform coalesced writes to scan destination
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let dataIndex = baseIndex + (i * THREADGROUP_SIZE) + localID;
        
        let col = ((i * THREADGROUP_SIZE) + localID) / ELEMENTS_PER_THREAD;
        let row = ((i * THREADGROUP_SIZE) + localID) % ELEMENTS_PER_THREAD;
        
        if (dataIndex < numValuesToScan) {
            scanDest[dataIndex] = ldsData[row][col];
        }
    }
}
