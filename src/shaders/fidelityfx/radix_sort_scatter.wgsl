// WebGPU/WGSL port of AMD FidelityFX Parallel Sort - Scatter Pass
// Based on parallelsort_scatter_pass.hlsl (FfxParallelSortScatter)

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
@group(0) @binding(2) var<storage, read_write> destKeys: array<u32>;
@group(0) @binding(3) var<storage, read> sumTable: array<u32>;
@group(0) @binding(4) var<storage, read> sourceValues: array<u32>;
@group(0) @binding(5) var<storage, read_write> destValues: array<u32>;

// AMD's algorithm uses these LDS arrays
var<workgroup> binOffsetCache: array<u32, SORT_BIN_COUNT>;
var<workgroup> localHistogram: array<atomic<u32>, SORT_BIN_COUNT>;
var<workgroup> ldsScratch: array<u32, THREADGROUP_SIZE>;
var<workgroup> ldsSums: array<u32, THREADGROUP_SIZE>;

// Prefix scan within workgroup (without subgroups for simplicity)
fn workgroupPrefixScan(localSum: u32, localID: u32) -> u32 {
    ldsSums[localID] = localSum;
    workgroupBarrier();
    
    // Simple prefix scan using standard doubling approach
    var step = 1u;
    while (step < THREADGROUP_SIZE) {
        let temp = select(0u, ldsSums[localID - step], localID >= step);
        workgroupBarrier();
        ldsSums[localID] += temp;
        workgroupBarrier();
        step *= 2u;
    }
    
    // Convert to exclusive scan
    let result = select(0u, ldsSums[localID - 1u], localID > 0u);
    workgroupBarrier();
    return result;
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(workgroup_id) groupId: vec3<u32>
) {
    let localID = localId.x;
    let groupID = groupId.x;
    
    // Load the sort bin threadgroup offsets into LDS for faster referencing
    if (localID < SORT_BIN_COUNT) {
        binOffsetCache[localID] = sumTable[localID * constants.numThreadGroups + groupID];
    }
    
    workgroupBarrier();
    
    let blockSize = i32(ELEMENTS_PER_THREAD * THREADGROUP_SIZE);
    
    // Figure out this thread group's index into the block data
    var threadgroupBlockStart = blockSize * constants.numBlocksPerThreadGroup * i32(groupID);
    var numBlocksToProcess = constants.numBlocksPerThreadGroup;
    
    if (groupID >= constants.numThreadGroups - constants.numThreadGroupsWithAdditionalBlocks) {
        threadgroupBlockStart += (i32(groupID) - (i32(constants.numThreadGroups) - i32(constants.numThreadGroupsWithAdditionalBlocks))) * blockSize;
        numBlocksToProcess++;
    }
    
    // Get the block start index for this thread
    var blockIndex = threadgroupBlockStart + i32(localID);
    
    // Count value occurrences
    for (var blockCount = 0; blockCount < numBlocksToProcess; blockCount++) {
        var dataIndex = blockIndex;
        
        // Pre-load the key and value arrays (AMD optimization)
        var srcKeys: array<u32, ELEMENTS_PER_THREAD>;
        var srcValues: array<u32, ELEMENTS_PER_THREAD>;
        
        srcKeys[0] = select(0xFFFFFFFFu, sourceKeys[dataIndex], dataIndex < i32(constants.numKeys));
        srcKeys[1] = select(0xFFFFFFFFu, sourceKeys[dataIndex + i32(THREADGROUP_SIZE)], 
                           dataIndex + i32(THREADGROUP_SIZE) < i32(constants.numKeys));
        srcKeys[2] = select(0xFFFFFFFFu, sourceKeys[dataIndex + i32(THREADGROUP_SIZE) * 2], 
                           dataIndex + i32(THREADGROUP_SIZE) * 2 < i32(constants.numKeys));
        srcKeys[3] = select(0xFFFFFFFFu, sourceKeys[dataIndex + i32(THREADGROUP_SIZE) * 3], 
                           dataIndex + i32(THREADGROUP_SIZE) * 3 < i32(constants.numKeys));
        
        srcValues[0] = select(0u, sourceValues[dataIndex], dataIndex < i32(constants.numKeys));
        srcValues[1] = select(0u, sourceValues[dataIndex + i32(THREADGROUP_SIZE)], 
                             dataIndex + i32(THREADGROUP_SIZE) < i32(constants.numKeys));
        srcValues[2] = select(0u, sourceValues[dataIndex + i32(THREADGROUP_SIZE) * 2], 
                             dataIndex + i32(THREADGROUP_SIZE) * 2 < i32(constants.numKeys));
        srcValues[3] = select(0u, sourceValues[dataIndex + i32(THREADGROUP_SIZE) * 3], 
                             dataIndex + i32(THREADGROUP_SIZE) * 3 < i32(constants.numKeys));
        
        for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
            // Clear the local histogram
            if (localID < SORT_BIN_COUNT) {
                atomicStore(&localHistogram[localID], 0u);
            }
            
            var localKey = srcKeys[i];
            var localValue = srcValues[i];
            
            // AMD's local sort algorithm - perform 2 passes of 2-bit radix sort within workgroup
            for (var bitShift = 0u; bitShift < SORT_BITS_PER_PASS; bitShift += 2u) {
                // Figure out the keyIndex for this 2-bit slice
                let keyIndex = (localKey >> constants.shift) & 0xFu;
                let bitKey = (keyIndex >> bitShift) & 0x3u;
                
                // Create a packed histogram (4 2-bit values in one u32)
                let packedHistogram = 1u << (bitKey * 8u);
                
                // Sum up all the packed keys (generates counted offsets up to current thread)
                let localSum = workgroupPrefixScan(packedHistogram, localID);
                
                // Last thread stores the updated histogram counts for the thread group
                if (localID == (THREADGROUP_SIZE - 1u)) {
                    ldsScratch[0] = localSum + packedHistogram;
                }
                
                workgroupBarrier();
                
                // Load the sums value for the thread group
                var packedSum = ldsScratch[0];
                
                // Add prefix offsets for all 4 2-bit "keys"
                packedSum = (packedSum << 8u) + (packedSum << 16u) + (packedSum << 24u);
                
                // Calculate the proper offset for this thread's value
                let localOffset = localSum + packedSum;
                
                // Calculate target offset
                let keyOffset = (localOffset >> (bitKey * 8u)) & 0xFFu;
                
                // Re-arrange the keys (store, sync, load)
                ldsSums[keyOffset] = localKey;
                workgroupBarrier();
                localKey = ldsSums[localID];
                workgroupBarrier();
                
                // Re-arrange the values (store, sync, load)
                ldsSums[keyOffset] = localValue;
                workgroupBarrier();
                localValue = ldsSums[localID];
                workgroupBarrier();
            }
            
            // Need to recalculate the keyIndex now that values have been sorted locally
            let keyIndex = (localKey >> constants.shift) & 0xFu;
            
            // AMD's approach: After local sort, reconstruct histogram
            if (localID < SORT_BIN_COUNT) {
                atomicStore(&localHistogram[localID], 0u);
            }
            workgroupBarrier();
            
            // Since keys are now locally sorted within the workgroup, we can count them more directly
            if (localKey != 0xFFFFFFFFu) {
                atomicAdd(&localHistogram[keyIndex], 1u);
            }
            workgroupBarrier();
            
            // Calculate prefix sum using subgroup operations (simplified)
            var histogramPrefixSum = 0u;
            if (localID < SORT_BIN_COUNT) {
                for (var j = 0u; j < localID; j++) {
                    histogramPrefixSum += atomicLoad(&localHistogram[j]);
                }
                ldsScratch[localID] = histogramPrefixSum;
            }
            
            // Get the global offset for this key out of the cache
            let globalOffset = binOffsetCache[keyIndex];
            
            workgroupBarrier();
            
            // Get the local offset - this is where AMD's algorithm is clever
            // Since keys are sorted locally, we can calculate the offset directly
            let localOffset = localID - ldsScratch[keyIndex];
            
            // Write to destination
            let totalOffset = globalOffset + localOffset;
            
            if (totalOffset < constants.numKeys && localKey != 0xFFFFFFFFu) {
                destKeys[totalOffset] = localKey;
                destValues[totalOffset] = localValue;
            }
            
            workgroupBarrier();
            
            // Update the cached histogram for the next set of entries
            if (localID < SORT_BIN_COUNT) {
                binOffsetCache[localID] += atomicLoad(&localHistogram[localID]);
            }
            
            dataIndex += i32(THREADGROUP_SIZE);
        }
        
        blockIndex += blockSize;
    }
}