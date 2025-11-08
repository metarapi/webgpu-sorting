// WebGPU/WGSL port of AMD FidelityFX Parallel Sort - Scan Add Pass
// Adds partial sums from reduced scan results

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

// In this pass, source and destination are the same buffer (sum table), used in-place
@group(0) @binding(1) var<storage, read_write> sumTable: array<u32>;
// Reduced scan results used as partial sums per chunk (scanned reduce table)
@group(0) @binding(2) var<storage, read> scanScratch: array<u32>;

var<workgroup> ldsData: array<array<u32, THREADGROUP_SIZE>, ELEMENTS_PER_THREAD>;
var<workgroup> subgroupSums: array<u32, THREADGROUP_SIZE>;

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) groupId: vec3<u32>,
  @builtin(subgroup_size) subgroupSize: u32,
  @builtin(subgroup_invocation_id) subgroupLane: u32
) {
  let localID = localId.x;
  let groupID = groupId.x;

  // Figure out which bin/chunk this group is processing
  let binID = groupID / constants.numReduceThreadgroupPerBin;
  let binOffset = binID * constants.numThreadGroups;
  let baseIndex = (groupID % constants.numReduceThreadgroupPerBin) * ELEMENTS_PER_THREAD * THREADGROUP_SIZE;

  // We scan NumThreadGroups values per bin (with coalesced access pattern)
  let numValuesToScan = constants.numThreadGroups;

  // Coalesced loads from source (sum table) into LDS
  for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
    let dataIndex = baseIndex + (i * THREADGROUP_SIZE) + localID;

    let col = ((i * THREADGROUP_SIZE) + localID) / ELEMENTS_PER_THREAD;
    let row = ((i * THREADGROUP_SIZE) + localID) % ELEMENTS_PER_THREAD;

  ldsData[row][col] = select(0u, sumTable[binOffset + dataIndex], dataIndex < numValuesToScan);
  }

  workgroupBarrier();

  // Local thread prefix accumulation through LDS
  var threadgroupSum = 0u;
  for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
    let tmp = ldsData[i][localID];
    ldsData[i][localID] = threadgroupSum;
    threadgroupSum += tmp;
  }

  // Subgroup exclusive scan
  let subgroupPrefixed = subgroupExclusiveAdd(threadgroupSum);
  let laneID = subgroupLane;
  let waveSize = subgroupSize;
  let subgroupID = localID / waveSize;

  // Last lane in each subgroup writes partial sum
  if (laneID == waveSize - 1u) {
    subgroupSums[subgroupID] = subgroupPrefixed + threadgroupSum;
  }

  workgroupBarrier();

  // First subgroup scans subgroup sums (uniform control flow)
  let numSubgroups = THREADGROUP_SIZE / waveSize;
  var valueToScan = 0u;
  if (localID < numSubgroups) {
    valueToScan = subgroupSums[localID];
  }
  let scannedValue = subgroupExclusiveAdd(valueToScan);
  if (localID < numSubgroups) {
    subgroupSums[localID] = scannedValue;
  }

  workgroupBarrier();

  // Add subgroup partial sum back to each thread's prefix
  let finalPrefixSum = subgroupPrefixed + subgroupSums[subgroupID];

  // Add block scanned-prefix back into LDS
  for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
    ldsData[i][localID] += finalPrefixSum;
  }

  workgroupBarrier();

  // Partial sum addition from reduced scan results for this chunk
  let partialSum = scanScratch[groupID];

  // Coalesced writes back to destination (same buffer), with partial sums added
  for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
    let dataIndex = baseIndex + (i * THREADGROUP_SIZE) + localID;

    let col = ((i * THREADGROUP_SIZE) + localID) / ELEMENTS_PER_THREAD;
    let row = ((i * THREADGROUP_SIZE) + localID) % ELEMENTS_PER_THREAD;

    if (dataIndex < numValuesToScan) {
      sumTable[binOffset + dataIndex] = ldsData[row][col] + partialSum;
    }
  }
}
