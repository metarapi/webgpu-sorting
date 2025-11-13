enable subgroups;

@group(0) @binding(0)
var<storage, read_write> outSize : array<u32, 1>;

@compute @workgroup_size(1)
fn main(@builtin(subgroup_size) subgroupSize : u32) {
    outSize[0] = subgroupSize;
}
