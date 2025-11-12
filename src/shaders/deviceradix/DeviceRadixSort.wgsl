//****************************************************************************
// GPUSorting
// Device Radix Sort
//
// SPDX-License-Identifier: MIT
// Copyright Thomas Smith 12/7/2024
// https://github.com/b0nes164/GPUSorting
//
// Modified for WGSL compatibility and atomic usage by Dino Metarapi, 2025
// Based on original work by Thomas Smith
//****************************************************************************

enable subgroups;

@diagnostic(off, subgroup_uniformity)
fn unsafeSubgroupInclusiveAdd(x: u32) -> u32 { return subgroupInclusiveAdd(x); }

@diagnostic(off, subgroup_uniformity)
fn unsafeSubgroupExclusiveAdd(x: u32) -> u32 { return subgroupExclusiveAdd(x); }

@diagnostic(off, subgroup_uniformity)
fn unsafeSubgroupShuffle(x: u32, source: u32) -> u32 { return subgroupShuffle(x, source); }

@diagnostic(off, subgroup_uniformity)
fn unsafeSubgroupBallot(pred: bool) -> vec4<u32> { return subgroupBallot(pred); }

struct InfoStruct
{
    size: u32,
    shift: u32,
    thread_blocks: u32,
    seed: u32,
};

@group(0) @binding(0)
var<uniform> info : InfoStruct; 

@group(0) @binding(1)
var<storage, read_write> bump: array<u32>;

@group(0) @binding(2)
var<storage, read_write> sort: array<u32>;

@group(0) @binding(3)
var<storage, read_write> alt: array<u32>;

@group(0) @binding(4)
var<storage, read_write> payload: array<u32>;

@group(0) @binding(5)
var<storage, read_write> alt_payload: array<u32>;

@group(0) @binding(6)
var<storage, read_write> hist: array<atomic<u32>>;

@group(0) @binding(7)
var<storage, read_write> pass_hist: array<u32>;

@group(0) @binding(8)
var<storage, read_write> status: array<u32>;

const BLOCK_DIM = 256u;
const MIN_SUBGROUP_SIZE = 16u;
const MAX_REDUCE_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE;

const RADIX = 256u;
const RADIX_MASK = 255u;
const RADIX_LOG = 8u;

const KEYS_PER_THREAD = 15u;
const PART_SIZE = KEYS_PER_THREAD * BLOCK_DIM;

const REDUCE_BLOCK_DIM = 128u;
const REDUCE_KEYS_PER_THREAD = 30u;
// const REDUCE_HIST_SIZE = REDUCE_BLOCK_DIM / 64u * RADIX;
const REDUCE_HIST_SIZE = REDUCE_BLOCK_DIM / MIN_SUBGROUP_SIZE * RADIX; // Sized for MIN_SUBGROUP_SIZE
const REDUCE_PART_SIZE = REDUCE_KEYS_PER_THREAD * REDUCE_BLOCK_DIM;

const MAX_SUBGROUPS_PER_BLOCK = BLOCK_DIM / MIN_SUBGROUP_SIZE;
const WARP_HIST_CAPACITY = MAX_SUBGROUPS_PER_BLOCK * RADIX;

const STATUS_ERR_REDUCE = 0u;
const STATUS_ERR_SCAN = 1u;
const STATUS_ERR_DVR = 2u;
const STATUS_SUBGROUP_BASE = 3u;
const STATUS_SUBGROUP_STRIDE = 3u;
const STATUS_STAGE_REDUCE = 0u;
const STATUS_STAGE_SCAN = 1u;
const STATUS_STAGE_DVR = 2u;

var<workgroup> wg_globalHist: array<atomic<u32>, REDUCE_HIST_SIZE>;

@compute @workgroup_size(REDUCE_BLOCK_DIM, 1, 1)
fn reduce_hist(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) wgid: vec3<u32>) {

    if (lane_count < MIN_SUBGROUP_SIZE || (REDUCE_BLOCK_DIM % lane_count) != 0u) {
        if (threadid.x == 0u) {
            status[STATUS_ERR_REDUCE] = 0xDEAD0001u;
        }
        return;
    }

    if (wgid.x == 0u && threadid.x == 0u) {
        let pass_idx = info.shift / RADIX_LOG;
        status[STATUS_SUBGROUP_BASE + pass_idx * STATUS_SUBGROUP_STRIDE + STATUS_STAGE_REDUCE] = lane_count;
    }

    let sid = threadid.x / lane_count;

    for (var i = threadid.x; i < REDUCE_HIST_SIZE; i += REDUCE_BLOCK_DIM) {
        atomicStore(&wg_globalHist[i], 0u);
    }
    workgroupBarrier();

    let radix_shift = info.shift;
    let hist_offset = sid * RADIX;
    //let hist_offset = sid * RADIX;
    {
        var i = threadid.x + wgid.x * REDUCE_PART_SIZE;
        if(wgid.x < info.thread_blocks - 1) {
            for (var k = 0u; k < REDUCE_KEYS_PER_THREAD; k += 1u) {
                let key = sort[i];
                atomicAdd(&wg_globalHist[((key >> radix_shift) & RADIX_MASK) + hist_offset], 1u);
                i += REDUCE_BLOCK_DIM;
            }
        }

        if(wgid.x == info.thread_blocks - 1) {
            for (var k = 0u; k < REDUCE_KEYS_PER_THREAD; k += 1u) {
                if (i < info.size) {
                    let key = sort[i];
                    atomicAdd(&wg_globalHist[((key >> radix_shift) & RADIX_MASK) + hist_offset], 1u);
                }
                i += REDUCE_BLOCK_DIM;
            }
        }
    }
    workgroupBarrier();

    let subgroup_histograms = REDUCE_BLOCK_DIM / lane_count;
    let lane_mask = lane_count - 1u;
    let circular_lane_shift = (laneid + lane_mask) & lane_mask;
    for (var i = threadid.x; i < RADIX; i += REDUCE_BLOCK_DIM) {
        var reduction = atomicLoad(&wg_globalHist[i]);
        var idx = i + RADIX;
        for (var h = 1u; h < subgroup_histograms; h += 1u) {
            let val = atomicLoad(&wg_globalHist[idx]);
            reduction += val;
            atomicStore(&wg_globalHist[idx], reduction - val);
            idx += RADIX;
        }
        pass_hist[wgid.x + i * info.thread_blocks] = reduction;
        let t = unsafeSubgroupInclusiveAdd(reduction);
        atomicStore(&wg_globalHist[i], unsafeSubgroupShuffle(t, circular_lane_shift));
    }
    workgroupBarrier();

    if (threadid.x < lane_count) {
        let pred = threadid.x < RADIX / lane_count;
        let t = unsafeSubgroupExclusiveAdd(select(0u, atomicLoad(&wg_globalHist[threadid.x * lane_count]), pred));
        if (pred) {
            atomicStore(&wg_globalHist[threadid.x * lane_count], t);
        }
    }
    workgroupBarrier();
    
    for (var i = threadid.x; i < RADIX; i += REDUCE_BLOCK_DIM) {
        atomicAdd(&hist[i + radix_shift * 32],
            atomicLoad(&wg_globalHist[i]) + select(0u, atomicLoad(&wg_globalHist[i / lane_count * lane_count]), laneid != 0u));
    }
}

const SCAN_SPT = 8u;
const SCAN_PART_SIZE = BLOCK_DIM * SCAN_SPT;

var<workgroup> wg_reduce: array<u32, MAX_REDUCE_SIZE>;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn scan(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) wgid: vec3<u32>) {
    
    if (lane_count < MIN_SUBGROUP_SIZE || (BLOCK_DIM % lane_count) != 0u) {
        if (threadid.x == 0u) {
            status[STATUS_ERR_SCAN] = 0xDEAD0002u;
        }
        return;
    }

    if (wgid.x == 0u && threadid.x == 0u) {
        let pass_idx = info.shift / RADIX_LOG;
        status[STATUS_SUBGROUP_BASE + pass_idx * STATUS_SUBGROUP_STRIDE + STATUS_STAGE_SCAN] = lane_count;
    }

    let sid = threadid.x / lane_count;
    let radix_offset = wgid.x * info.thread_blocks;
    let lane_log = u32(countTrailingZeros(lane_count));
    let s_offset = laneid + sid * lane_count * SCAN_SPT;
    let local_spine_size = BLOCK_DIM >> lane_log;
    let local_aligned_size = 1u << ((u32(countTrailingZeros(local_spine_size)) + lane_log - 1u) / lane_log * lane_log);
    let aligned_size = (info.thread_blocks + SCAN_PART_SIZE - 1u) / SCAN_PART_SIZE * SCAN_PART_SIZE;
    var t_scan = array<u32, SCAN_SPT>();
    
    var prev_red = 0u;
    for(var dev_offset = 0u; dev_offset < aligned_size; dev_offset += SCAN_PART_SIZE){
        {
            var i = s_offset + dev_offset;
            for(var k = 0u; k < SCAN_SPT; k += 1u){
                if(i < info.thread_blocks){
                    t_scan[k] = pass_hist[i + radix_offset];
                }
                i += lane_count;
            }
        }

        var prev = 0u;
        for(var k = 0u; k < SCAN_SPT; k += 1u){
            t_scan[k] = unsafeSubgroupInclusiveAdd(t_scan[k]) + prev;
            prev = unsafeSubgroupShuffle(t_scan[k], lane_count - 1);
        }

        if(laneid == lane_count - 1u){
            wg_reduce[sid] = prev;
        }
        workgroupBarrier();

        {   
            var offset0 = 0u;
            var offset1 = 0u;
            for(var j = lane_count; j <= local_aligned_size; j <<= lane_log){
                let i0 = ((threadid.x + offset0) << offset1) - select(0u, 1u, j != lane_count);
                let pred0 = i0 < local_spine_size;
                let t0 = unsafeSubgroupInclusiveAdd(select(0u, wg_reduce[i0], pred0));
                if(pred0){
                    wg_reduce[i0] = t0;
                }
                workgroupBarrier();

                if(j != lane_count){
                    let rshift = j >> lane_log;
                    let i1 = threadid.x + rshift;
                    if ((i1 & (j - 1u)) >= rshift){
                        let pred1 = i1 < local_spine_size;
                        let t1 = select(0u, wg_reduce[((i1 >> offset1) << offset1) - 1u], pred1);
                        if(pred1 && ((i1 + 1u) & (rshift - 1u)) != 0u){
                            wg_reduce[i1] += t1;
                        }
                    }
                } else {
                    offset0 += 1u;
                }
                offset1 += lane_log;
            }
        }   
        workgroupBarrier();

        {
            let prev = select(0u, wg_reduce[sid - 1u], sid != 0u) + prev_red;
            var i: u32 = s_offset + dev_offset;
            for(var k = 0u; k < SCAN_SPT; k += 1u){
                if(i < info.thread_blocks){
                    pass_hist[i + radix_offset] = t_scan[k] + prev;
                }
                i += lane_count;
            }
        }

        prev_red += subgroupBroadcast(wg_reduce[local_spine_size - 1u], 0u);
        workgroupBarrier();
    }
}    

// var<workgroup> wg_warpHist: array<atomic<u32>, PART_SIZE>;
var<workgroup> wg_warpHist: array<atomic<u32>, WARP_HIST_CAPACITY>;
var<workgroup> wg_localHist: array<u32, RADIX>;

fn WLMS(key: u32, shift: u32, laneid: u32, lane_count: u32, lane_mask_lt: u32, s_offset: u32) -> u32 {
    var eq_mask = 0xffffffffu;
    for(var k = 0u; k < RADIX_LOG; k += 1u) {
        let curr_bit = 1u << (k + shift);
        let pred = (key & curr_bit) != 0u;
        let ballot = unsafeSubgroupBallot(pred);
        eq_mask &= select(~ballot.x, ballot.x, pred);
    }
    var subgroup_mask = 0xffffffffu;
    if (lane_count != 32u) {
        subgroup_mask = (1u << lane_count) - 1u;
    }
    eq_mask &= subgroup_mask;
    var out = countOneBits(eq_mask & lane_mask_lt);
    let highest_rank_peer = select(lane_count - 1u, 31u - countLeadingZeros(eq_mask), eq_mask != 0u);
    var pre_inc = 0u;
    if (laneid == highest_rank_peer && eq_mask != 0u) {
        pre_inc = atomicAdd(&wg_warpHist[((key >> shift) & RADIX_MASK) + s_offset], out + 1u);
    }
    workgroupBarrier();
    out += unsafeSubgroupShuffle(pre_inc, highest_rank_peer);
    return out;
}

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn dvr_pass(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) wgid: vec3<u32>) {
    
    let sid = threadid.x / lane_count;

    let warp_hists_size = (BLOCK_DIM / lane_count) * RADIX;
    if (warp_hists_size > WARP_HIST_CAPACITY) {
        if (threadid.x == 0u) {
            status[STATUS_ERR_DVR] = 0xDEAD0004u;
        }
        return;
    }

    if (wgid.x == 0u && threadid.x == 0u) {
        let pass_idx = info.shift / RADIX_LOG;
        status[STATUS_SUBGROUP_BASE + pass_idx * STATUS_SUBGROUP_STRIDE + STATUS_STAGE_DVR] = lane_count;
    }
    // let warp_hists_size = clamp(BLOCK_DIM / lane_count * RADIX, 0u, PART_SIZE);
    for (var i = threadid.x; i < warp_hists_size; i += BLOCK_DIM) {
        atomicStore(&wg_warpHist[i], 0u);
    }
    workgroupBarrier();

    var keys = array<u32, KEYS_PER_THREAD>();
    {
        let dev_offset = wgid.x * PART_SIZE;
        let s_offset = sid * lane_count * KEYS_PER_THREAD;
        var i = laneid + s_offset + dev_offset;
        if (wgid.x < info.thread_blocks - 1) {
            for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
                keys[k] = sort[i];
                i += lane_count;
            }
        }

        if (wgid.x == info.thread_blocks - 1) {
            for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
                keys[k] = select(0xffffffffu, sort[i], i < info.size);
                i += lane_count;
            }
        }
    }

    var offsets = array<u32, KEYS_PER_THREAD>();
    {
        let shift = info.shift;
        let lane_mask_lt = (1u << laneid) - 1u;
        let s_offset = sid * RADIX;
        for(var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
            offsets[k] = WLMS(keys[k], shift, laneid, lane_count, lane_mask_lt, s_offset);
        }
    }
    workgroupBarrier();

    if (threadid.x < RADIX) {
        var reduction = atomicLoad(&wg_warpHist[threadid.x]);
        for (var i = threadid.x + RADIX; i < warp_hists_size; i += RADIX) {
            reduction += atomicLoad(&wg_warpHist[i]);
            atomicStore(&wg_warpHist[i], reduction - atomicLoad(&wg_warpHist[i]));
        }

        let lane_mask = lane_count - 1u;
        let circular_lane_shift = (laneid + lane_mask) & lane_mask;
        let t = unsafeSubgroupInclusiveAdd(reduction);
        atomicStore(&wg_warpHist[threadid.x], unsafeSubgroupShuffle(t, circular_lane_shift));
    }
    workgroupBarrier();

    if (threadid.x < lane_count) {
        let pred = threadid.x < RADIX / lane_count;
        let t = unsafeSubgroupExclusiveAdd(select(0u, atomicLoad(&wg_warpHist[threadid.x * lane_count]), pred));
        if (pred) {
            atomicStore(&wg_warpHist[threadid.x * lane_count], t);
        }
    }
    workgroupBarrier();
    
    if (threadid.x < RADIX && laneid != 0u) {
        let lhs = atomicLoad(&wg_warpHist[threadid.x]);
        let rhs = atomicLoad(&wg_warpHist[threadid.x / lane_count * lane_count]);
        atomicStore(&wg_warpHist[threadid.x], lhs + rhs);
    }
    workgroupBarrier();

    if (threadid.x >= lane_count) {
        let s_offset = sid * RADIX;
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
            let t = (keys[k] >> info.shift) & RADIX_MASK;
            offsets[k] += atomicLoad(&wg_warpHist[t]) + atomicLoad(&wg_warpHist[t + s_offset]);
        }
    } else {
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
            offsets[k] += atomicLoad(&wg_warpHist[(keys[k] >> info.shift) & RADIX_MASK]);
        }
    }

    if (threadid.x < RADIX) {
        wg_localHist[threadid.x] = atomicLoad(&hist[threadid.x + info.shift * 32]) +
            select(0u, pass_hist[wgid.x + info.thread_blocks * threadid.x - 1u], wgid.x != 0u) 
            - atomicLoad(&wg_warpHist[threadid.x]);
    }
    workgroupBarrier();

    for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
        atomicStore(&wg_warpHist[offsets[k]], keys[k]);
    }
    workgroupBarrier();

    if (wgid.x < info.thread_blocks - 1u) {
        var i = threadid.x;
        for (var k = 0u; k < KEYS_PER_THREAD; k += 1u) {
            alt[wg_localHist[(atomicLoad(&wg_warpHist[i]) >> info.shift) & RADIX_MASK] + i] = atomicLoad(&wg_warpHist[i]);
            i += BLOCK_DIM;
        }
    }

    if (wgid.x == info.thread_blocks - 1u) {
        let final_size = info.size - wgid.x * PART_SIZE;
        for (var i = threadid.x; i < final_size; i += BLOCK_DIM) {
            alt[wg_localHist[(atomicLoad(&wg_warpHist[i]) >> info.shift) & RADIX_MASK] + i] = atomicLoad(&wg_warpHist[i]);
        }
    }
}
