# WebGPU Sorting Comparison

A browser-based harness that validates and compares multiple GPU radix sort implementations in WGSL across vendors and architectures, with focus on subgroup behavior, memory ordering, and variant selection.

## What’s inside

- **FidelityFX Parallel Sort** (4-bit digits, 8 passes)
- **DeviceRadixSort** (8-bit digits, 4 passes; reduce, scan, scatter)
- **OneSweep** (8-bit digits, 4 passes; decoupled-lookback prefix) with lane-agnostic ballot/shuffle fixes and per-lane variants

## Purpose

- Verify correctness and portability of subgroup operations (ballot, shuffle, broadcasts) across 16/32/64 and potentially 8-wide subgroups.
- Exercise memory-ordering and forward progress assumptions required by decoupled-lookback scans and chained-scan style algorithms.
- Provide fallbacks and variant selection when device subgroup behavior differs between runs or pipelines.

## Algorithms under test

- **FidelityFX Parallel Sort**: 4-bit radix, 8 passes for 32-bit keys, commonly structured as five stage actions per pass.
- **DeviceRadixSort**: 8-bit radix, four passes, implemented as reduce_hist → scan → scatter kernels in WGSL.
- **OneSweep**: 8-bit radix, four passes, using decoupled-lookback single-pass scan primitives, ported to WGSL with subgroup-safe ballot handling.

## Features

- Optional GPU timestamp-based profiling when the timestamp-query feature is available and enabled at device creation.
- Cross-validation against a CPU baseline to verify per-pass correctness and final output equivalence.
- Configurable problem sizes up to limits permitted by device caps such as maxBufferSize and maxStorageBufferBindingSize.
- Mode selection: run-all or per-algorithm execution for FidelityFX, DeviceRadixSort, OneSweep, or CPU-only.
- Automatic OneSweep variant selection by detected subgroup size range to balance portability and performance.

## Requirements

- Browser/runtime with WebGPU subgroups available; broadly available in Chrome 134+, earlier versions offered experimental support with limitations.
- Device feature “subgroups” requested at device creation and “enable subgroups;” present in WGSL modules that use subgroup intrinsics.
- For timestamped profiling, request and use the timestamp-query feature with guarded code paths.

## Subgroup sizes and variants

The app detects the supported subgroup range using adapter info and device limits (min/max subgroup size) and records the effective subgroup size per dispatch.

OneSweep ships dedicated variants for 16, 32, and 64 lanes, and the ballot logic in WLMS is implemented with vec4<u32> ballots to cover 8–64 lanes robustly.

On Intel Arc, pipelines may compile to SIMD8 or SIMD16 depending on heuristics; a wave8-compatible configuration or variant selection is required for reliable runs.

## Wave8 guidance (Intel)

If a pipeline runs at subgroup_size=8, per-subgroup histogram capacity scales as (BLOCK_DIM / subgroup_size) × RADIX, which can exceed a 16-specialized capacity if not adjusted.

Provide a wave8 variant for DeviceRadixSort/OneSweep by reducing the pass workgroup size (e.g., BLOCK_DIM=128) so WARP_HIST_CAPACITY matches the required histogram bins at wave8.

Keep reduce/global-hist workgroups at sizes that remain divisible by lane_count while sizing shared arrays with MIN_SUBGROUP_SIZE=8 in the wave8 build.

## Running locally

```sh
npm install
npm run dev
npm run build
npm run preview
```

Visit the local preview address and verify that the device reports the expected subgroup range and that features are enabled as indicated by the UI.

## Usage notes

- Select algorithms individually or run all; the result panel shows per-pass timings (if timestamp queries are enabled), correctness, and relative speedup.
- The harness records the subgroup size observed per dispatch so you can correlate correctness and performance with the active wave width.
- For large arrays, ensure device limits requested at creation time are sufficient for buffer sizes and binding sizes required by chosen problem size.

## Clearing persistent state

Clear or reinitialize shared scratch buffers (hist, pass_hist, bump, status) between runs, as algorithms use atomics and prefix data that otherwise persist across dispatches.

A small memset pass or host writes to zero these buffers prevents accumulation artifacts on repeated invocations.

## Troubleshooting

- Errors like “warp hist capacity exceeded” indicate subgroup_size was smaller than the variant assumed, inflating per-subgroup histogram usage beyond the compiled capacity.
- Errors like “shader error in global_hist: 0xDEAD0001” typically mean lane_count < the entrypoint’s MIN_SUBGROUP_SIZE or workgroup_size was not divisible by the active subgroup size.
- If runs succeed once then fail on subsequent runs, suspect subgroup-size changes between pipelines or stale scratch buffers not cleared between dispatches.

## Implementation notes

- Subgroup ballots are handled via vec4<u32> bitfields, combining .x/.y for 64 lanes and avoiding undefined 32+ bit shifts in rank computations.
- DeviceRadixSort kernels: reduce_hist, scan, and scatter (dvr_pass) use @builtin(subgroup_size) with atomics and subgroup scans to merge per-subgroup histograms safely.
- OneSweep variants maintain the decoupled-lookback pattern while ensuring subgroup-agnostic spine scans and per-subgroup histogram merges.

## Performance measurement

When available, timestamp queries are used to measure GPU times per pass and aggregate totals; guard these paths and provide CPU-side timing fallbacks.

Note that timestamp precision and availability vary by platform and driver, so comparisons should be interpreted alongside device limits and active subgroup sizes.

## Credits

FidelityFX Parallel Sort by AMD, with public documentation describing its radix organization and stages.

DeviceRadixSort and OneSweep originate from Thomas Smith's ([@b0nes164](https://github.com/b0nes164/GPUSorting)) work and the OneSweep paper by Adinets and Merrill.

WGSL ports, subgroup fixes, and WebGPU harness integration by project contributors.

## License

MIT license for this repository; consult third-party algorithm licenses as applicable.
