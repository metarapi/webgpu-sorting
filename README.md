# WebGPU Sorting Comparison



A hardware compatibility testing tool for WebGPU sorting implementations. Deployed via GitHub Pages to enable quick browser-based testing across different GPU vendors (Apple, Intel, AMD, Nvidia) without local setup. [**Try it live on GitHub Pages**](https://metarapi.github.io/webgpu-sorting/)

## Purpose

This tool verifies that WGSL sorting implementations work correctly across diverse hardware configurations by testing:
- **Subgroup operations** (ballot, shuffle, broadcast) vendor compatibility
- **Progress guarantees** and memory ordering across GPU architectures
- **Escape hatches** and fallback paths for unsupported features

Simply open the GitHub Pages deployment in a browser on any system to validate sorting behavior and performance characteristics specific to that GPU.

## What It Tests

Two GPU radix sort implementations ported to WGSL:
1. **FidelityFX Radix Sort** (AMD) - 8 passes, 4-bit radix, 5 kernels per pass
2. **DeviceRadixSort** (b0nes164) - 4 passes, 8-bit radix, 3 kernels per pass

Both compared against JavaScript `Array.sort()` baseline with validation.

## Features

- GPU timestamp-based performance measurement (when available)
- Cross-validation between all implementations
- Configurable array sizes up to 10M elements
- Dark mode UI
- Results show execution time, correctness, and relative speedup

## Getting Started

```sh
npm install
npm run dev # Development server
npm run build # Production build
npm run preview # Preview production build
```

Visit `http://localhost:3000/webgpu-sorting/`

## Recent Changes (Nov 2025)

### Subgroup Size Reporting
- The app now reports the actual GPU subgroup size range using `adapter.info.subgroupMinSize` and `adapter.info.subgroupMaxSize` (when available), falling back to device/adapter limits if needed. This ensures the UI displays the true hardware subgroup configuration for more accurate compatibility testing.

### DeviceRadixSort & OneSweep Shader Adaptation
- Both shaders were adapted to support variable subgroup sizes (16, 32, 64, etc.) rather than assuming a fixed size (e.g., 64). This includes:
	- Dynamic workgroup memory sizing and histogram partitioning based on runtime subgroup size.
	- Use of `@builtin(subgroup_size)` and atomic operations for safe, portable reductions.
	- Logic for merging multiple subgroup histograms and prefix sums, supporting all modern GPU architectures.
- See `tmp/AdaptationExplanation.md` for a detailed technical summary of these changes.

### UI Improvements
- The results panel now shows detected subgroup sizes for each algorithm, and DeviceRadixSort pass details are collapsible for clarity.

### Why?
- These changes improve hardware portability, make debugging easier, and ensure the tool surfaces true GPU capabilities for subgroup/wave operations.

## Requirements

- Browser with WebGPU subgroups support (Chrome 113+, Edge 113+)
- GPU with subgroup/wave operations

## Credits

- **FidelityFX Radix Sort**: AMD FidelityFX SDK
- **DeviceRadixSort**: Thomas Smith ([@b0nes164](https://github.com/b0nes164/GPUSorting))
- WGSL ports and WebGPU implementation by project contributors

## License

MIT
