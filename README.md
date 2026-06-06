# ternary-hotswap-inference

Adaptive ternary inference with atomic model hotswap and CRDT audit trail. Proves zero-downtime model swap is possible with ternary weights (16x density) and full version tracking.

## Overview

# Adaptive Inference Experiment

Demonstrates three capabilities that traditional GPU programming cannot provide:

## Stats

- **Tests**: 20
- **LOC**: 709
- **License**: Apache-2.0

## Part of the Oxide Stack

This crate is part of the [Flux→PTX](https://github.com/SuperInstance/cuda-oxide/blob/main/FLUX_TO_PTX.md) experimental suite, testing synergies between the five layers of the distributed GPU runtime:

1. **open-parallel** — async runtime (tokio fork)
2. **pincher** — "Vector DB as runtime, LLM as compiler"
3. **flux-core** — bytecode VM + A2A agent protocol
4. **cuda-oxide** — Flux→MIR→Pliron→NVVM→PTX compiler
5. **cudaclaw** — persistent GPU kernels, warp-level consensus, SmartCRDT

## Usage

```rust
use ternary_hotswap_inference::*;
// See tests in src/lib.rs for examples
```

## License

Apache-2.0
