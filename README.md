# ternary-hotswap-inference

Adaptive ternary inference with atomic model hotswap and CRDT audit trail. Proves zero-downtime model swap is possible with ternary weights (16x density) and full version tracking.

## Why This Matters

# Adaptive Inference Experiment
Demonstrates three capabilities that traditional GPU programming cannot provide:
1. **Ternary weight compression** (`TernaryTensor`): 2 bits per weight = 16× FP32 density.
Matmul becomes conditional adds/subtracts — no FMA units needed.
2. **Atomic kernel hotswap** (`HotswapSlot`): Replace the running model between batches

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub enum Trit
pub fn from_float
pub fn as_float
pub struct TernaryTensor
pub fn from_floats
pub fn get
pub fn matvec
pub fn packed_bytes
pub fn fp32_bytes
pub struct ModelVersion
pub struct TernaryModel
pub fn new
```

## Usage

```toml
[dependencies]
ternary-hotswap-inference = "0.1.0"
```

```rust
use ternary_hotswap_inference::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/ternary-hotswap-inference.git
cd ternary-hotswap-inference
cargo test    # 20 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 20 |
| Lines of Rust | 710 |
| Public API | 32 items |

## License

Apache-2.0
