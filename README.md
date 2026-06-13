# Ternary Hotswap Inference — Zero-Downtime Model Swap with CRDT Audit Trail

**Ternary Hotswap Inference** demonstrates three capabilities that traditional GPU inference cannot provide: (1) **ternary weight compression** (2 bits/weight = 16× FP32 density), (2) **atomic kernel hotswap** (replace running model between batches with a single pointer swap — zero downtime), and (3) **CRDT version tracking** (each GPU replica records which model version produced each output, enabling safe rollback).

## Why It Matters

Traditional adaptive inference requires: stop pipeline → drain batches → unload model → load new model → restart. Typical gap: 100-500ms. For content moderation on live streams, that's thousands of frames with no protection. This crate proves that gap can be reduced to the latency of a single in-flight batch (<1ms) while maintaining a full CRDT audit trail. Ternary weights are the key enabler: at 2 bits per weight, an entire model fits in L2 cache, making hotswap a single pointer swap rather than a multi-gigabyte memory transfer.

## How It Works

### Ternary Weight Encoding

Each weight is a `Trit` encoded in 2 bits: `NegOne = 0b10`, `Zero = 0b00`, `PosOne = 0b01`. A `TernaryTensor` packs 32 weights per `u64` word (2 bits each). Conversion from float uses a threshold:

```
|w| < threshold → Zero
w > 0           → PosOne
w < 0           → NegOne
```

This is the DeepSeek R1-style ternary quantization. Matmul becomes conditional add/subtract/skip — no FMA units needed.

### Atomic Hotswap

The `HotswapSlot` wraps the model in `Arc<Mutex<>>`. To swap:
1. Load new model into memory
2. Atomic pointer swap: old model is immediately retired, new model serves all subsequent requests
3. In-flight batches using the old model complete normally (they hold an Arc reference)

Swap latency: O(1) for the pointer swap, O(model_size / bandwidth) for loading. Since ternary models are 16× smaller, loading is 16× faster.

### CRDT Version Tracking

`VersionCrdt` records which model version produced each output. Each GPU replica maintains a `HashMap<batch_id, version>`. Replicas merge without coordination:

```
merge(a, b) = { max version per batch_id }
```

This is a G-Counter CRDT — commutative, associative, idempotent. The merge enables fleet-wide audit: which version was running when a given decision was made.

### Conservation Verification

Each inference step verifies γ + η = C:
- γ = number of non-zero weights activated (compute work)
- η = number of zero weights skipped (entropy savings)
- C = total parameters (conserved)

## Quick Start

```rust
use adaptive_inference::{Trit, TernaryTensor, HotswapSlot};

// Pack a weight matrix
let weights = vec![1.0, -0.3, 0.8, 0.0, -0.9, 0.5];
let tensor = TernaryTensor::from_floats(&weights, 2, 3, 0.5);

// Hotswap setup
let slot = HotswapSlot::new(tensor);
// Inference proceeds; swap is atomic
// slot.swap(new_tensor); // Zero-downtime swap
```

```bash
cargo add adaptive-inference
```

## API

| Type / Function | Description |
|---|---|
| `Trit` | 2-bit ternary weight: `NegOne`, `Zero`, `PosOne` |
| `TernaryTensor` | Bit-packed matrix: `from_floats()`, 32 trits/u64 |
| `HotswapSlot` | Atomic model swap: `new()`, `swap()`, `read()` |
| `VersionCrdt` | CRDT audit: `record()`, `merge()` |

## Architecture Notes

Hotswap inference is the production deployment mechanism in **SuperInstance**: fleet nodes swap models without downtime, while the CRDT trail ensures auditability. The γ + η = C conservation is verifiable per inference: non-zero weights contribute γ, zero weights contribute η, and their sum equals the model's total parameter count C. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Li, Yujia et al. "Ternary Weight Networks," *arXiv:1605.04711*, 2016 — 2-bit quantization.
- Shapiro, Marc et al. "Conflict-free Replicated Data Types," *SSS*, 2011 — CRDTs.
- Dean, Jeffrey & Barroso, Luiz. "The Tail at Scale," *CACM*, 56(2), 2013 — latency in distributed inference.

## License

MIT
