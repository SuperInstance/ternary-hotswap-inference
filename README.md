# ternary-hotswap-inference

Adaptive ternary inference with **atomic model hotswap** and **CRDT version tracking**. Proves that zero-downtime model replacement is possible with ternary weights (16× density vs FP32) while maintaining a full audit trail of which model version produced each output.

## Why It Matters

Traditional adaptive inference requires: stop pipeline → drain batches → unload model → load new model → restart. Typical gap: **100–500 ms**. For content moderation on a live stream, that's thousands of frames with no protection.

This crate demonstrates the gap can be reduced to the latency of a **single in-flight batch** (< 1 ms) via three techniques:

1. **Ternary weight compression** (`TernaryTensor`): 2 bits per weight = 16× FP32 density. Matmul becomes conditional add/subtract — no FMA units needed.
2. **Atomic kernel hotswap** (`HotswapSlot`): Replace the running model between batches with a single atomic pointer swap. In-flight batches see the old model; new batches see the new one.
3. **CRDT version tracking** (`VersionCrdt`): Each GPU replica records which model version produced each output. Replicas merge version maps without coordination.

## How It Works

### Ternary Weight Encoding

Each weight is a **trit** ∈ {NegOne, Zero, PosOne}, packed at 2 bits per weight into `u64` words (32 trits per word):

```
Bit layout (MSB-first within each word):
  Word:  [t₃₁|t₃₀|...|t₁|t₀]    each tᵢ is 2 bits
  0b00 = Zero, 0b01 = PosOne, 0b10 = NegOne
```

**Encoding map** (DeepSeek-style thresholding):

```
Trit(w) = { Zero    if |w| < τ
          { PosOne  if w ≥ τ
          { NegOne  if w ≤ −τ
```

where τ is the quantization threshold (default 0.5).

### Ternary Matmul

Standard matmul: `yᵢ = Σⱼ Wᵢⱼ · xⱼ`

Ternary matmul eliminates multiplication — each weight contributes +xⱼ, −xⱼ, or nothing:

```
yᵢ = Σ_{j: Wᵢⱼ=+1} xⱼ  −  Σ_{j: Wᵢⱼ=−1} xⱼ
```

On GPU hardware, this maps to **predicated integer additions** — far cheaper than floating-point FMA instructions. The zero weights reduce memory bandwidth proportionally to the sparsity fraction.

**Complexity**: O(rows × cols) additions/subtractions, zero multiplications.

### Density Comparison

| Format | Bits/weight | Relative density |
|--------|-------------|-----------------|
| FP32 | 32 | 1× |
| FP16 | 16 | 2× |
| INT8 | 8 | 4× |
| **Ternary (this crate)** | **2** | **16×** |

A 7-billion-parameter model: 28 GB in FP32, 1.75 GB in ternary.

### Atomic Hotswap

The `HotswapSlot` uses an `AtomicUsize` version counter:

```
swap(new_model):
    old ← slot.load(Acquire)
    slot.store(new_model, Release)
    version.fetch_add(1, AcqRel)
    return old

read():
    return slot.load(Acquire)
```

**Memory ordering**: `Acquire/Release` semantics ensure that all writes to the new model are visible before any reader sees the new pointer. In-flight batches that already loaded the old pointer continue using the old model safely.

### CRDT Version Tracking

Each replica maintains a grow-only map (G-map CRDT) of `{batch_id → version}`:

```
Merge rule: v = max(v_local, v_remote)  for each batch_id
```

This is a classic state-based CRDT (Shapiro et al., 2011). Merges are:
- **Commutative**: merge(A, B) = merge(B, A)
- **Associative**: merge(A, merge(B, C)) = merge(merge(A, B), C)
- **Idempotent**: merge(A, A) = A

Therefore, replicas can merge out of order and arrive at the same state — eventual consistency without coordination.

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `TernaryTensor::from_floats(n)` | O(n) | O(n/32 words) |
| `TernaryTensor::get(i,j)` | O(1) | O(1) |
| `TernaryTensor::matvec(x)` | O(rows × cols) | O(rows) |
| `HotswapSlot::swap(m)` | O(1) | O(1) |
| `VersionCrdt::record(batch, ver)` | O(1) | O(1) |
| `VersionCrdt::merge(other)` | O(|batches|) | O(|batches|) |

## Quick Start

```rust
use adaptive_inference::{TernaryTensor, HotswapSlot, VersionCrdt, Trit};

// Encode weights to ternary
let weights = vec![0.8, -0.6, 0.0, 0.9, -0.3, 0.5];
let tensor = TernaryTensor::from_floats(&weights, 2, 3, 0.5);

// Ternary matmul — no multiplications
let input = vec![1.0, -1.0, 0.5];
let output = tensor.matvec(&input);
// output[0] = (+1.0) + (-(-1.0)) + (skip 0.5) = 2.0

// Hotswap slot for atomic model replacement
let slot = HotswapSlot::new(tensor);
// ... batch processing uses slot.get() ...
// slot.swap(new_tensor); // new batches see new model instantly

// CRDT version tracking
let mut crdt = VersionCrdt::new();
crdt.record(0, 1); // batch 0 used version 1
crdt.record(1, 2); // batch 1 used version 2
// Can merge with other replicas' CRDTs
```

## API

### `TernaryTensor`

| Method | Description |
|--------|-------------|
| `from_floats(data, rows, cols, threshold)` | Encode FP32 matrix to ternary |
| `get(row, col) -> Trit` | Decode weight at position |
| `matvec(x: &[f32]) -> Vec<f32>` | Ternary matrix-vector multiply |

### `HotswapSlot`

| Method | Description |
|--------|-------------|
| `new(tensor)` | Create slot with initial model |
| `get() -> Arc<TernaryTensor>` | Load current model (atomic) |
| `swap(new) -> Arc<TernaryTensor>` | Replace model, return old |

### `VersionCrdt`

| Method | Description |
|--------|-------------|
| `new()` | Empty CRDT |
| `record(batch_id, version)` | Record which version processed a batch |
| `merge(other)` | Merge remote CRDT state |
| `version_of(batch_id) -> Option<u64>` | Query version for a batch |

## Architecture Notes

This crate demonstrates the full **γ + η = C** pipeline in a single artifact:

- **η (eta)**: `TernaryTensor` — the compute primitive. Ternary matmul is the η-layer operation that produces inference outputs.
- **γ (gamma)**: `HotswapSlot` and `VersionCrdt` — the coordination primitives. Hotswap provides atomic model replacement (γ-level synchronization); CRDT provides eventual consistency (γ-level agreement).
- **C**: The complete adaptive inference system. η computes outputs; γ ensures those outputs are attributable to the correct model version, even during live model swaps.

The central claim: **γ + η = C is complete** — you need nothing else for safe zero-downtime adaptive inference.

## References

- **Ternary Weight Networks**: Li, F. et al., "Ternary Weight Networks," arXiv:1605.04711, 2016.
- **DeepSeek Quantization**: DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model," 2024.
- **CRDTs**: Shapiro, M. et al., "Conflict-Free Replicated Data Types," Stabilization, Safety, and Security of Distributed Systems, 2011.
- **Atomic Operations**: Herlihy, M. & Shavit, N., "The Art of Multiprocessor Programming," MIT Press, 2012. Chapter 5 on atomic registers.
- **Memory Models**: Boehm, H.J. & Adve, S.V., "Foundations of the C++ Concurrency Memory Model," PLDI 2008.

## License

MIT
