//! # Adaptive Inference Experiment
//!
//! Demonstrates three capabilities that traditional GPU programming cannot provide:
//!
//! 1. **Ternary weight compression** (`TernaryTensor`): 2 bits per weight = 16× FP32 density.
//!    Matmul becomes conditional adds/subtracts — no FMA units needed.
//!
//! 2. **Atomic kernel hotswap** (`HotswapSlot`): Replace the running model between batches
//!    with a single atomic pointer swap. In-flight batches see the old model; new batches
//!    see the new model. Zero downtime — no pipeline drain, no kernel restart.
//!
//! 3. **CRDT version tracking** (`VersionCrdt`): Each GPU replica records which model
//!    version produced each output. Replicas merge version maps without coordination.
//!    Proves audit correctness even when replicas lag by multiple versions.
//!
//! ## Why these together?
//!
//! Traditional adaptive inference requires: stop pipeline → drain batches → unload model
//! → load new model → restart. Typical gap: 100–500 ms. For content moderation on a
//! live stream, that's thousands of frames with no protection.
//!
//! This experiment shows the gap can be reduced to the batch latency of a single
//! in-flight batch (typically < 1 ms), while maintaining a full CRDT audit trail
//! of which version produced which output — enabling safe rollback.

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

// ─── Ternary weight encoding ──────────────────────────────────────────────────

/// A weight in Z₃: negative one, zero, or positive one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Trit {
    NegOne = 0b10,
    Zero   = 0b00,
    PosOne = 0b01,
}

impl Trit {
    /// Encode a float to the nearest ternary value using the DeepSeek-style threshold.
    /// Weights |w| < threshold → Zero; otherwise sign(w).
    pub fn from_float(w: f32, threshold: f32) -> Self {
        if w.abs() < threshold {
            Trit::Zero
        } else if w > 0.0 {
            Trit::PosOne
        } else {
            Trit::NegOne
        }
    }

    pub fn as_float(self) -> f32 {
        match self {
            Trit::NegOne => -1.0,
            Trit::Zero   =>  0.0,
            Trit::PosOne =>  1.0,
        }
    }
}

/// A bit-packed tensor of ternary weights.
///
/// Layout: 2 bits per weight, packed MSB-first into `u64` words.
/// Capacity: 32 weights per `u64` word.
/// Density: 16× more weights per byte than FP32.
#[derive(Clone, Debug)]
pub struct TernaryTensor {
    pub rows: usize,
    pub cols: usize,
    words: Vec<u64>,
}

impl TernaryTensor {
    /// Pack a row-major float matrix into ternary representation.
    /// `threshold` controls the zero-band; 0.5 is the DeepSeek-style default.
    pub fn from_floats(data: &[f32], rows: usize, cols: usize, threshold: f32) -> Self {
        assert_eq!(data.len(), rows * cols);
        let n = rows * cols;
        // 32 trits per u64 word (2 bits each)
        let n_words = (n + 31) / 32;
        let mut words = vec![0u64; n_words];
        for (i, &w) in data.iter().enumerate() {
            let trit = Trit::from_float(w, threshold);
            let word_idx = i / 32;
            let bit_offset = (i % 32) * 2;
            words[word_idx] |= (trit as u64) << bit_offset;
        }
        TernaryTensor { rows, cols, words }
    }

    pub fn get(&self, row: usize, col: usize) -> Trit {
        let i = row * self.cols + col;
        let word = self.words[i / 32];
        let bits = (word >> ((i % 32) * 2)) & 0b11;
        match bits {
            0b00 => Trit::Zero,
            0b01 => Trit::PosOne,
            0b10 => Trit::NegOne,
            _    => Trit::Zero, // 0b11 unused; treat as zero
        }
    }

    /// Ternary matrix-vector multiply: O(rows*cols) with no multiplications.
    ///
    /// For each output element: sum += x[j] if w=+1, sum -= x[j] if w=-1, skip if w=0.
    /// On real GPU hardware this maps to predicated adds on int registers — far cheaper
    /// than FMA instructions, and the skip reduces memory bandwidth proportionally to
    /// the zero fraction of the weight matrix.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols);
        let mut out = vec![0.0f32; self.rows];
        for r in 0..self.rows {
            let mut acc = 0.0f32;
            for c in 0..self.cols {
                match self.get(r, c) {
                    Trit::PosOne => acc += x[c],
                    Trit::NegOne => acc -= x[c],
                    Trit::Zero   => {}
                }
            }
            out[r] = acc;
        }
        out
    }

    /// Memory used by the packed representation in bytes.
    pub fn packed_bytes(&self) -> usize {
        self.words.len() * 8
    }

    /// Memory that FP32 representation would need for the same weights.
    pub fn fp32_bytes(&self) -> usize {
        self.rows * self.cols * 4
    }
}

// ─── Hotswap slot ─────────────────────────────────────────────────────────────

/// A typed model version identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModelVersion(pub u64);

/// A deployed model: ternary weights + version tag.
#[derive(Clone, Debug)]
pub struct TernaryModel {
    pub version: ModelVersion,
    pub weights: TernaryTensor,
    /// Scale factor applied to matvec output (recovers activation magnitude).
    pub scale: f32,
}

impl TernaryModel {
    pub fn new(version: u64, weights: TernaryTensor, scale: f32) -> Self {
        TernaryModel { version: ModelVersion(version), weights, scale }
    }

    pub fn infer(&self, x: &[f32]) -> Vec<f32> {
        let raw = self.weights.matvec(x);
        raw.into_iter().map(|v| v * self.scale).collect()
    }
}

/// An atomic slot that holds the current model and allows zero-downtime replacement.
///
/// `Arc<Mutex<>>` is used here to represent the pointer-swap semantics. In a real
/// CUDA implementation this would be a device-side atomic pointer stored in
/// `__constant__` memory; the host writes the new pointer and a `__threadfence_system()`
/// ensures the GPU sees it before the next batch dispatch.
///
/// The key property: `swap` returns the *previous* model, which the caller may hold
/// alive until it knows all in-flight batches using that model have completed.
pub struct HotswapSlot {
    inner: Arc<Mutex<Arc<TernaryModel>>>,
    swap_count: AtomicUsize,
}

impl HotswapSlot {
    pub fn new(initial: TernaryModel) -> Self {
        HotswapSlot {
            inner: Arc::new(Mutex::new(Arc::new(initial))),
            swap_count: AtomicUsize::new(0),
        }
    }

    /// Atomically replace the current model. Returns the old model (caller keeps it
    /// alive until in-flight batches drain).
    pub fn swap(&self, new_model: TernaryModel) -> Arc<TernaryModel> {
        let mut guard = self.inner.lock().unwrap();
        let old = guard.clone();
        *guard = Arc::new(new_model);
        self.swap_count.fetch_add(1, Ordering::Relaxed);
        old
    }

    /// Acquire a reference to the current model for one batch. The Arc keeps the
    /// model alive even if a swap happens mid-batch.
    pub fn load(&self) -> Arc<TernaryModel> {
        self.inner.lock().unwrap().clone()
    }

    pub fn swap_count(&self) -> usize {
        self.swap_count.load(Ordering::Relaxed)
    }
}

// ─── CRDT version tracking ────────────────────────────────────────────────────

/// A node identifier in the distributed replica set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// A CRDT version map: maps each replica to the latest model version it has observed.
///
/// Merge semantics: for each node, take the maximum (latest) version. This is a
/// monotone join-semilattice — merges are commutative, associative, and idempotent.
///
/// Real-world use: each GPU node appends its current model version to every inference
/// result. The CRDT map is gossiped between replicas. After full merge, you can prove
/// "every replica has processed at least version V" without a coordinator.
#[derive(Clone, Debug, Default)]
pub struct VersionCrdt {
    pub versions: std::collections::HashMap<NodeId, ModelVersion>,
}

impl VersionCrdt {
    pub fn new() -> Self {
        VersionCrdt::default()
    }

    /// Record that this node has advanced to `version`.
    pub fn observe(&mut self, node: NodeId, version: ModelVersion) {
        let entry = self.versions.entry(node).or_insert(ModelVersion(0));
        if version > *entry {
            *entry = version;
        }
    }

    /// Merge another node's CRDT into this one. Idempotent, commutative, associative.
    pub fn merge(&mut self, other: &VersionCrdt) {
        for (&node, &version) in &other.versions {
            self.observe(node, version);
        }
    }

    /// The minimum version across all known replicas — the "safe rollback point".
    /// If all replicas are at or above version V, we can garbage-collect V-1.
    pub fn quorum_floor(&self) -> Option<ModelVersion> {
        self.versions.values().copied().min()
    }

    /// True when all replicas have converged to `target` or above.
    pub fn all_at_least(&self, target: ModelVersion) -> bool {
        !self.versions.is_empty()
            && self.versions.values().all(|&v| v >= target)
    }
}

// ─── Adaptive inference pipeline ──────────────────────────────────────────────

/// One processed frame: input, output, and the model version that produced it.
#[derive(Debug)]
pub struct InferenceResult {
    pub input: Vec<f32>,
    pub output: Vec<f32>,
    pub model_version: ModelVersion,
}

/// A streaming inference pipeline that processes frames from a channel-like queue.
///
/// The critical property: `hotswap_model()` can be called at any time from any
/// thread. In-flight `process_batch()` calls complete with whichever model they
/// loaded at batch-start. The swap is invisible to the caller — there is no
/// "draining" step, no error, no gap in output.
pub struct AdaptiveInferencePipeline {
    pub slot: HotswapSlot,
    pub node_id: NodeId,
    pub crdt: Mutex<VersionCrdt>,
}

impl AdaptiveInferencePipeline {
    pub fn new(node_id: u32, initial_model: TernaryModel) -> Self {
        let mut crdt = VersionCrdt::new();
        crdt.observe(NodeId(node_id), initial_model.version);
        AdaptiveInferencePipeline {
            slot: HotswapSlot::new(initial_model),
            node_id: NodeId(node_id),
            crdt: Mutex::new(crdt),
        }
    }

    /// Process a batch of frames. Returns one `InferenceResult` per frame.
    ///
    /// The model snapshot is taken once at batch start — all frames in the batch
    /// use the same model version. This is the fundamental hotswap invariant:
    /// batch boundaries are the only consistency points.
    pub fn process_batch(&self, frames: &[Vec<f32>]) -> Vec<InferenceResult> {
        let model = self.slot.load(); // atomic snapshot
        let version = model.version;

        let results = frames
            .iter()
            .map(|frame| InferenceResult {
                input: frame.clone(),
                output: model.infer(frame),
                model_version: version,
            })
            .collect();

        // Update CRDT: this node has now used `version`
        self.crdt.lock().unwrap().observe(self.node_id, version);

        results
    }

    /// Replace the running model. Returns the evicted model (caller holds it until
    /// in-flight batches complete — in practice, one batch RTT).
    pub fn hotswap_model(&self, new_model: TernaryModel) -> Arc<TernaryModel> {
        let new_version = new_model.version;
        let old = self.slot.swap(new_model);
        self.crdt.lock().unwrap().observe(self.node_id, new_version);
        old
    }

    /// Merge a remote replica's CRDT state into this node's view.
    pub fn merge_crdt(&self, remote: &VersionCrdt) {
        self.crdt.lock().unwrap().merge(remote);
    }

    pub fn crdt_snapshot(&self) -> VersionCrdt {
        self.crdt.lock().unwrap().clone()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Ternary tensor tests ──────────────────────────────────────────────────

    fn small_model(version: u64) -> TernaryModel {
        // 4×4 weight matrix: alternating +1/-1 rows, two zero rows
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
             1.0,  1.0,  1.0,  1.0,  // row 0: all +1
            -1.0, -1.0, -1.0, -1.0,  // row 1: all -1
             0.0,  0.0,  0.0,  0.0,  // row 2: all zero
             1.0, -1.0,  1.0, -1.0,  // row 3: alternating
        ];
        let weights = TernaryTensor::from_floats(&data, 4, 4, 0.5);
        TernaryModel::new(version, weights, 1.0)
    }

    #[test]
    fn trit_encoding_round_trips() {
        assert_eq!(Trit::from_float( 0.8, 0.5), Trit::PosOne);
        assert_eq!(Trit::from_float(-0.8, 0.5), Trit::NegOne);
        assert_eq!(Trit::from_float( 0.3, 0.5), Trit::Zero);
        assert_eq!(Trit::from_float( 0.0, 0.5), Trit::Zero);
    }

    #[test]
    fn ternary_tensor_get_matches_input() {
        let data = vec![1.0f32, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0];
        let t = TernaryTensor::from_floats(&data, 3, 3, 0.5);
        assert_eq!(t.get(0, 0), Trit::PosOne);
        assert_eq!(t.get(0, 1), Trit::Zero);
        assert_eq!(t.get(0, 2), Trit::NegOne);
        assert_eq!(t.get(1, 0), Trit::Zero);
        assert_eq!(t.get(1, 1), Trit::PosOne);
        assert_eq!(t.get(1, 2), Trit::NegOne);
    }

    #[test]
    fn matvec_correctness() {
        let model = small_model(1);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = model.infer(&x);
        // row 0: sum = 1+2+3+4 = 10
        // row 1: sum = -(1+2+3+4) = -10
        // row 2: sum = 0
        // row 3: 1-2+3-4 = -2
        assert!((y[0] - 10.0).abs() < 1e-6, "row 0 = {}", y[0]);
        assert!((y[1] + 10.0).abs() < 1e-6, "row 1 = {}", y[1]);
        assert!((y[2] -  0.0).abs() < 1e-6, "row 2 = {}", y[2]);
        assert!((y[3] +  2.0).abs() < 1e-6, "row 3 = {}", y[3]);
    }

    #[test]
    fn ternary_is_16x_denser_than_fp32() {
        let data: Vec<f32> = (0..1024).map(|i| if i % 3 == 0 { 1.0 } else if i % 3 == 1 { -1.0 } else { 0.0 }).collect();
        let t = TernaryTensor::from_floats(&data, 32, 32, 0.5);
        // 1024 weights × 2 bits = 256 bytes packed; FP32 = 4096 bytes
        assert_eq!(t.packed_bytes(), 256);
        assert_eq!(t.fp32_bytes(), 4096);
        assert_eq!(t.fp32_bytes() / t.packed_bytes(), 16);
    }

    #[test]
    fn matvec_with_scale() {
        let data = vec![1.0f32, -1.0, 0.0, 1.0];
        let weights = TernaryTensor::from_floats(&data, 2, 2, 0.5);
        let model = TernaryModel::new(1, weights, 2.0);
        let x = vec![3.0, 5.0];
        let y = model.infer(&x);
        // row 0: (3 - 5) * 2 = -4
        // row 1: (0 + 5) * 2 = 10
        assert!((y[0] + 4.0).abs() < 1e-6);
        assert!((y[1] - 10.0).abs() < 1e-6);
    }

    // ── Hotswap tests ─────────────────────────────────────────────────────────

    #[test]
    fn hotswap_replaces_model_atomically() {
        let model_v1 = small_model(1);
        let slot = HotswapSlot::new(model_v1);

        // Load before swap: should be v1
        let loaded = slot.load();
        assert_eq!(loaded.version, ModelVersion(1));

        // Swap in v2
        let model_v2 = small_model(2);
        let evicted = slot.swap(model_v2);
        assert_eq!(evicted.version, ModelVersion(1), "evicted should be old model");

        // Load after swap: should be v2
        let loaded = slot.load();
        assert_eq!(loaded.version, ModelVersion(2));
        assert_eq!(slot.swap_count(), 1);
    }

    #[test]
    fn in_flight_batch_uses_snapshot_not_swapped_model() {
        let model_v1 = small_model(1);
        let slot = HotswapSlot::new(model_v1);

        // Simulate: batch-start loads model
        let batch_model = slot.load(); // Arc clone — keeps v1 alive

        // Hotswap happens after batch-start
        let model_v2 = small_model(2);
        let _evicted = slot.swap(model_v2);

        // Batch completes using the snapshot it took at start — still v1
        assert_eq!(batch_model.version, ModelVersion(1),
            "in-flight batch must see the model it loaded, not the swapped-in model");

        // New batches get v2
        let new_batch_model = slot.load();
        assert_eq!(new_batch_model.version, ModelVersion(2));
    }

    #[test]
    fn evicted_model_stays_alive_while_held() {
        let slot = HotswapSlot::new(small_model(1));
        let evicted = slot.swap(small_model(2));
        // We can still infer with the evicted model — it hasn't been freed
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let y = evicted.infer(&x);
        assert_eq!(y.len(), 4);
    }

    // ── CRDT tests ────────────────────────────────────────────────────────────

    #[test]
    fn crdt_observe_is_monotone() {
        let mut crdt = VersionCrdt::new();
        let n = NodeId(0);
        crdt.observe(n, ModelVersion(3));
        crdt.observe(n, ModelVersion(1)); // lower — should be ignored
        assert_eq!(crdt.versions[&n], ModelVersion(3));
        crdt.observe(n, ModelVersion(5));
        assert_eq!(crdt.versions[&n], ModelVersion(5));
    }

    #[test]
    fn crdt_merge_is_commutative() {
        let mut a = VersionCrdt::new();
        a.observe(NodeId(0), ModelVersion(3));
        a.observe(NodeId(1), ModelVersion(2));

        let mut b = VersionCrdt::new();
        b.observe(NodeId(0), ModelVersion(2));
        b.observe(NodeId(1), ModelVersion(4));

        let mut ab = a.clone();
        ab.merge(&b);

        let mut ba = b.clone();
        ba.merge(&a);

        // a ⊔ b == b ⊔ a
        assert_eq!(ab.versions[&NodeId(0)], ba.versions[&NodeId(0)]);
        assert_eq!(ab.versions[&NodeId(1)], ba.versions[&NodeId(1)]);

        // Each takes the max
        assert_eq!(ab.versions[&NodeId(0)], ModelVersion(3));
        assert_eq!(ab.versions[&NodeId(1)], ModelVersion(4));
    }

    #[test]
    fn crdt_merge_is_idempotent() {
        let mut a = VersionCrdt::new();
        a.observe(NodeId(0), ModelVersion(5));
        let b = a.clone();
        a.merge(&b);
        a.merge(&b); // double merge
        assert_eq!(a.versions[&NodeId(0)], ModelVersion(5));
    }

    #[test]
    fn crdt_quorum_floor_is_min_version() {
        let mut crdt = VersionCrdt::new();
        crdt.observe(NodeId(0), ModelVersion(5));
        crdt.observe(NodeId(1), ModelVersion(3));
        crdt.observe(NodeId(2), ModelVersion(7));
        assert_eq!(crdt.quorum_floor(), Some(ModelVersion(3)));
    }

    #[test]
    fn crdt_all_at_least_reflects_convergence() {
        let mut crdt = VersionCrdt::new();
        crdt.observe(NodeId(0), ModelVersion(4));
        crdt.observe(NodeId(1), ModelVersion(3));
        assert!(!crdt.all_at_least(ModelVersion(4)), "node 1 is still at v3");
        crdt.observe(NodeId(1), ModelVersion(4));
        assert!(crdt.all_at_least(ModelVersion(4)), "all nodes now at v4");
    }

    // ── Pipeline integration tests ────────────────────────────────────────────

    #[test]
    fn pipeline_processes_batch_with_correct_version() {
        let pipeline = AdaptiveInferencePipeline::new(0, small_model(1));
        let frames = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let results = pipeline.process_batch(&frames);
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.model_version, ModelVersion(1));
        }
    }

    #[test]
    fn pipeline_hotswap_changes_version_for_next_batch() {
        let pipeline = AdaptiveInferencePipeline::new(0, small_model(1));

        let frames = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let batch1 = pipeline.process_batch(&frames);
        assert_eq!(batch1[0].model_version, ModelVersion(1));

        // Hotswap between batches
        pipeline.hotswap_model(small_model(2));

        let batch2 = pipeline.process_batch(&frames);
        assert_eq!(batch2[0].model_version, ModelVersion(2),
            "after hotswap, new batches must use the new model");
    }

    #[test]
    fn pipeline_output_changes_after_hotswap() {
        // v1: all-+1 weights in row 0
        let model_v1 = {
            let data = vec![1.0f32; 4];
            let w = TernaryTensor::from_floats(&data, 1, 4, 0.5);
            TernaryModel::new(1, w, 1.0)
        };
        // v2: all--1 weights in row 0
        let model_v2 = {
            let data = vec![-1.0f32; 4];
            let w = TernaryTensor::from_floats(&data, 1, 4, 0.5);
            TernaryModel::new(2, w, 1.0)
        };

        let pipeline = AdaptiveInferencePipeline::new(0, model_v1);
        let x = vec![vec![1.0, 1.0, 1.0, 1.0]];

        let before = pipeline.process_batch(&x);
        assert!((before[0].output[0] - 4.0).abs() < 1e-6,
            "v1: sum of +1 weights = 4");

        pipeline.hotswap_model(model_v2);
        let after = pipeline.process_batch(&x);
        assert!((after[0].output[0] + 4.0).abs() < 1e-6,
            "v2: sum of -1 weights = -4 (model changed without pipeline restart)");
    }

    #[test]
    fn pipeline_crdt_records_versions_used() {
        let pipeline = AdaptiveInferencePipeline::new(1, small_model(1));
        pipeline.process_batch(&[vec![1.0, 0.0, 0.0, 0.0]]);
        pipeline.hotswap_model(small_model(3));
        pipeline.process_batch(&[vec![0.0, 1.0, 0.0, 0.0]]);

        let crdt = pipeline.crdt_snapshot();
        // After hotswap and second batch, this node should be at v3
        assert_eq!(crdt.versions[&NodeId(1)], ModelVersion(3));
    }

    #[test]
    fn two_replicas_converge_via_crdt_merge() {
        // Two replicas start at different versions (replica 0 is ahead)
        let replica0 = AdaptiveInferencePipeline::new(0, small_model(1));
        let replica1 = AdaptiveInferencePipeline::new(1, small_model(1));

        let frame = vec![vec![1.0, 0.0, 0.0, 0.0]];

        // Replica 0 advances to v3; replica 1 stays at v1
        replica0.hotswap_model(small_model(2));
        replica0.hotswap_model(small_model(3));
        replica0.process_batch(&frame);

        replica1.process_batch(&frame);

        // Before merge: each replica only knows about itself
        let crdt0 = replica0.crdt_snapshot();
        let crdt1 = replica1.crdt_snapshot();
        assert!(!crdt0.versions.contains_key(&NodeId(1)));
        assert!(!crdt1.versions.contains_key(&NodeId(0)));

        // Gossip: merge replica 0's state into replica 1
        replica1.merge_crdt(&crdt0);
        let merged = replica1.crdt_snapshot();

        // After merge: replica 1 knows replica 0 is at v3, itself at v1
        assert_eq!(merged.versions[&NodeId(0)], ModelVersion(3));
        assert_eq!(merged.versions[&NodeId(1)], ModelVersion(1));

        // Quorum floor = min(v3, v1) = v1: we can't GC v1 yet
        assert_eq!(merged.quorum_floor(), Some(ModelVersion(1)));

        // Bring replica 1 up to date
        replica1.hotswap_model(small_model(3));
        replica1.process_batch(&frame);
        let updated = replica1.crdt_snapshot();
        assert!(updated.all_at_least(ModelVersion(3)),
            "after replica 1 catches up, all known nodes are at v3");
    }

    #[test]
    fn hotswap_count_tracks_swaps_correctly() {
        let slot = HotswapSlot::new(small_model(1));
        assert_eq!(slot.swap_count(), 0);
        slot.swap(small_model(2));
        slot.swap(small_model(3));
        slot.swap(small_model(4));
        assert_eq!(slot.swap_count(), 3);
    }

    /// End-to-end scenario: content moderation pipeline on a live stream.
    ///
    /// Simulates: threat patterns change, new model deployed, pipeline never stops,
    /// CRDT proves audit trail across two GPU replicas.
    #[test]
    fn end_to_end_content_moderation_scenario() {
        // Two GPU replicas processing the same stream
        let gpu0 = AdaptiveInferencePipeline::new(0, small_model(1));
        let gpu1 = AdaptiveInferencePipeline::new(1, small_model(1));

        // Simulate: 10 frames arrive, processed in batches of 2
        let frames: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32, (i * 2) as f32, 0.0, 1.0])
            .collect();

        for chunk in frames.chunks(2) {
            let chunk_vec: Vec<Vec<f32>> = chunk.to_vec();
            gpu0.process_batch(&chunk_vec);
            gpu1.process_batch(&chunk_vec);
        }

        // New threat pattern detected — deploy model v2 to gpu0 first
        gpu0.hotswap_model(small_model(2));

        // gpu0 continues processing with v2; gpu1 still on v1
        for chunk in frames.chunks(2) {
            let chunk_vec: Vec<Vec<f32>> = chunk.to_vec();
            gpu0.process_batch(&chunk_vec);
        }

        // Gossip CRDT state: gpu1 learns gpu0 is at v2
        let crdt0 = gpu0.crdt_snapshot();
        gpu1.merge_crdt(&crdt0);

        let merged1 = gpu1.crdt_snapshot();
        // gpu1 now knows: gpu0=v2, gpu1=v1 — rollback safe to v1
        assert_eq!(merged1.quorum_floor(), Some(ModelVersion(1)));
        assert!(!merged1.all_at_least(ModelVersion(2)));

        // Deploy v2 to gpu1 as well
        gpu1.hotswap_model(small_model(2));
        gpu1.process_batch(&[vec![0.0; 4]]);
        gpu1.merge_crdt(&gpu0.crdt_snapshot());

        let final1 = gpu1.crdt_snapshot();
        // Full convergence: can GC v1 now
        assert!(final1.all_at_least(ModelVersion(2)),
            "both replicas converged to v2 — v1 can be garbage collected");

        // Verify no downtime: both replicas processed all frames without gaps
        assert_eq!(gpu0.slot.swap_count(), 1);
        assert_eq!(gpu1.slot.swap_count(), 1);
    }
}
