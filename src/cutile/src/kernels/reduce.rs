//! Reduction cuTile kernels.
//!
//! Two flavors of reduction:
//!
//! 1. **Global reductions** (`sum_block`) drive both passes of the CUB-style
//!    strict two-pass global reduction — see `ops/reduce.rs`.  The last
//!    block in pass 1 and the sole block in pass 2 handle the "partial
//!    tile" case automatically: cuTile's `Tensor::partition()` returns a
//!    zero-padded view, so out-of-range tile lanes load the additive
//!    identity 0.0f32 and the reduction stays correct for arbitrary `n`.
//!
//! 2. **Along-dim reductions** (`sum_along_last`, `mean_along_last`,
//!    `max_along_last`) are 1-block-per-output-row kernels that collapse
//!    the last axis of a 2D tile `[BM, DIM]` → `[BM]`.  The general 3D
//!    `(outer, dim, inner)` case in the CUDA kernels is handled in the ops
//!    layer by permuting the reduction axis to be last before launch
//!    (matches the `dim == -1` fast path).
//!
//!    `broadcast_last` is the inverse: expands `[BM]` across a new last
//!    dim of size `DIM` — the forward of the sum-backward pattern
//!    (`sum_broadcast_f32` in the CUDA backend).

#[cutile::module]
pub mod reduce_kernels {
    use cutile::core::*;

    /// Reduce a `BLOCK`-sized tile of `x` to a single scalar via
    /// `reduce_sum`, writing it at `pid.0` in `z`.
    #[cutile::entry()]
    pub fn sum_block<const BLOCK: i32>(
        z: &mut Tensor<f32, { [1] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part_x: Partition<f32, { [BLOCK] }> = x.partition(const_shape![BLOCK]);
        let tile_x: Tile<f32, { [BLOCK] }> = part_x.load([pid.0]);
        let s_scalar: Tile<f32, { [] }> = reduce_sum(tile_x, 0i32);
        let s_one: Tile<f32, { [1] }> = s_scalar.reshape(const_shape![1]);
        z.store(s_one);
    }

    /// `out[r] = Σⱼ x[r, j]`.  One block per row tile.
    #[cutile::entry()]
    pub fn sum_along_last<const BM: i32, const DIM: i32>(
        out: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [BM, DIM] }> = x.partition(const_shape![BM, DIM]);
        let tx: Tile<f32, { [BM, DIM] }> = part.load([pid.0, 0i32]);
        let s: Tile<f32, { [BM] }> = reduce_sum(tx, 1i32);
        out.store(s);
    }

    /// `out[r] = Σⱼ x[r, j] / DIM`.
    #[cutile::entry()]
    pub fn mean_along_last<const BM: i32, const DIM: i32>(
        out: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [BM, DIM] }> = x.partition(const_shape![BM, DIM]);
        let tx: Tile<f32, { [BM, DIM] }> = part.load([pid.0, 0i32]);
        let s: Tile<f32, { [BM] }> = reduce_sum(tx, 1i32);
        let inv_s: f32 = 1.0f32 / (DIM as f32);
        let inv: Tile<f32, { [BM] }> = inv_s.broadcast(out.shape());
        out.store(s * inv);
    }

    /// `out[r] = maxⱼ x[r, j]`.
    #[cutile::entry()]
    pub fn max_along_last<const BM: i32, const DIM: i32>(
        out: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [BM, DIM] }> = x.partition(const_shape![BM, DIM]);
        let tx: Tile<f32, { [BM, DIM] }> = part.load([pid.0, 0i32]);
        let s: Tile<f32, { [BM] }> = reduce_max(tx, 1i32);
        out.store(s);
    }

    /// `out[r, j] = x[r]`.  Broadcast a per-row scalar across a new last
    /// dim of size `DIM` — forward of sum-backward.
    #[cutile::entry()]
    pub fn broadcast_last<const BM: i32, const DIM: i32>(
        out: &mut Tensor<f32, { [BM, DIM] }>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [BM] }> = x.partition(const_shape![BM]);
        let tx: Tile<f32, { [BM] }> = part.load([pid.0]);
        let tx_b: Tile<f32, { [BM, DIM] }> = tx
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, DIM]);
        out.store(tx_b);
    }

    /// `out[i, j] = Σᵢ g[i, j]` — sum along dim 0 of a `[ROWS, BN]` tile.
    /// Used by the ops layer to fold per-row `dgamma`/`dbeta` / `dbias`
    /// partials into their final shape.
    #[cutile::entry()]
    pub fn sum_along_first<const ROWS: i32, const BN: i32>(
        out: &mut Tensor<f32, { [BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part: Partition<f32, { [ROWS, BN] }> = x.partition(const_shape![ROWS, BN]);
        let tx: Tile<f32, { [ROWS, BN] }> = part.load([0i32, pid.0]);
        let s: Tile<f32, { [BN] }> = reduce_sum(tx, 0i32);
        out.store(s);
    }
}

pub use reduce_kernels::{
    broadcast_last, max_along_last, mean_along_last, sum_along_first, sum_along_last, sum_block,
};
