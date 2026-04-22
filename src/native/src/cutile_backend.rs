//! N-API entry points that route through the cuTile backend.
//!
//! Enabled by the `cutile` cargo feature.  Each export is prefixed with
//! `cutile_` so it coexists with the existing native (cudarc / wgpu / cpu)
//! exports in the same `.node` module — JS callers pick the backend by
//! choosing which family of functions to call.
//!
//! The cuTile backend keeps its own engine state (a separate `TensorStore`
//! living entirely on the GPU), so cuTile-backed tensor IDs and native
//! tensor IDs are NOT interchangeable.  Cross-backend transfers go through
//! `cutile_to_float32` / `cutile_from_float32` host round-trips, the same
//! pattern used between any two distinct backends.

use mni_framework_cutile::ops;
use mni_framework_cutile::tensor::{TensorId, TensorStore};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::Mutex;
use std::sync::OnceLock;

struct CutileEngine {
    store: TensorStore,
}

static ENGINE: OnceLock<Mutex<CutileEngine>> = OnceLock::new();

fn engine() -> &'static Mutex<CutileEngine> {
    ENGINE.get_or_init(|| {
        Mutex::new(CutileEngine {
            store: TensorStore::new(),
        })
    })
}

// ---------------------------------------------------------------------------
// Tensor lifecycle
// ---------------------------------------------------------------------------

#[napi]
pub fn cutile_zeros(shape: Vec<i64>) -> u32 {
    let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    engine().lock().store.zeros(&shape) as u32
}

#[napi]
pub fn cutile_ones(shape: Vec<i64>) -> u32 {
    let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    engine().lock().store.ones(&shape) as u32
}

#[napi]
pub fn cutile_rand(shape: Vec<i64>) -> u32 {
    let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    engine().lock().store.rand(&shape) as u32
}

#[napi]
pub fn cutile_randn(shape: Vec<i64>) -> u32 {
    let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    engine().lock().store.randn(&shape) as u32
}

#[napi]
pub fn cutile_from_float32(data: Float32Array, shape: Vec<i64>) -> u32 {
    let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    engine().lock().store.from_slice(data.as_ref(), &shape) as u32
}

#[napi]
pub fn cutile_to_float32(id: u32) -> Float32Array {
    let data = engine().lock().store.to_host(id as TensorId);
    Float32Array::new(data)
}

#[napi]
pub fn cutile_shape(id: u32) -> Vec<i64> {
    engine()
        .lock()
        .store
        .shape(id as TensorId)
        .iter()
        .map(|&s| s as i64)
        .collect()
}

#[napi]
pub fn cutile_size(id: u32) -> i64 {
    engine().lock().store.size(id as TensorId) as i64
}

#[napi]
pub fn cutile_get_scalar(id: u32) -> f64 {
    engine().lock().store.get_scalar(id as TensorId) as f64
}

#[napi]
pub fn cutile_free(id: u32) {
    engine().lock().store.free(id as TensorId);
}

// ---------------------------------------------------------------------------
// Elementwise ops
// ---------------------------------------------------------------------------

#[napi]
pub fn cutile_add(a: u32, b: u32) -> u32 {
    let mut eng = engine().lock();
    ops::elementwise::add(&mut eng.store, a as TensorId, b as TensorId) as u32
}

#[napi]
pub fn cutile_sub(a: u32, b: u32) -> u32 {
    let mut eng = engine().lock();
    ops::elementwise::sub(&mut eng.store, a as TensorId, b as TensorId) as u32
}

#[napi]
pub fn cutile_mul(a: u32, b: u32) -> u32 {
    let mut eng = engine().lock();
    ops::elementwise::mul(&mut eng.store, a as TensorId, b as TensorId) as u32
}

#[napi]
pub fn cutile_div(a: u32, b: u32) -> u32 {
    let mut eng = engine().lock();
    ops::elementwise::div(&mut eng.store, a as TensorId, b as TensorId) as u32
}

#[napi]
pub fn cutile_neg(a: u32) -> u32 {
    let mut eng = engine().lock();
    ops::elementwise::neg(&mut eng.store, a as TensorId) as u32
}

#[napi]
pub fn cutile_mul_scalar(a: u32, s: f64) -> u32 {
    let mut eng = engine().lock();
    ops::elementwise::mul_scalar(&mut eng.store, a as TensorId, s as f32) as u32
}

#[napi]
pub fn cutile_saxpy(a: f64, x: u32, y: u32) -> u32 {
    let mut eng = engine().lock();
    ops::elementwise::saxpy(&mut eng.store, a as f32, x as TensorId, y as TensorId) as u32
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

#[napi]
pub fn cutile_relu(a: u32) -> u32 {
    let mut eng = engine().lock();
    ops::activation::relu(&mut eng.store, a as TensorId) as u32
}

#[napi]
pub fn cutile_relu_backward(x: u32, grad: u32) -> u32 {
    let mut eng = engine().lock();
    ops::activation::relu_backward(&mut eng.store, x as TensorId, grad as TensorId) as u32
}

// ---------------------------------------------------------------------------
// MatMul + reductions
// ---------------------------------------------------------------------------

#[napi]
pub fn cutile_matmul(a: u32, b: u32) -> u32 {
    let mut eng = engine().lock();
    ops::matmul::matmul(&mut eng.store, a as TensorId, b as TensorId) as u32
}

#[napi]
pub fn cutile_sum_all(a: u32) -> u32 {
    let mut eng = engine().lock();
    ops::reduce::sum_all(&mut eng.store, a as TensorId) as u32
}

#[napi]
pub fn cutile_mean_all(a: u32) -> u32 {
    let mut eng = engine().lock();
    ops::reduce::mean_all(&mut eng.store, a as TensorId) as u32
}
