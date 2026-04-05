use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore, compute_strides, shape_size};
use crate::utils::{broadcast_shape, unbroadcast, to_coord};

#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use cudarc::driver::{DevicePtr, LaunchConfig};

#[cfg(feature = "cuda")]
fn launch_cfg(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(feature = "cpu")]
fn broadcast_binary(
    a: TensorId, b: TensorId, store: &mut TensorStore,
    f: fn(f32, f32) -> f32,
) -> (Vec<f32>, Vec<usize>) {
    let a_shape = store.shape(a).to_vec();
    let b_shape = store.shape(b).to_vec();
    let out_shape = broadcast_shape(&a_shape, &b_shape);
    let out_size = shape_size(&out_shape);
    let out_strides = compute_strides(&out_shape);

    let a_data = store.to_host(a);
    let b_data = store.to_host(b);
    let a_strides = compute_strides(&a_shape);
    let b_strides = compute_strides(&b_shape);

    let mut out = vec![0.0f32; out_size];
    let ndim = out_shape.len();
    let a_off = ndim - a_shape.len();
    let b_off = ndim - b_shape.len();

    for i in 0..out_size {
        let coord = to_coord(i, &out_shape, &out_strides);
        let mut ai = 0;
        for d in 0..a_shape.len() {
            let c = if a_shape[d] == 1 { 0 } else { coord[d + a_off] };
            ai += c * a_strides[d];
        }
        let mut bi = 0;
        for d in 0..b_shape.len() {
            let c = if b_shape[d] == 1 { 0 } else { coord[d + b_off] };
            bi += c * b_strides[d];
        }
        out[i] = f(a_data[ai], b_data[bi]);
    }
    (out, out_shape)
}

// =========================================================================
// add
// =========================================================================

#[cfg(feature = "cpu")]
pub fn add(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let (data, shape) = broadcast_binary(a, b, store, |x, y| x + y);
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Add,
        output_id: out,
        input_ids: smallvec![a, b],
        saved: SavedContext::TensorsAndShape(smallvec![a, b], shape),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn add(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let a_shape = store.shape(a).to_vec();
    let b_shape = store.shape(b).to_vec();
    let out_shape = broadcast_shape(&a_shape, &b_shape);

    if a_shape == b_shape {
        let n = shape_size(&a_shape);
        let out = store.zeros(&out_shape);
        let out_ptr = *store.get(out).data.device_ptr();
        let a_ptr = *store.get(a).data.device_ptr();
        let b_ptr = *store.get(b).data.device_ptr();
        let dev = GpuDevice::instance();
        let func = dev.get_func("add_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&out_ptr)
                .arg(&a_ptr)
                .arg(&b_ptr)
                .arg(&(n as i32))
                .launch(launch_cfg(n as u32))
                .unwrap();
        }
        tape.record(TapeEntry {
            op: BackwardOp::Add,
            output_id: out,
            input_ids: smallvec![a, b],
            saved: SavedContext::TensorsAndShape(smallvec![a, b], out_shape),
        });
        out
    } else if b_shape.len() == 1 && *a_shape.last().unwrap() == b_shape[0] {
        let total = shape_size(&a_shape);
        let bias_size = b_shape[0];
        let out = store.zeros(&out_shape);
        let out_ptr = *store.get(out).data.device_ptr();
        let a_ptr = *store.get(a).data.device_ptr();
        let b_ptr = *store.get(b).data.device_ptr();
        let dev = GpuDevice::instance();
        let func = dev.get_func("add_bias_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&out_ptr)
                .arg(&a_ptr)
                .arg(&b_ptr)
                .arg(&(total as i32))
                .arg(&(bias_size as i32))
                .launch(launch_cfg(total as u32))
                .unwrap();
        }
        tape.record(TapeEntry {
            op: BackwardOp::Add,
            output_id: out,
            input_ids: smallvec![a, b],
            saved: SavedContext::TensorsAndShape(smallvec![a, b], out_shape),
        });
        out
    } else {
        let a_data = store.to_host(a);
        let b_data = store.to_host(b);
        let out_size = shape_size(&out_shape);
        let out_strides = compute_strides(&out_shape);
        let a_strides = compute_strides(&a_shape);
        let b_strides = compute_strides(&b_shape);
        let ndim = out_shape.len();
        let a_off = ndim - a_shape.len();
        let b_off = ndim - b_shape.len();

        let mut data = vec![0.0f32; out_size];
        for i in 0..out_size {
            let coord = to_coord(i, &out_shape, &out_strides);
            let mut ai = 0;
            for d in 0..a_shape.len() {
                let c = if a_shape[d] == 1 { 0 } else { coord[d + a_off] };
                ai += c * a_strides[d];
            }
            let mut bi = 0;
            for d in 0..b_shape.len() {
                let c = if b_shape[d] == 1 { 0 } else { coord[d + b_off] };
                bi += c * b_strides[d];
            }
            data[i] = a_data[ai] + b_data[bi];
        }
        let out = store.from_vec(data, &out_shape);
        tape.record(TapeEntry {
            op: BackwardOp::Add,
            output_id: out,
            input_ids: smallvec![a, b],
            saved: SavedContext::TensorsAndShape(smallvec![a, b], out_shape),
        });
        out
    }
}

// =========================================================================
// add_backward
// =========================================================================

#[cfg(feature = "cpu")]
pub fn add_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorsAndShape(ids, _) = saved {
        let a_shape = store.shape(ids[0]).to_vec();
        let b_shape = store.shape(ids[1]).to_vec();
        let grad_data = store.to_host(grad);
        let grad_shape = store.shape(grad).to_vec();

        let ga = unbroadcast(&grad_data, &grad_shape, &a_shape);
        let gb = unbroadcast(&grad_data, &grad_shape, &b_shape);
        vec![Some(store.from_vec(ga, &a_shape)), Some(store.from_vec(gb, &b_shape))]
    } else { vec![None, None] }
}

#[cfg(feature = "cuda")]
pub fn add_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorsAndShape(ids, _) = saved {
        let a_shape = store.shape(ids[0]).to_vec();
        let b_shape = store.shape(ids[1]).to_vec();
        let grad_shape = store.shape(grad).to_vec();

        if a_shape == grad_shape && b_shape == grad_shape {
            let grad_data = store.to_host(grad);
            let ga = store.from_vec(grad_data.clone(), &a_shape);
            let gb = store.from_vec(grad_data, &b_shape);
            vec![Some(ga), Some(gb)]
        } else {
            let grad_data = store.to_host(grad);
            let ga = unbroadcast(&grad_data, &grad_shape, &a_shape);
            let gb = unbroadcast(&grad_data, &grad_shape, &b_shape);
            vec![Some(store.from_vec(ga, &a_shape)), Some(store.from_vec(gb, &b_shape))]
        }
    } else { vec![None, None] }
}

// =========================================================================
// mul
// =========================================================================

#[cfg(feature = "cpu")]
pub fn mul(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let (data, shape) = broadcast_binary(a, b, store, |x, y| x * y);
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Mul,
        output_id: out,
        input_ids: smallvec![a, b],
        saved: SavedContext::Tensors(smallvec![a, b]),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn mul(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let a_shape = store.shape(a).to_vec();
    let b_shape = store.shape(b).to_vec();

    if a_shape == b_shape {
        let n = shape_size(&a_shape);
        let out = store.zeros(&a_shape);
        let out_ptr = *store.get(out).data.device_ptr();
        let a_ptr = *store.get(a).data.device_ptr();
        let b_ptr = *store.get(b).data.device_ptr();
        let dev = GpuDevice::instance();
        let func = dev.get_func("mul_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&out_ptr)
                .arg(&a_ptr)
                .arg(&b_ptr)
                .arg(&(n as i32))
                .launch(launch_cfg(n as u32))
                .unwrap();
        }
        tape.record(TapeEntry {
            op: BackwardOp::Mul,
            output_id: out,
            input_ids: smallvec![a, b],
            saved: SavedContext::Tensors(smallvec![a, b]),
        });
        out
    } else {
        let out_shape = broadcast_shape(&a_shape, &b_shape);
        let a_data = store.to_host(a);
        let b_data = store.to_host(b);
        let out_size = shape_size(&out_shape);
        let out_strides = compute_strides(&out_shape);
        let a_strides = compute_strides(&a_shape);
        let b_strides = compute_strides(&b_shape);
        let ndim = out_shape.len();
        let a_off = ndim - a_shape.len();
        let b_off = ndim - b_shape.len();

        let mut data = vec![0.0f32; out_size];
        for i in 0..out_size {
            let coord = to_coord(i, &out_shape, &out_strides);
            let mut ai = 0;
            for d in 0..a_shape.len() {
                let c = if a_shape[d] == 1 { 0 } else { coord[d + a_off] };
                ai += c * a_strides[d];
            }
            let mut bi = 0;
            for d in 0..b_shape.len() {
                let c = if b_shape[d] == 1 { 0 } else { coord[d + b_off] };
                bi += c * b_strides[d];
            }
            data[i] = a_data[ai] * b_data[bi];
        }
        let out = store.from_vec(data, &out_shape);
        tape.record(TapeEntry {
            op: BackwardOp::Mul,
            output_id: out,
            input_ids: smallvec![a, b],
            saved: SavedContext::Tensors(smallvec![a, b]),
        });
        out
    }
}

// =========================================================================
// mul_backward
// =========================================================================

#[cfg(feature = "cpu")]
pub fn mul_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensors(ids) = saved {
        let a = ids[0]; let b = ids[1];
        let a_shape = store.shape(a).to_vec();
        let b_shape = store.shape(b).to_vec();
        let grad_shape = store.shape(grad).to_vec();
        let a_data = store.to_host(a);
        let b_data = store.to_host(b);
        let grad_data = store.to_host(grad);
        let grad_size = shape_size(&grad_shape);

        let out_strides = compute_strides(&grad_shape);
        let ndim = grad_shape.len();

        let mut ga_full = vec![0.0f32; grad_size];
        let mut gb_full = vec![0.0f32; grad_size];
        let a_strides_o = compute_strides(&a_shape);
        let b_strides_o = compute_strides(&b_shape);
        let a_off = ndim - a_shape.len();
        let b_off = ndim - b_shape.len();

        for i in 0..grad_size {
            let coord = to_coord(i, &grad_shape, &out_strides);
            let mut ai = 0;
            for d in 0..a_shape.len() {
                let c = if a_shape[d] == 1 { 0 } else { coord[d + a_off] };
                ai += c * a_strides_o[d];
            }
            let mut bi = 0;
            for d in 0..b_shape.len() {
                let c = if b_shape[d] == 1 { 0 } else { coord[d + b_off] };
                bi += c * b_strides_o[d];
            }
            ga_full[i] = grad_data[i] * b_data[bi];
            gb_full[i] = grad_data[i] * a_data[ai];
        }

        let ga = unbroadcast(&ga_full, &grad_shape, &a_shape);
        let gb = unbroadcast(&gb_full, &grad_shape, &b_shape);
        vec![Some(store.from_vec(ga, &a_shape)), Some(store.from_vec(gb, &b_shape))]
    } else { vec![None, None] }
}

#[cfg(feature = "cuda")]
pub fn mul_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensors(ids) = saved {
        let a = ids[0]; let b = ids[1];
        let a_shape = store.shape(a).to_vec();
        let b_shape = store.shape(b).to_vec();
        let grad_shape = store.shape(grad).to_vec();

        if a_shape == b_shape && a_shape == grad_shape {
            let n = shape_size(&grad_shape);
            let ga = store.zeros(&a_shape);
            let gb = store.zeros(&b_shape);
            let ga_ptr = *store.get(ga).data.device_ptr();
            let gb_ptr = *store.get(gb).data.device_ptr();
            let grad_ptr = *store.get(grad).data.device_ptr();
            let a_ptr = *store.get(a).data.device_ptr();
            let b_ptr = *store.get(b).data.device_ptr();
            let dev = GpuDevice::instance();

            let func_a = dev.get_func("mul_f32");
            unsafe {
                dev.stream.launch_builder(func_a)
                    .arg(&ga_ptr)
                    .arg(&grad_ptr)
                    .arg(&b_ptr)
                    .arg(&(n as i32))
                    .launch(launch_cfg(n as u32))
                    .unwrap();
            }
            let func_b = dev.get_func("mul_f32");
            unsafe {
                dev.stream.launch_builder(func_b)
                    .arg(&gb_ptr)
                    .arg(&grad_ptr)
                    .arg(&a_ptr)
                    .arg(&(n as i32))
                    .launch(launch_cfg(n as u32))
                    .unwrap();
            }
            vec![Some(ga), Some(gb)]
        } else {
            let a_data = store.to_host(a);
            let b_data = store.to_host(b);
            let grad_data = store.to_host(grad);
            let grad_size = shape_size(&grad_shape);
            let out_strides = compute_strides(&grad_shape);
            let ndim = grad_shape.len();

            let mut ga_full = vec![0.0f32; grad_size];
            let mut gb_full = vec![0.0f32; grad_size];
            let a_strides_o = compute_strides(&a_shape);
            let b_strides_o = compute_strides(&b_shape);
            let a_off = ndim - a_shape.len();
            let b_off = ndim - b_shape.len();

            for i in 0..grad_size {
                let coord = to_coord(i, &grad_shape, &out_strides);
                let mut ai = 0;
                for d in 0..a_shape.len() {
                    let c = if a_shape[d] == 1 { 0 } else { coord[d + a_off] };
                    ai += c * a_strides_o[d];
                }
                let mut bi = 0;
                for d in 0..b_shape.len() {
                    let c = if b_shape[d] == 1 { 0 } else { coord[d + b_off] };
                    bi += c * b_strides_o[d];
                }
                ga_full[i] = grad_data[i] * b_data[bi];
                gb_full[i] = grad_data[i] * a_data[ai];
            }

            let ga = unbroadcast(&ga_full, &grad_shape, &a_shape);
            let gb = unbroadcast(&gb_full, &grad_shape, &b_shape);
            vec![Some(store.from_vec(ga, &a_shape)), Some(store.from_vec(gb, &b_shape))]
        }
    } else { vec![None, None] }
}

// =========================================================================
// sub
// =========================================================================

#[cfg(feature = "cpu")]
pub fn sub(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let (data, shape) = broadcast_binary(a, b, store, |x, y| x - y);
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Sub,
        output_id: out,
        input_ids: smallvec![a, b],
        saved: SavedContext::TensorsAndShape(smallvec![a, b], shape),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn sub(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let a_shape = store.shape(a).to_vec();
    let b_shape = store.shape(b).to_vec();

    if a_shape == b_shape {
        let n = shape_size(&a_shape);
        let out = store.zeros(&a_shape);
        let out_ptr = *store.get(out).data.device_ptr();
        let a_ptr = *store.get(a).data.device_ptr();
        let b_ptr = *store.get(b).data.device_ptr();
        let dev = GpuDevice::instance();
        let func = dev.get_func("sub_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&out_ptr)
                .arg(&a_ptr)
                .arg(&b_ptr)
                .arg(&(n as i32))
                .launch(launch_cfg(n as u32))
                .unwrap();
        }
        tape.record(TapeEntry {
            op: BackwardOp::Sub,
            output_id: out,
            input_ids: smallvec![a, b],
            saved: SavedContext::TensorsAndShape(smallvec![a, b], a_shape),
        });
        out
    } else {
        let out_shape = broadcast_shape(&a_shape, &b_shape);
        let a_data = store.to_host(a);
        let b_data = store.to_host(b);
        let out_size = shape_size(&out_shape);
        let out_strides = compute_strides(&out_shape);
        let a_strides = compute_strides(&a_shape);
        let b_strides = compute_strides(&b_shape);
        let ndim = out_shape.len();
        let a_off = ndim - a_shape.len();
        let b_off = ndim - b_shape.len();

        let mut data = vec![0.0f32; out_size];
        for i in 0..out_size {
            let coord = to_coord(i, &out_shape, &out_strides);
            let mut ai = 0;
            for d in 0..a_shape.len() {
                let c = if a_shape[d] == 1 { 0 } else { coord[d + a_off] };
                ai += c * a_strides[d];
            }
            let mut bi = 0;
            for d in 0..b_shape.len() {
                let c = if b_shape[d] == 1 { 0 } else { coord[d + b_off] };
                bi += c * b_strides[d];
            }
            data[i] = a_data[ai] - b_data[bi];
        }
        let out = store.from_vec(data, &out_shape);
        tape.record(TapeEntry {
            op: BackwardOp::Sub,
            output_id: out,
            input_ids: smallvec![a, b],
            saved: SavedContext::TensorsAndShape(smallvec![a, b], out_shape),
        });
        out
    }
}

// =========================================================================
// sub_backward
// =========================================================================

#[cfg(feature = "cpu")]
pub fn sub_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorsAndShape(ids, _) = saved {
        let a_shape = store.shape(ids[0]).to_vec();
        let b_shape = store.shape(ids[1]).to_vec();
        let grad_data = store.to_host(grad);
        let grad_shape = store.shape(grad).to_vec();

        let ga = unbroadcast(&grad_data, &grad_shape, &a_shape);
        let neg_grad: Vec<f32> = grad_data.iter().map(|x| -x).collect();
        let gb = unbroadcast(&neg_grad, &grad_shape, &b_shape);
        vec![Some(store.from_vec(ga, &a_shape)), Some(store.from_vec(gb, &b_shape))]
    } else { vec![None, None] }
}

#[cfg(feature = "cuda")]
pub fn sub_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorsAndShape(ids, _) = saved {
        let a_shape = store.shape(ids[0]).to_vec();
        let b_shape = store.shape(ids[1]).to_vec();
        let grad_shape = store.shape(grad).to_vec();

        if a_shape == grad_shape && b_shape == grad_shape {
            let n = shape_size(&grad_shape);
            let grad_data = store.to_host(grad);
            let ga = store.from_vec(grad_data, &a_shape);

            let gb = store.zeros(&b_shape);
            let gb_ptr = *store.get(gb).data.device_ptr();
            let grad_ptr = *store.get(grad).data.device_ptr();
            let dev = GpuDevice::instance();
            let func = dev.get_func("neg_f32");
            unsafe {
                dev.stream.launch_builder(func)
                    .arg(&gb_ptr)
                    .arg(&grad_ptr)
                    .arg(&(n as i32))
                    .launch(launch_cfg(n as u32))
                    .unwrap();
            }
            vec![Some(ga), Some(gb)]
        } else {
            let grad_data = store.to_host(grad);
            let ga = unbroadcast(&grad_data, &grad_shape, &a_shape);
            let neg_grad: Vec<f32> = grad_data.iter().map(|x| -x).collect();
            let gb = unbroadcast(&neg_grad, &grad_shape, &b_shape);
            vec![Some(store.from_vec(ga, &a_shape)), Some(store.from_vec(gb, &b_shape))]
        }
    } else { vec![None, None] }
}

// =========================================================================
// neg
// =========================================================================

#[cfg(feature = "cpu")]
pub fn neg(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let data: Vec<f32> = store.to_host(a).iter().map(|x| -x).collect();
    let shape = store.shape(a).to_vec();
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Neg, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::None,
    });
    out
}

#[cfg(feature = "cuda")]
pub fn neg(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let shape = store.shape(a).to_vec();
    let n = shape_size(&shape);
    let out = store.zeros(&shape);
    let out_ptr = *store.get(out).data.device_ptr();
    let a_ptr = *store.get(a).data.device_ptr();
    let dev = GpuDevice::instance();
    let func = dev.get_func("neg_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&a_ptr)
            .arg(&(n as i32))
            .launch(launch_cfg(n as u32))
            .unwrap();
    }
    tape.record(TapeEntry {
        op: BackwardOp::Neg, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::None,
    });
    out
}

// =========================================================================
// neg_backward
// =========================================================================

#[cfg(feature = "cpu")]
pub fn neg_backward(grad: TensorId, _saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    let data: Vec<f32> = store.to_host(grad).iter().map(|x| -x).collect();
    let shape = store.shape(grad).to_vec();
    vec![Some(store.from_vec(data, &shape))]
}

#[cfg(feature = "cuda")]
pub fn neg_backward(grad: TensorId, _saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    let shape = store.shape(grad).to_vec();
    let n = shape_size(&shape);
    let out = store.zeros(&shape);
    let out_ptr = *store.get(out).data.device_ptr();
    let grad_ptr = *store.get(grad).data.device_ptr();
    let dev = GpuDevice::instance();
    let func = dev.get_func("neg_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&grad_ptr)
            .arg(&(n as i32))
            .launch(launch_cfg(n as u32))
            .unwrap();
    }
    vec![Some(out)]
}

// =========================================================================
// mul_scalar
// =========================================================================

#[cfg(feature = "cpu")]
pub fn mul_scalar(a: TensorId, s: f32, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let data: Vec<f32> = store.to_host(a).iter().map(|x| x * s).collect();
    let shape = store.shape(a).to_vec();
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::MulScalar, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::TensorAndScalar(a, s),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn mul_scalar(a: TensorId, s: f32, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let shape = store.shape(a).to_vec();
    let n = shape_size(&shape);
    let out = store.zeros(&shape);
    let out_ptr = *store.get(out).data.device_ptr();
    let a_ptr = *store.get(a).data.device_ptr();
    let dev = GpuDevice::instance();
    let func = dev.get_func("mul_scalar_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&a_ptr)
            .arg(&s)
            .arg(&(n as i32))
            .launch(launch_cfg(n as u32))
            .unwrap();
    }
    tape.record(TapeEntry {
        op: BackwardOp::MulScalar, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::TensorAndScalar(a, s),
    });
    out
}

// =========================================================================
// mul_scalar_backward
// =========================================================================

#[cfg(feature = "cpu")]
pub fn mul_scalar_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorAndScalar(_, s) = saved {
        let data: Vec<f32> = store.to_host(grad).iter().map(|x| x * s).collect();
        let shape = store.shape(grad).to_vec();
        vec![Some(store.from_vec(data, &shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn mul_scalar_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorAndScalar(_, s) = saved {
        let shape = store.shape(grad).to_vec();
        let n = shape_size(&shape);
        let out = store.zeros(&shape);
        let out_ptr = *store.get(out).data.device_ptr();
        let grad_ptr = *store.get(grad).data.device_ptr();
        let dev = GpuDevice::instance();
        let func = dev.get_func("mul_scalar_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&out_ptr)
                .arg(&grad_ptr)
                .arg(s)
                .arg(&(n as i32))
                .launch(launch_cfg(n as u32))
                .unwrap();
        }
        vec![Some(out)]
    } else { vec![None] }
}

// =========================================================================
// exp
// =========================================================================

#[cfg(feature = "cpu")]
pub fn exp(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let data: Vec<f32> = store.to_host(a).iter().map(|x| x.exp()).collect();
    let shape = store.shape(a).to_vec();
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Exp, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(out),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn exp(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let shape = store.shape(a).to_vec();
    let n = shape_size(&shape);
    let out = store.zeros(&shape);
    let out_ptr = *store.get(out).data.device_ptr();
    let a_ptr = *store.get(a).data.device_ptr();
    let dev = GpuDevice::instance();
    let func = dev.get_func("exp_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&a_ptr)
            .arg(&(n as i32))
            .launch(launch_cfg(n as u32))
            .unwrap();
    }
    tape.record(TapeEntry {
        op: BackwardOp::Exp, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(out),
    });
    out
}

// =========================================================================
// exp_backward
// =========================================================================

#[cfg(feature = "cpu")]
pub fn exp_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(out) = saved {
        let out_data = store.to_host(*out);
        let grad_data = store.to_host(grad);
        let data: Vec<f32> = grad_data.iter().zip(&out_data).map(|(g, o)| g * o).collect();
        let shape = store.shape(grad).to_vec();
        vec![Some(store.from_vec(data, &shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn exp_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(exp_out) = saved {
        let shape = store.shape(grad).to_vec();
        let n = shape_size(&shape);
        let result = store.zeros(&shape);
        let result_ptr = *store.get(result).data.device_ptr();
        let grad_ptr = *store.get(grad).data.device_ptr();
        let exp_ptr = *store.get(*exp_out).data.device_ptr();
        let dev = GpuDevice::instance();
        let func = dev.get_func("mul_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&result_ptr)
                .arg(&grad_ptr)
                .arg(&exp_ptr)
                .arg(&(n as i32))
                .launch(launch_cfg(n as u32))
                .unwrap();
        }
        vec![Some(result)]
    } else { vec![None] }
}

// =========================================================================
// log
// =========================================================================

#[cfg(feature = "cpu")]
pub fn log(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let data: Vec<f32> = store.to_host(a).iter().map(|x| x.ln()).collect();
    let shape = store.shape(a).to_vec();
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Log, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(a),
    });
    out
}

#[cfg(feature = "cuda")]
pub fn log(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let shape = store.shape(a).to_vec();
    let n = shape_size(&shape);
    let out = store.zeros(&shape);
    let out_ptr = *store.get(out).data.device_ptr();
    let a_ptr = *store.get(a).data.device_ptr();
    let dev = GpuDevice::instance();
    let func = dev.get_func("log_f32");
    unsafe {
        dev.stream.launch_builder(func)
            .arg(&out_ptr)
            .arg(&a_ptr)
            .arg(&(n as i32))
            .launch(launch_cfg(n as u32))
            .unwrap();
    }
    tape.record(TapeEntry {
        op: BackwardOp::Log, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(a),
    });
    out
}

// =========================================================================
// log_backward
// =========================================================================

#[cfg(feature = "cpu")]
pub fn log_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(inp) = saved {
        let inp_data = store.to_host(*inp);
        let grad_data = store.to_host(grad);
        let data: Vec<f32> = grad_data.iter().zip(&inp_data).map(|(g, x)| g / x).collect();
        let shape = store.shape(grad).to_vec();
        vec![Some(store.from_vec(data, &shape))]
    } else { vec![None] }
}

#[cfg(feature = "cuda")]
pub fn log_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(inp) = saved {
        let shape = store.shape(grad).to_vec();
        let n = shape_size(&shape);
        let result = store.zeros(&shape);
        let result_ptr = *store.get(result).data.device_ptr();
        let grad_ptr = *store.get(grad).data.device_ptr();
        let inp_ptr = *store.get(*inp).data.device_ptr();
        let dev = GpuDevice::instance();
        let func = dev.get_func("div_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&result_ptr)
                .arg(&grad_ptr)
                .arg(&inp_ptr)
                .arg(&(n as i32))
                .launch(launch_cfg(n as u32))
                .unwrap();
        }
        vec![Some(result)]
    } else { vec![None] }
}
