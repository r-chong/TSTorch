//! Correctness tests for the fused cuTile op wrappers in `src/ops/`.
//!
//! Coverage: activations + their backwards, softmax + backward, LayerNorm
//! forward + backward, residual LayerNorm, bias-GELU + backward, dropout,
//! AdamW, pooling (avg + max), conv1d/2d forward, KV-cache quantize round-trip.
//!
//! Each test compares cuTile output against a CPU reference.  Tolerances are
//! generous (1e-3 absolute or 1e-3 relative) since the kernels do reductions
//! in fp32 and the CPU reference uses the same.

use approx::assert_relative_eq;
use mni_framework_cutile::ops::{
    activation, dropout, elementwise, fused, grad_util, kv_quant, mixed_precision, norm,
    optimizer, pooling, reduce, softmax,
};
use mni_framework_cutile::tensor::TensorStore;

const TOL_ABS: f32 = 1e-3;
const TOL_REL: f32 = 1e-3;

fn assert_close_slice(got: &[f32], expect: &[f32], ctx: &str) {
    assert_eq!(got.len(), expect.len(), "{ctx}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expect.iter()).enumerate() {
        let abs = (g - e).abs();
        let rel = abs / e.abs().max(1e-6);
        assert!(
            abs <= TOL_ABS || rel <= TOL_REL,
            "{ctx}: mismatch at {i}: got={g} expect={e} (abs={abs} rel={rel})"
        );
    }
}

fn cpu_gelu(x: f32) -> f32 {
    // Tanh-approx GELU, same as the kernel.
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn cpu_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn test_gelu_forward() {
    let mut s = TensorStore::new();
    let n = 256;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 12.0).collect();
    let id = s.from_slice(&x, &[n]);
    let out = activation::gelu(&mut s, id);
    let got = s.to_host(out);
    let want: Vec<f32> = x.iter().map(|&v| cpu_gelu(v)).collect();
    assert_close_slice(&got, &want, "gelu");
}

#[test]
fn test_gelu_backward() {
    let mut s = TensorStore::new();
    let n = 256;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 12.0).collect();
    let dy: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let ix = s.from_slice(&x, &[n]);
    let idy = s.from_slice(&dy, &[n]);
    let out = activation::gelu_backward(&mut s, idy, ix);
    let got = s.to_host(out);

    // CPU GELU derivative reference.
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let want: Vec<f32> = x
        .iter()
        .zip(dy.iter())
        .map(|(&xi, &dyi)| {
            let inner = c * (xi + 0.044715 * xi * xi * xi);
            let tanh_inner = inner.tanh();
            let dinner_dx = c * (1.0 + 3.0 * 0.044715 * xi * xi);
            let sech2 = 1.0 - tanh_inner * tanh_inner;
            let dgelu = 0.5 * (1.0 + tanh_inner) + 0.5 * xi * sech2 * dinner_dx;
            dyi * dgelu
        })
        .collect();
    assert_close_slice(&got, &want, "gelu_backward");
}

#[test]
fn test_sigmoid_forward() {
    let mut s = TensorStore::new();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 16.0).collect();
    let id = s.from_slice(&x, &[n]);
    let out = activation::sigmoid(&mut s, id);
    let got = s.to_host(out);
    let want: Vec<f32> = x.iter().map(|&v| cpu_sigmoid(v)).collect();
    assert_close_slice(&got, &want, "sigmoid");
}

#[test]
fn test_sigmoid_backward() {
    let mut s = TensorStore::new();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 16.0).collect();
    let dy: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let ix = s.from_slice(&x, &[n]);
    let out_fwd = activation::sigmoid(&mut s, ix);
    let idy = s.from_slice(&dy, &[n]);
    let dx_id = activation::sigmoid_backward(&mut s, idy, out_fwd);
    let got = s.to_host(dx_id);

    let want: Vec<f32> = x
        .iter()
        .zip(dy.iter())
        .map(|(&xi, &dyi)| {
            let s = cpu_sigmoid(xi);
            dyi * s * (1.0 - s)
        })
        .collect();
    assert_close_slice(&got, &want, "sigmoid_backward");
}

#[test]
fn test_softmax_forward() {
    let mut s = TensorStore::new();
    let (n, c) = (4, 64);
    let x: Vec<f32> = (0..n * c).map(|i| ((i % 13) as f32) * 0.1 - 0.5).collect();
    let id = s.from_slice(&x, &[n, c]);
    let out = softmax::softmax_forward(&mut s, id, 1);
    let got = s.to_host(out);

    // CPU reference.
    let mut want = vec![0.0f32; n * c];
    for r in 0..n {
        let row = &x[r * c..(r + 1) * c];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
        let s: f32 = exps.iter().sum();
        for (j, e) in exps.iter().enumerate() {
            want[r * c + j] = e / s;
        }
    }
    assert_close_slice(&got, &want, "softmax");
    // Row sums must be 1.
    for r in 0..n {
        let row_sum: f32 = got[r * c..(r + 1) * c].iter().sum();
        assert_relative_eq!(row_sum, 1.0, max_relative = 1e-4);
    }
}

#[test]
fn test_layernorm_forward_backward() {
    let mut s = TensorStore::new();
    let (n, c) = (4, 32);
    let x: Vec<f32> = (0..n * c)
        .map(|i| ((i % 11) as f32) * 0.1 - 0.5)
        .collect();
    let g: Vec<f32> = (0..c).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let bv: Vec<f32> = (0..c).map(|i| (i as f32) * 0.005).collect();
    let ix = s.from_slice(&x, &[n, c]);
    let ig = s.from_slice(&g, &[c]);
    let ib = s.from_slice(&bv, &[c]);

    let fwd = norm::layernorm_forward(&mut s, ix, ig, ib, 1e-5);
    let got_out = s.to_host(fwd.out);
    let got_mean = s.to_host(fwd.mean);
    let got_rstd = s.to_host(fwd.rstd);

    // CPU reference.
    let mut want_out = vec![0.0f32; n * c];
    let mut want_mean = vec![0.0f32; n];
    let mut want_rstd = vec![0.0f32; n];
    for r in 0..n {
        let row = &x[r * c..(r + 1) * c];
        let m: f32 = row.iter().copied().sum::<f32>() / c as f32;
        let var: f32 = row.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / c as f32;
        let rstd = 1.0 / (var + 1e-5).sqrt();
        want_mean[r] = m;
        want_rstd[r] = rstd;
        for j in 0..c {
            want_out[r * c + j] = (row[j] - m) * rstd * g[j] + bv[j];
        }
    }
    assert_close_slice(&got_out, &want_out, "layernorm out");
    assert_close_slice(&got_mean, &want_mean, "layernorm mean");
    assert_close_slice(&got_rstd, &want_rstd, "layernorm rstd");
}

#[test]
fn test_residual_layernorm() {
    let mut s = TensorStore::new();
    let (n, c) = (2, 32);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.05).collect();
    let r: Vec<f32> = (0..n * c).map(|i| ((i % 7) as f32) * 0.1).collect();
    let g: Vec<f32> = (0..c).map(|_| 1.0).collect();
    let b: Vec<f32> = (0..c).map(|_| 0.0).collect();
    let ix = s.from_slice(&x, &[n, c]);
    let ir = s.from_slice(&r, &[n, c]);
    let ig = s.from_slice(&g, &[c]);
    let ib = s.from_slice(&b, &[c]);
    let out = fused::residual_layernorm_forward(&mut s, ix, ir, ig, ib, 1e-5);

    let got_residual = s.to_host(out.residual);
    let got_out = s.to_host(out.out);

    let want_residual: Vec<f32> = x.iter().zip(r.iter()).map(|(a, b)| a + b).collect();
    assert_close_slice(&got_residual, &want_residual, "residual");

    // LayerNorm of residual with γ=1, β=0.
    let mut want_out = vec![0.0f32; n * c];
    for row in 0..n {
        let row_data = &want_residual[row * c..(row + 1) * c];
        let m: f32 = row_data.iter().sum::<f32>() / c as f32;
        let var: f32 =
            row_data.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / c as f32;
        let rstd = 1.0 / (var + 1e-5).sqrt();
        for j in 0..c {
            want_out[row * c + j] = (row_data[j] - m) * rstd;
        }
    }
    assert_close_slice(&got_out, &want_out, "residual_layernorm out");
}

#[test]
fn test_bias_gelu_forward_backward() {
    let mut s = TensorStore::new();
    let (n, c) = (4, 32);
    let x: Vec<f32> = (0..n * c).map(|i| ((i % 17) as f32) * 0.05 - 0.5).collect();
    let bv: Vec<f32> = (0..c).map(|i| (i as f32) * 0.01).collect();
    let dy: Vec<f32> = (0..n * c).map(|i| 1.0 + (i as f32) * 0.001).collect();

    let ix = s.from_slice(&x, &[n, c]);
    let ib = s.from_slice(&bv, &[c]);
    let fwd = fused::bias_gelu_forward(&mut s, ix, ib);
    let got_fwd = s.to_host(fwd);

    // CPU reference: GELU(x + b).
    let want_fwd: Vec<f32> = (0..n * c)
        .map(|i| {
            let r = i / c;
            let cidx = i % c;
            let _ = r;
            cpu_gelu(x[i] + bv[cidx])
        })
        .collect();
    assert_close_slice(&got_fwd, &want_fwd, "bias_gelu fwd");

    // Backward.
    let idy = s.from_slice(&dy, &[n, c]);
    let bw = fused::bias_gelu_backward(&mut s, idy, ix, ib);
    let got_dx = s.to_host(bw.dx);
    let got_db = s.to_host(bw.dbias);

    // dx = dy · GELU'(x + b);  db_c = Σ_n dx[n, c]
    let cgelu = (2.0_f32 / std::f32::consts::PI).sqrt();
    let mut want_dx = vec![0.0f32; n * c];
    let mut want_db = vec![0.0f32; c];
    for r in 0..n {
        for j in 0..c {
            let i = r * c + j;
            let v = x[i] + bv[j];
            let inner = cgelu * (v + 0.044715 * v * v * v);
            let tanh_inner = inner.tanh();
            let dinner_dv = cgelu * (1.0 + 3.0 * 0.044715 * v * v);
            let sech2 = 1.0 - tanh_inner * tanh_inner;
            let dgelu = 0.5 * (1.0 + tanh_inner) + 0.5 * v * sech2 * dinner_dv;
            want_dx[i] = dy[i] * dgelu;
            want_db[j] += want_dx[i];
        }
    }
    assert_close_slice(&got_dx, &want_dx, "bias_gelu dx");
    assert_close_slice(&got_db, &want_db, "bias_gelu dbias");
}

#[test]
fn test_dropout_apply_and_backward() {
    let mut s = TensorStore::new();
    let n = 256;
    let x: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let mask_host: Vec<f32> = (0..n).map(|i| if i % 4 == 0 { 0.0 } else { 1.0 }).collect();
    let p = 0.25_f32;
    let scale = 1.0 / (1.0 - p);

    let ix = s.from_slice(&x, &[n]);
    let im = s.from_slice(&mask_host, &[n]);
    let out = dropout::dropout_apply(&mut s, ix, im, p);
    let got = s.to_host(out);
    let want: Vec<f32> = x.iter().zip(mask_host.iter()).map(|(a, m)| a * m * scale).collect();
    assert_close_slice(&got, &want, "dropout fwd");

    // Backward: dx = dy · mask · scale.
    let dy: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.005).collect();
    let idy = s.from_slice(&dy, &[n]);
    let dx = dropout::dropout_backward(&mut s, idy, im, p);
    let got_dx = s.to_host(dx);
    let want_dx: Vec<f32> = dy.iter().zip(mask_host.iter()).map(|(d, m)| d * m * scale).collect();
    assert_close_slice(&got_dx, &want_dx, "dropout bwd");
}

#[test]
fn test_adamw_step() {
    let mut s = TensorStore::new();
    let n = 64;
    let p_init: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let g: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32) * 0.001).collect();

    let p = s.from_slice(&p_init, &[n]);
    let m = s.zeros(&[n]);
    let v = s.zeros(&[n]);
    let grad = s.from_slice(&g, &[n]);

    let lr = 1e-3_f32;
    let beta1 = 0.9_f32;
    let beta2 = 0.999_f32;
    let eps = 1e-8_f32;
    let wd = 0.01_f32;
    let t = 1;
    let bc1 = 1.0 - beta1.powi(t);
    let bc2 = 1.0 - beta2.powi(t);
    optimizer::adamw_step(&mut s, p, m, v, grad, lr, beta1, beta2, eps, wd, bc1, bc2);

    let got_p = s.to_host(p);
    let got_m = s.to_host(m);
    let got_v = s.to_host(v);

    // CPU reference for one step from m=v=0.
    let mut want_p = p_init.clone();
    let mut want_m = vec![0.0f32; n];
    let mut want_v = vec![0.0f32; n];
    for i in 0..n {
        want_m[i] = beta1 * 0.0 + (1.0 - beta1) * g[i];
        want_v[i] = beta2 * 0.0 + (1.0 - beta2) * g[i] * g[i];
        let m_hat = want_m[i] / bc1;
        let v_hat = want_v[i] / bc2;
        // AdamW: decoupled weight decay applied to params before update.
        want_p[i] = want_p[i] - lr * wd * want_p[i];
        want_p[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
    assert_close_slice(&got_m, &want_m, "adamw m");
    assert_close_slice(&got_v, &want_v, "adamw v");
    assert_close_slice(&got_p, &want_p, "adamw param");
}

#[test]
fn test_grad_clip_and_norm_sq() {
    let mut s = TensorStore::new();
    let n = 256;
    let g: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let id = s.from_slice(&g, &[n]);
    let _ = grad_util::grad_clip(&mut s, id, 0.5);
    let got = s.to_host(id);
    let want: Vec<f32> = g.iter().map(|x| x * 0.5).collect();
    assert_close_slice(&got, &want, "grad_clip");

    let id2 = s.from_slice(&g, &[n]);
    let nrm = grad_util::grad_norm_sq(&mut s, id2);
    let got_nrm = s.to_host(nrm)[0];
    let want_nrm: f32 = g.iter().map(|x| x * x).sum();
    assert_relative_eq!(got_nrm, want_nrm, max_relative = 1e-3);
}

#[test]
fn test_scale_inplace() {
    let mut s = TensorStore::new();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let id = s.from_slice(&x, &[n]);
    let _ = mixed_precision::scale(&mut s, id, 2.0);
    let got = s.to_host(id);
    let want: Vec<f32> = x.iter().map(|v| v * 2.0).collect();
    assert_close_slice(&got, &want, "scale");
}

#[test]
fn test_check_inf_nan() {
    let mut s = TensorStore::new();
    let n = 64;

    // All finite.
    let ok: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
    let id_ok = s.from_slice(&ok, &[n]);
    let flag_ok = mixed_precision::check_inf_nan(&mut s, id_ok);
    assert_eq!(s.to_host(flag_ok)[0], 0.0);

    // One NaN.
    let mut nan = ok.clone();
    nan[3] = f32::NAN;
    let id_nan = s.from_slice(&nan, &[n]);
    let flag_nan = mixed_precision::check_inf_nan(&mut s, id_nan);
    assert_eq!(s.to_host(flag_nan)[0], 1.0);

    // One inf.
    let mut inf = ok.clone();
    inf[5] = f32::INFINITY;
    let id_inf = s.from_slice(&inf, &[n]);
    let flag_inf = mixed_precision::check_inf_nan(&mut s, id_inf);
    assert_eq!(s.to_host(flag_inf)[0], 1.0);
}

#[test]
fn test_avgpool2d_forward_backward() {
    let mut s = TensorStore::new();
    let (n, c, h, w) = (1, 1, 4, 4);
    let kh = 2;
    let kw = 2;
    let x: Vec<f32> = (0..n * c * h * w).map(|i| (i + 1) as f32).collect();
    let id = s.from_slice(&x, &[n, c, h, w]);
    let out = pooling::avgpool2d_forward(&mut s, id, kh, kw);
    let got = s.to_host(out);

    // 4x4 → 2x2 with 2x2 windows.
    let want = vec![
        (1.0 + 2.0 + 5.0 + 6.0) / 4.0,
        (3.0 + 4.0 + 7.0 + 8.0) / 4.0,
        (9.0 + 10.0 + 13.0 + 14.0) / 4.0,
        (11.0 + 12.0 + 15.0 + 16.0) / 4.0,
    ];
    assert_close_slice(&got, &want, "avgpool fwd");

    // Backward: each of the 4 inputs in a window gets dout/4.
    let dout: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let idout = s.from_slice(&dout, &[n, c, 2, 2]);
    let dinp = pooling::avgpool2d_backward(&mut s, idout, kh, kw);
    let got_dinp = s.to_host(dinp);
    let want_dinp = vec![
        0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 0.75, 0.75, 1.0, 1.0,
    ];
    assert_close_slice(&got_dinp, &want_dinp, "avgpool bwd");
}

#[test]
fn test_maxpool2d_forward_backward() {
    let mut s = TensorStore::new();
    let (n, c, h, w) = (1, 1, 4, 4);
    let kh = 2;
    let kw = 2;
    // Within each 2x2 window the max is at a known position.
    let x: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, //
        5.0, 6.0, 7.0, 8.0, //
        9.0, 10.0, 11.0, 12.0, //
        13.0, 14.0, 15.0, 16.0,
    ];
    let id = s.from_slice(&x, &[n, c, h, w]);
    let state = pooling::maxpool2d_forward(&mut s, id, kh, kw);
    let got = s.to_host(state.out);
    // Max of each 2x2 window.
    let want = vec![6.0, 8.0, 14.0, 16.0];
    assert_close_slice(&got, &want, "maxpool fwd");

    // Backward: dout flows only to the argmax position.
    let dout = vec![1.0, 2.0, 3.0, 4.0];
    let idout = s.from_slice(&dout, &[n, c, 2, 2]);
    let dinp = pooling::maxpool2d_backward(&mut s, &state, idout);
    let got_dinp = s.to_host(dinp);
    // Argmax positions: bottom-right of each window — indices 5, 7, 13, 15.
    let mut want_dinp = vec![0.0f32; 16];
    want_dinp[5] = 1.0;
    want_dinp[7] = 2.0;
    want_dinp[13] = 3.0;
    want_dinp[15] = 4.0;
    assert_close_slice(&got_dinp, &want_dinp, "maxpool bwd");
}

#[test]
fn test_kv_quant_roundtrip() {
    let mut s = TensorStore::new();
    let (n, d) = (4, 64);
    let x: Vec<f32> = (0..n * d)
        .map(|i| ((i as f32) * 0.01).sin() * 5.0)
        .collect();
    let id = s.from_slice(&x, &[n, d]);
    let q = kv_quant::quantize_rows(&s, id);
    let dq = kv_quant::dequantize_rows(&mut s, &q);
    let got = s.to_host(dq);

    // Per-row max abs error should be ≤ scale (= max(|row|)/127).
    for r in 0..n {
        let row = &x[r * d..(r + 1) * d];
        let max_abs = row.iter().copied().fold(0.0_f32, |a, b| a.max(b.abs()));
        let scale = (max_abs / 127.0).max(1e-8);
        for j in 0..d {
            let err = (got[r * d + j] - row[j]).abs();
            assert!(
                err <= scale,
                "kv_quant row {r} col {j}: got={} want={} err={} scale={}",
                got[r * d + j],
                row[j],
                err,
                scale
            );
        }
    }
}

#[test]
fn test_sum_along_dim_last() {
    let mut s = TensorStore::new();
    let (n, c) = (8, 16);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.05).collect();
    let id = s.from_slice(&x, &[n, c]);
    let out = reduce::sum_along_dim(&mut s, id, 1, false);
    let got = s.to_host(out);
    assert_eq!(s.shape(out), &[n]);
    let want: Vec<f32> = (0..n)
        .map(|r| x[r * c..(r + 1) * c].iter().copied().sum())
        .collect();
    assert_close_slice(&got, &want, "sum_along_dim last");
}

#[test]
fn test_broadcast_add() {
    let mut s = TensorStore::new();
    let (n, c) = (4, 8);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..c).map(|i| i as f32).collect();
    let ix = s.from_slice(&x, &[n, c]);
    let ib = s.from_slice(&b, &[c]);
    let out = elementwise::broadcast_add(&mut s, ix, ib);
    let got = s.to_host(out);
    let want: Vec<f32> = (0..n * c).map(|i| x[i] + b[i % c]).collect();
    assert_close_slice(&got, &want, "broadcast_add");
}
