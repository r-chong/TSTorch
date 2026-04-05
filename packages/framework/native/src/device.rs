#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, CudaStream};
#[cfg(feature = "cuda")]
use cudarc::cublas::CudaBlas;
#[cfg(feature = "cuda")]
use cudarc::nvrtc;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

// ---------------------------------------------------------------------------
// CUDA device
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub struct GpuDevice {
    pub ctx: Arc<CudaContext>,
    pub stream: CudaStream,
    pub blas: CudaBlas,
    functions: HashMap<String, CudaFunction>,
}

#[cfg(feature = "cuda")]
static GPU: OnceLock<GpuDevice> = OnceLock::new();

#[cfg(feature = "cuda")]
impl GpuDevice {
    pub fn instance() -> &'static Self {
        GPU.get_or_init(Self::init)
    }

    fn init() -> Self {
        let ctx = CudaContext::new(0).expect("CUDA context creation failed");
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(ctx.clone()).expect("cuBLAS handle creation failed");

        let mut dev = Self {
            ctx,
            stream,
            blas,
            functions: HashMap::new(),
        };
        dev.load_all_kernels();
        dev
    }

    fn compile_and_load(&mut self, name: &str, source: &str, kernel_names: &[&str]) {
        let ptx = nvrtc::compile_ptx(source)
            .unwrap_or_else(|e| panic!("Failed to compile {name}: {e}"));
        let module = self
            .ctx
            .load_module(ptx)
            .unwrap_or_else(|e| panic!("Failed to load module {name}: {e}"));
        for &kname in kernel_names {
            let func = module
                .get_function(kname)
                .unwrap_or_else(|e| panic!("Failed to get function {kname}: {e}"));
            self.functions.insert(kname.to_string(), func);
        }
    }

    fn load_all_kernels(&mut self) {
        self.compile_and_load(
            "elementwise",
            include_str!("../kernels/elementwise.cu"),
            &[
                "add_f32",
                "sub_f32",
                "mul_f32",
                "neg_f32",
                "mul_scalar_f32",
                "exp_f32",
                "log_f32",
                "add_bias_f32",
                "fill_f32",
                "div_f32",
            ],
        );
        self.compile_and_load(
            "activation",
            include_str!("../kernels/activation.cu"),
            &[
                "gelu_forward_f32",
                "gelu_backward_f32",
                "relu_forward_f32",
                "relu_backward_f32",
            ],
        );
        self.compile_and_load(
            "reduce",
            include_str!("../kernels/reduce.cu"),
            &[
                "sum_along_dim_f32",
                "mean_along_dim_f32",
                "max_along_dim_f32",
                "sum_broadcast_f32",
            ],
        );
        self.compile_and_load(
            "layernorm",
            include_str!("../kernels/layernorm.cu"),
            &["layernorm_forward_f32", "layernorm_backward_f32"],
        );
        self.compile_and_load(
            "softmax",
            include_str!("../kernels/softmax.cu"),
            &["softmax_forward_f32", "softmax_backward_f32"],
        );
        self.compile_and_load(
            "cross_entropy",
            include_str!("../kernels/cross_entropy.cu"),
            &["cross_entropy_forward_f32", "cross_entropy_backward_f32"],
        );
        self.compile_and_load(
            "embedding",
            include_str!("../kernels/embedding.cu"),
            &["embedding_forward_f32", "embedding_backward_f32"],
        );
        self.compile_and_load(
            "dropout",
            include_str!("../kernels/dropout.cu"),
            &["dropout_apply_f32", "dropout_backward_f32"],
        );
        self.compile_and_load(
            "adamw",
            include_str!("../kernels/adamw.cu"),
            &["adamw_step_f32"],
        );
        self.compile_and_load(
            "grad_util",
            include_str!("../kernels/grad_util.cu"),
            &["grad_norm_sq_partial_f32", "grad_clip_f32"],
        );
    }

    pub fn get_func(&self, name: &str) -> &CudaFunction {
        self.functions
            .get(name)
            .unwrap_or_else(|| panic!("Kernel function '{name}' not found"))
    }
}

// ---------------------------------------------------------------------------
// CPU stub (no-op)
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu")]
pub struct GpuDevice;

#[cfg(feature = "cpu")]
impl GpuDevice {
    pub fn instance() -> &'static Self {
        static STUB: GpuDevice = GpuDevice;
        &STUB
    }
}
