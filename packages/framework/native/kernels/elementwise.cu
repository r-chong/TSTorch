extern "C" __global__
void add_f32(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__
void sub_f32(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

extern "C" __global__
void mul_f32(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

extern "C" __global__
void neg_f32(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -a[i];
}

extern "C" __global__
void mul_scalar_f32(float* out, const float* a, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * s;
}

extern "C" __global__
void exp_f32(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(a[i]);
}

extern "C" __global__
void log_f32(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = logf(a[i]);
}

extern "C" __global__
void add_bias_f32(float* out, const float* a, const float* bias, int total, int bias_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = a[i] + bias[i % bias_size];
}

extern "C" __global__
void fill_f32(float* out, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = val;
}

extern "C" __global__
void div_f32(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}
