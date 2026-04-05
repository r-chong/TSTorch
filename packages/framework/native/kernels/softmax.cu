extern "C" __global__
void softmax_forward_f32(float* out, const float* x, int outer, int dim_size, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total) return;

    int o = idx / inner;
    int j = idx % inner;

    float max_val = x[o * dim_size * inner + j];
    for (int d = 1; d < dim_size; d++) {
        float v = x[(o * dim_size + d) * inner + j];
        if (v > max_val) max_val = v;
    }

    float sum = 0.0f;
    for (int d = 0; d < dim_size; d++) {
        float e = expf(x[(o * dim_size + d) * inner + j] - max_val);
        out[(o * dim_size + d) * inner + j] = e;
        sum += e;
    }

    float inv_sum = 1.0f / sum;
    for (int d = 0; d < dim_size; d++) {
        out[(o * dim_size + d) * inner + j] *= inv_sum;
    }
}

extern "C" __global__
void softmax_backward_f32(float* dx, const float* dy, const float* out, int outer, int dim_size, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total) return;

    int o = idx / inner;
    int j = idx % inner;

    float dot = 0.0f;
    for (int d = 0; d < dim_size; d++) {
        int pos = (o * dim_size + d) * inner + j;
        dot += dy[pos] * out[pos];
    }

    for (int d = 0; d < dim_size; d++) {
        int pos = (o * dim_size + d) * inner + j;
        dx[pos] = out[pos] * (dy[pos] - dot);
    }
}
