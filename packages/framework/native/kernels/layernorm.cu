extern "C" __global__
void layernorm_forward_f32(float* out, float* mean_out, float* rstd_out,
                           const float* x, const float* gamma, const float* beta,
                           int n, int c, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    const float* row_x = x + row * c;
    float* row_out = out + row * c;

    float sum = 0.0f;
    for (int j = 0; j < c; j++) sum += row_x[j];
    float mean = sum / (float)c;

    float var_sum = 0.0f;
    for (int j = 0; j < c; j++) {
        float diff = row_x[j] - mean;
        var_sum += diff * diff;
    }
    float rstd = rsqrtf(var_sum / (float)c + eps);

    for (int j = 0; j < c; j++) {
        row_out[j] = gamma[j] * (row_x[j] - mean) * rstd + beta[j];
    }

    if (mean_out) mean_out[row] = mean;
    if (rstd_out) rstd_out[row] = rstd;
}

extern "C" __global__
void layernorm_backward_f32(float* dx, float* dgamma, float* dbeta,
                            const float* dy, const float* x,
                            const float* mean, const float* rstd, const float* gamma,
                            int n, int c) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    const float* row_dy = dy + row * c;
    const float* row_x = x + row * c;
    float* row_dx = dx + row * c;
    float m = mean[row];
    float r = rstd[row];

    float dot_dy_xhat = 0.0f;
    float dot_dy = 0.0f;
    for (int j = 0; j < c; j++) {
        float xhat = (row_x[j] - m) * r;
        dot_dy_xhat += row_dy[j] * gamma[j] * xhat;
        dot_dy += row_dy[j] * gamma[j];
    }

    float inv_c = 1.0f / (float)c;
    for (int j = 0; j < c; j++) {
        float xhat = (row_x[j] - m) * r;
        row_dx[j] = r * (row_dy[j] * gamma[j] - inv_c * (dot_dy + xhat * dot_dy_xhat));
    }

    for (int j = 0; j < c; j++) {
        float xhat = (row_x[j] - m) * r;
        atomicAdd(&dgamma[j], row_dy[j] * xhat);
        atomicAdd(&dbeta[j], row_dy[j]);
    }
}
