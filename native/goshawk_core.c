/*
 * Goshawk R1 Native C99 Inference Kernels
 * Compiled with -O3 -march=native -ffast-math for AVX2/NEON auto-vectorization.
 * Zero Python / PyTorch runtime overhead; callable directly via Bun FFI.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

void goshawk_rmsnorm(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int dim,
    float eps
) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) {
        ss += x[i] * x[i];
    }
    float inv_rms = 1.0f / sqrtf((ss / (float)dim) + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

void goshawk_gemv_f32(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ out,
    int out_dim,
    int in_dim
) {
    for (int r = 0; r < out_dim; r++) {
        const float* row = W + (size_t)r * (size_t)in_dim;
        float acc = 0.0f;
        for (int c = 0; c < in_dim; c++) {
            acc += row[c] * x[c];
        }
        out[r] = acc;
    }
}

/*
 * Fused Packed INT4 GEMV:
 * Each byte in W_packed holds two 4-bit weights offset by +8:
 *   low nibble  = even column (2*b)
 *   high nibble = odd column  (2*b + 1)
 */
void goshawk_gemv_q4(
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    int out_dim,
    int in_dim
) {
    int packed_in = (in_dim + 1) / 2;
    for (int r = 0; r < out_dim; r++) {
        const uint8_t* row = W_packed + (size_t)r * (size_t)packed_in;
        float acc = 0.0f;
        int full_pairs = in_dim / 2;

        for (int b = 0; b < full_pairs; b++) {
            uint8_t byte = row[b];
            float w0 = (float)((int)(byte & 0x0F) - 8);
            float w1 = (float)((int)((byte >> 4) & 0x0F) - 8);
            acc += w0 * x[2 * b] + w1 * x[2 * b + 1];
        }
        if (in_dim & 1) {
            uint8_t byte = row[full_pairs];
            float w0 = (float)((int)(byte & 0x0F) - 8);
            acc += w0 * x[in_dim - 1];
        }
        out[r] = acc * scales[r];
    }
}

void goshawk_swiglu(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int dim
) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

int goshawk_inspect_gshk(
    const char* path,
    int32_t* out_version,
    int32_t* out_num_tensors
) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        return -1;
    }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GSHK", 4) != 0) {
        fclose(f);
        return -2;
    }

    uint32_t version = 0;
    uint32_t meta_len = 0;
    uint32_t num_tensors = 0;

    if (fread(&version, sizeof(uint32_t), 1, f) != 1 ||
        fread(&meta_len, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return -3;
    }

    if (fseek(f, (long)meta_len, SEEK_CUR) != 0) {
        fclose(f);
        return -4;
    }

    if (fread(&num_tensors, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return -5;
    }

    fclose(f);
    *out_version = (int32_t)version;
    *out_num_tensors = (int32_t)num_tensors;
    return 0;
}
