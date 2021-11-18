#pragma once

#include "rxmesh/kernels/rxmesh_query_dispatcher.cuh"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_context.h"
#include "rxmesh/util/math.h"
#include "rxmesh/util/vector.h"
/**
 * mass_spring()
 */

#include <helper_cuda.h>
#include <helper_math.h>
#include <cstdlib>
#include "float33.h"

/*
__constant__ float  mass = std::stof(std::getenv("__mass"));
__constant__ float  stiffness = std::stof(std::getenv("__stiffness"));
__constant__ float  dt = std::stof(std::getenv("__dt"));
*/
const float dt = 2e-5;
__constant__ float  stiffness = 1e1;

float& __host__ __device__ vis(float3 &x, int i) {
    return ((float*)(&x))[i];
}

template <typename T, uint32_t blockThreads>
__launch_bounds__(blockThreads, 6) __global__
    static void compute_mass_spring(const RXMESH::RXMeshContext      context,
                                    const RXMESH::RXMeshAttribute<T> ox,
                                    RXMESH::RXMeshAttribute<T>       x,
                                    RXMESH::RXMeshAttribute<T>       f, float mass)
{
    using namespace RXMESH;
    auto ms_lambda = [&](uint32_t vert_id, RXMeshIterator& vv) {
		if (x(vert_id, 1) < 0.1) return;
        float3 total_f{0, 0, 0};

        for (int i = 0; i < vv.size(); ++i) {
            float3 disp{x(vert_id, 0) - x(vv[i], 0),
                        x(vert_id, 1) - x(vv[i], 1),
                        x(vert_id, 2) - x(vv[i], 2)};
            float3 rest_disp{ox(vert_id, 0) - ox(vv[i], 0),
                             ox(vert_id, 1) - ox(vv[i], 1),
                             ox(vert_id, 2) - ox(vv[i], 2)};
            total_f += -stiffness * (length(disp) - length(rest_disp)) *
                       normalize(disp);
        }

        for (int i = 0; i < 3; i++) {
            f(vert_id, i) = vis(total_f, i);
        }
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, ms_lambda);
}

float __host__ __device__ pow3(float k) {
    return k * k * k;
}

float __host__ __device__ pow2(float k) {
    return k * k;
}

template <typename T>
float3 __host__ __device__ f3(RXMESH::RXMeshAttribute<T> k, int i) {
    return make_float3(k(i, 0), k(i, 1), k(i, 2));
}

template <typename T, uint32_t blockThreads>
__launch_bounds__(blockThreads, 6) __global__
    static void matmul(const RXMESH::RXMeshContext      context,
                                    RXMESH::RXMeshAttribute<T> ret,
                                    const RXMESH::RXMeshAttribute<T> ox,
                                    RXMESH::RXMeshAttribute<T>       x,
                                    RXMESH::RXMeshAttribute<T>       vel, float mass)
{
    using namespace RXMESH;
    auto ms_lambda = [&](uint32_t u, RXMeshIterator& vv) {
        float3 sum{0, 0, 0};

        for (int z = 0; z < vv.size(); ++z) {
            int v = vv[z];
            float3 dist{x(u, 0) - x(v, 0),
                        x(u, 1) - x(v, 1),
                        x(u, 2) - x(v, 2)};
            float3 rest_disp{ox(u, 0) - ox(v, 0),
                             ox(u, 1) - ox(v, 1),
                             ox(u, 2) - ox(v, 2)};
            float l = length(rest_disp);
            float33 df;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    df(i, j) = vis(dist, i) * vis(dist, j) * l / pow3(length(dist));
                    if (i == j) df(i, j) += 1 - l / length(dist);
                    df(i, j) *= -stiffness * l;
                }
            }
            sum += mul(df, f3(vel, u) - f3(vel, v));
        }
        float3 ans = (f3(vel, u) - pow2(dt) / mass * sum);
        for (int i = 0; i < 3; i++) {
            ret(u, i) = vis(ans, i);
        }
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, ms_lambda);
}

template <typename T, uint32_t blockThreads>
__launch_bounds__(blockThreads, 6) __global__
    static void ev_mass_spring(const RXMESH::RXMeshContext      context,
                                    const RXMESH::RXMeshAttribute<T> ox,
                                    RXMESH::RXMeshAttribute<T>       x,
                                    RXMESH::RXMeshAttribute<T>       f, float mass)
{
    using namespace RXMESH;
    auto ms_lambda = [&](uint32_t edge_id, RXMeshIterator& iter) {
        int v0(iter[0]), v1(iter[1]);
        float3 disp{x(v0, 0) - x(v1, 0),
                    x(v0, 1) - x(v1, 1),
                    x(v0, 2) - x(v1, 2)};
        float3 rest_disp{ox(v0, 0) - ox(v1, 0),
                         ox(v0, 1) - ox(v1, 1),
                         ox(v0, 2) - ox(v1, 2)};
        float3 force = stiffness * (length(disp) - length(rest_disp)) * normalize(disp);
        for (int i = 0; i < 3; i++) {
            atomicAdd(&f(v0, i), -vis(force, i));
            atomicAdd(&f(v1, i), vis(force, i));
        }
    };

    query_block_dispatcher<Op::EV, blockThreads>(context, ms_lambda);
}

template <typename T, uint32_t blockThreads>
__launch_bounds__(blockThreads, 6) __global__
    static void advect(int n, const RXMESH::RXMeshContext      context,
                                    RXMESH::RXMeshAttribute<T>       x,
                                    RXMESH::RXMeshAttribute<T>       v,
                                    RXMESH::RXMeshAttribute<T>       f, float *m)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n || x(p, 1) < 0.1) return;
    v(p, 1) += 9.8 * dt;
    for (int i = 0; i < 3; i++) {
        v(p, i) += dt * f(p, i) / m[p];
        f(p, i) = 0;
        x(p, i) += dt * v(p, i);
    }
}

__global__ void vector_add(int n, float* ans, const float *x, const float k, const float *y) { // c = a + k * b
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    ans[i] = x[i] + k * y[i];
}

__global__ void vector_addon(int n, float* ans, const float k, const float *y) { // a += k * b
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    ans[i] += k * y[i];
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void dot_product(int n, float *ans, const float *x, const float *y) { // ans = a.dot(b)
    float sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += x[i] * y[i];
    }
    sum = warpReduceSum(sum);
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    static __shared__ float shared[32];
    if (lane == 0) shared[wid] = sum;
    __syncthreads();

    sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) sum = warpReduceSum(sum);
    if (threadIdx.x == 0) atomicAdd(ans, sum);
}

template<typename T>
__global__ void advance_im(int n,   RXMESH::RXMeshAttribute<T>       x,
                                    RXMESH::RXMeshAttribute<T>       v,
                                    RXMESH::RXMeshAttribute<T>       f, float mass) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    for (int i = 0; i < 3; i++) {
        f(p, i) = 0;
        if (x(p, 1) >= 0.1) {
            x(p, i) += dt * v(p, i);
        }
        else {
            v(p, i) = 0;
        }
    }
}

template<typename T>
__global__ void fix_gravity(int n, RXMESH::RXMeshAttribute<T> f, float mass) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    f(p, 1) += 9.8 * mass;
}