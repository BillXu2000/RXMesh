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

/*
__constant__ float  mass = std::stof(std::getenv("__mass"));
__constant__ float  stiffness = std::stof(std::getenv("__stiffness"));
__constant__ float  dt = std::stof(std::getenv("__dt"));
*/
__constant__ float  stiffness = 1e1;
__constant__ float  dt = 2e-5;

float& __host__ __device__ vis(float3 &x, int i) {
    return ((float*)(&x))[i];
}

template <typename T, uint32_t blockThreads>
__launch_bounds__(blockThreads, 6) __global__
    static void compute_mass_spring(const RXMESH::RXMeshContext      context,
                                    const RXMESH::RXMeshAttribute<T> ox,
                                    RXMESH::RXMeshAttribute<T>       x,
                                    RXMESH::RXMeshAttribute<T>       v, float mass)
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
            total_f += stiffness * (length(disp) - length(rest_disp)) *
                       normalize(disp);
        }

        for (int i = 0; i < 3; i++) {
            v(vert_id, i) -= dt * vis(total_f, i) / mass;
        }
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, ms_lambda);
}

template <typename T, uint32_t blockThreads>
__launch_bounds__(blockThreads, 6) __global__
    static void ev_mass_spring(const RXMESH::RXMeshContext      context,
                                    const RXMESH::RXMeshAttribute<T> ox,
                                    RXMESH::RXMeshAttribute<T>       x,
                                    RXMESH::RXMeshAttribute<T>       v, float mass)
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
            atomicAdd(&v(v0, i), -dt * vis(force, i) / mass);
            atomicAdd(&v(v1, i), dt * vis(force, i) / mass);
        }
    };

    query_block_dispatcher<Op::EV, blockThreads>(context, ms_lambda);
}

template <typename T, uint32_t blockThreads>
__launch_bounds__(blockThreads, 6) __global__
    static void advect(const RXMESH::RXMeshContext      context,
                                    const RXMESH::RXMeshAttribute<T> ox,
                                    RXMESH::RXMeshAttribute<T>       x,
                                    RXMESH::RXMeshAttribute<T>       v, float mass)
{
    using namespace RXMESH;
    auto ms_lambda = [&](uint32_t vert_id, RXMeshIterator& vv) {
		if (x(vert_id, 1) < 0.1) return;
        v(vert_id, 1) += 9.8 * dt;
        for (int i = 0; i < 3; i++) {
            x(vert_id, i) += dt * v(vert_id, i);
        }
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, ms_lambda);
}


/*template <typename T, uint32_t blockThreads>
__launch_bounds__(blockThreads, 6) __global__
    static void advance(const RXMESH::RXMeshContext      context,
                                    const RXMESH::RXMeshAttribute<T> ox,
                                    RXMESH::RXMeshAttribute<T>       x,
                                    RXMESH::RXMeshAttribute<T>       v)
{
    using namespace RXMESH;
    auto ms_lambda = [&](uint32_t vert_id, RXMeshIterator& vv) {
		if (x(vert_id, 1) < 0.1) return;
		for (int i = 0; i < 3; i++) {
            v(vert_id, i) += 
			x(vert_id, i) += dt * v(vert_id, i);
		}
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, ms_lambda);
}*/