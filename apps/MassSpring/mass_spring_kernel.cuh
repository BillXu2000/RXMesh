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

__constant__ float3 g = {0.f, 9.8f, 0.f};
__constant__ float  mass = std::stof(std::getenv("__mass"));
__constant__ float  stiffness = std::stof(std::getenv("__stiffness"));
__constant__ float  dt = std::stof(std::getenv("__dt"));

template <typename T, uint32_t blockThreads>
__launch_bounds__(blockThreads, 6) __global__
    static void compute_mass_spring(const RXMESH::RXMeshContext      context,
                                    const RXMESH::RXMeshAttribute<T> ox,
                                    RXMESH::RXMeshAttribute<T>       x,
                                    RXMESH::RXMeshAttribute<T>       v)
{
    using namespace RXMESH;
    auto ms_lambda = [&](uint32_t vert_id, RXMeshIterator& vv) {
        float3 total_f = g;

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

        v(vert_id, 0) += dt * total_f.x / mass;
        v(vert_id, 1) += dt * total_f.y / mass;
        v(vert_id, 2) += dt * total_f.z / mass;
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, ms_lambda);
}
