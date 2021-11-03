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
__constant__ float  dt = 2e-4;

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
            total_f += -stiffness * (length(disp) - length(rest_disp)) *
                       normalize(disp);
        }

        v(vert_id, 0) += dt * total_f.x / mass;
        v(vert_id, 1) += dt * (total_f.y / mass + 9.8);
        v(vert_id, 2) += dt * total_f.z / mass;
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