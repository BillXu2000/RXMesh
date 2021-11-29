#pragma once

#include "rxmesh/kernels/rxmesh_query_dispatcher.cuh"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_context.h"
#include "rxmesh/util/vector.h"

/**
 * update_step()
 */
template <typename T>
__device__ __inline__ T update_step(
    const uint32_t                    v0_id,
    const uint32_t                    v1_id,
    const uint32_t                    v2_id,
    const RXMESH::RXMeshAttribute<T>& geo_distance,
    const RXMESH::RXMeshAttribute<T>& coords,
    const T                           infinity_val)
{
    using namespace RXMESH;
    const Vector<3, T> v0(coords(v0_id, 0), coords(v0_id, 1), coords(v0_id, 2));
    const Vector<3, T> v1(coords(v1_id, 0), coords(v1_id, 1), coords(v1_id, 2));
    const Vector<3, T> v2(coords(v2_id, 0), coords(v2_id, 1), coords(v2_id, 2));
    const Vector<3, T> x0 = v1 - v0;
    const Vector<3, T> x1 = v2 - v0;

    T t[2];
    t[0] = geo_distance(v1_id);
    t[1] = geo_distance(v2_id);

    T q[2][2];

    q[0][0] = dot(x0, x0);
    q[0][1] = dot(x0, x1);
    q[1][0] = dot(x1, x0);
    q[1][1] = dot(x1, x1);


    T det = q[0][0] * q[1][1] - q[0][1] * q[1][0];
    T Q[2][2];
    Q[0][0] = q[1][1] / det;
    Q[0][1] = -q[0][1] / det;
    Q[1][0] = -q[1][0] / det;
    Q[1][1] = q[0][0] / det;

    T delta = t[0] * (Q[0][0] + Q[1][0]) + t[1] * (Q[0][1] + Q[1][1]);
    T dis = delta * delta -
            (Q[0][0] + Q[0][1] + Q[1][0] + Q[1][1]) *
                (t[0] * t[0] * Q[0][0] + t[0] * t[1] * (Q[1][0] + Q[0][1]) +
                 t[1] * t[1] * Q[1][1] - 1);
    T p = (delta + std::sqrt(dis)) / (Q[0][0] + Q[0][1] + Q[1][0] + Q[1][1]);
    T tp[2];
    tp[0] = t[0] - 1;
    tp[1] = t[1] - 1;
    const Vector<3, T> n = (x0 * Q[0][0] + x1 * Q[1][0]) * tp[0] +
                           (x0 * Q[0][1] + x1 * Q[1][1]) * tp[1];
    T cond[2];
    cond[0] = dot(x0, n);
    cond[1] = dot(x1, n);

    T c[2];
    c[0] = cond[0] * Q[0][0] + cond[1] * Q[0][1];
    c[1] = cond[0] * Q[1][0] + cond[1] * Q[1][1];

    if (t[0] == infinity_val || t[1] == infinity_val || dis < 0 || c[0] >= 0 ||
        c[1] >= 0) {
        T dp[2];
        dp[0] = geo_distance(v1_id) + x0.norm();
        dp[1] = geo_distance(v2_id) + x1.norm();
        p = dp[dp[1] < dp[0]];
    }
    return p;
}


template <typename T, uint32_t blockThreads>
__launch_bounds__(blockThreads) __global__ static void relax_ptp_rxmesh(
    const RXMESH::RXMeshContext             context,
    const RXMESH::RXMeshAttribute<T>        coords,
    RXMESH::RXMeshAttribute<T>              new_geo_dist,
    const RXMESH::RXMeshAttribute<T>        old_geo_dist,
    const RXMESH::RXMeshAttribute<uint32_t> toplesets,
    const uint32_t                          band_start,
    const uint32_t                          band_end,
    uint32_t*                               d_error,
    const T                                 infinity_val,
    const T                                 error_tol)
{
    using namespace RXMESH;

    auto in_active_set = [&](uint32_t p_id) {
        uint32_t my_band = toplesets(p_id);
        return my_band >= band_start && my_band < band_end;
    };

    auto geo_lambda = [&](uint32_t p_id, RXMeshIterator& iter) {
        // this vertex (p_id) update_band
        uint32_t my_band = toplesets(p_id);

        // this is the last vertex in the one-ring (before r_id)
        uint32_t q_id = iter.back();

        // one-ring enumeration
        T current_dist = old_geo_dist(p_id);
        T new_dist = current_dist;
        for (uint32_t v = 0; v < iter.size(); ++v) {
            // the current one ring vertex
            uint32_t r_id = iter[v];

            T dist = update_step(p_id, q_id, r_id, old_geo_dist, coords,
                                 infinity_val);
            if (dist < new_dist) {
                new_dist = dist;
            }
            q_id = r_id;
        }

        new_geo_dist(p_id) = new_dist;
        // update our distance
        if (my_band == band_start) {
            T error = fabs(new_dist - current_dist) / current_dist;
            if (error < error_tol) {
                atomicAdd(d_error, 1);
            }
        }
    };


    query_block_dispatcher<Op::VV, blockThreads>(context, geo_lambda,
                                                 in_active_set, true);
}


namespace bx2k {
    template <typename T, uint32_t blockThreads>
    __launch_bounds__(blockThreads) __global__ static void relax_ptp_rxmesh(
        const RXMESH::RXMeshContext             context,
        const RXMESH::RXMeshAttribute<T>        coords,
        RXMESH::RXMeshAttribute<T>              new_geo_dist,
        const RXMESH::RXMeshAttribute<T>        old_geo_dist,
        const RXMESH::RXMeshAttribute<uint32_t> toplesets,
        const uint32_t                          band_start,
        const uint32_t                          band_end,
        uint32_t*                               d_error,
        const T                                 infinity_val,
        const T                                 error_tol)
    {
        using namespace RXMESH;

        auto in_active_set = [&](uint32_t p_id) {
            uint32_t my_band = toplesets(p_id);
            return my_band >= band_start && my_band < band_end;
        };

        auto geo_lambda = [&](uint32_t p_id, RXMeshIterator& iter) {
            // this vertex (p_id) update_band
            uint32_t my_band = toplesets(p_id);
            if (!in_active_set(p_id)) return;

            // this is the last vertex in the one-ring (before r_id)
            uint32_t q_id = iter.back();

            // one-ring enumeration
            T current_dist = old_geo_dist(p_id);
            T new_dist = current_dist;
            for (uint32_t v = 0; v < iter.size(); ++v) {
                // the current one ring vertex
                uint32_t r_id = iter[v];

                T dist = update_step(p_id, q_id, r_id, old_geo_dist, coords,
                                    infinity_val);
                if (dist < new_dist) {
                    new_dist = dist;
                }
                q_id = r_id;
            }

            new_geo_dist(p_id) = new_dist;
            // update our distance
            if (my_band == band_start) {
                T error = fabs(new_dist - current_dist) / current_dist;
                if (error < error_tol) {
                    atomicAdd(d_error, 1);
                }
            }
        };


        query_block_dispatcher<Op::VV, blockThreads>(context, geo_lambda);
    }
}