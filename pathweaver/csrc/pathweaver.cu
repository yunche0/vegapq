/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * Modified by Sukjin Kim & Assistant, 2025
 * 
 * Added true PQ distance computation with direction pruning fallback.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cstdint>
#include <iostream>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace py = pybind11;

typedef float DATA_T;
typedef uint32_t INDEX_T;
typedef float DISTANCE_T;

#define NO_PRUNE 0
#define RANDOM_PRUNE 1
#define SIGN_BIT_PRUNE 2

#define NO_THRESHOLD 0
#define USE_THRESHOLD 1
#define AGRESSIVE_THRESHOLD 2

#define NO_SEED 0
#define USE_SEED 1


constexpr int PQ_M = 8;         
constexpr int PQ_Ks = 256;       

/////////////////////////////////////////////////////// hash table ///////////////////////////////////////////////////////

__host__ __device__ inline uint32_t hashtable_getsize(const uint32_t BITLEN)
{
    return 1 << BITLEN;
}

__device__ inline void hashtable_init(INDEX_T* const table, const unsigned BITLEN, unsigned FIRST_TID = 0)
{
    if (threadIdx.x < FIRST_TID) return;
    for (uint32_t i = threadIdx.x - FIRST_TID; i < hashtable_getsize(BITLEN); i += blockDim.x - FIRST_TID)
    {
        table[i] = std::numeric_limits<INDEX_T>::max();
    }
}

__device__ inline uint32_t hashtable_insert(INDEX_T* const table, const unsigned BITLEN, const INDEX_T key)
{
    const uint32_t size     = hashtable_getsize(BITLEN);
    const uint32_t bit_mask = size - 1;
    INDEX_T index                = (key ^ (key >> BITLEN)) & bit_mask;
    constexpr uint32_t stride = 1;
    for (unsigned i = 0; i < size; i++)
    {
        const INDEX_T old = atomicCAS(&table[index], ~static_cast<INDEX_T>(0), key);
        if (old == ~static_cast<INDEX_T>(0))
            return 1;
        else if (old == key)
            return 0;
        index = (index + stride) & bit_mask;
    }
    return 0;
}

__device__ inline void hashtable_restore
(   
    INDEX_T* const table, 
    const unsigned BITLEN,
    const INDEX_T* itopk_indices,
    const uint32_t itopk_size,
    const uint32_t first_tid = 0 
)
{
    constexpr INDEX_T index_msb_1_mask = 0x80000000;
    if (threadIdx.x < first_tid) return;
    for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid)
    {
        auto key = itopk_indices[i] & ~index_msb_1_mask;
        hashtable_insert(table, BITLEN, key);
    }
}

/////////////////////////////////////////////////////// sort routines ///////////////////////////////////////////////////////

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, const unsigned lane_offset, const bool asc)
{
  auto k1 = __shfl_xor_sync(~0u, k0, lane_offset);
  auto v1 = __shfl_xor_sync(~0u, v0, lane_offset);
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    k0 = k1;
    v0 = v1;
  }
}

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, K& k1, V& v1, const bool asc)
{
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    const auto tmp_k = k0;
    k0               = k1;
    k1               = tmp_k;
    const auto tmp_v = v0;
    v0               = v1;
    v1               = tmp_v;
  }
}

template <class K, class V, unsigned _N, unsigned warp_size>
__device__ inline void warp_merge_core(K k[2], V v[2], const std::uint32_t range, const bool asc)
{
    if (_N == 2)
    {
        constexpr unsigned N = 2;
        const auto lane_id   = threadIdx.x % warp_size;

        if (range == 1) {
            const auto p = ((lane_id & 1) == 0);
            swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
            return;
        }

        const std::uint32_t b = range;
        for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
        #pragma unroll
            for (std::uint32_t i = 0; i < N; i++) {
            swap_if_needed<float, uint32_t>(k[i], v[i], c, p);
            }
        }
        const auto p = ((lane_id & b) == 0);
        swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
    }

    else if (_N == 4)
    {
        constexpr unsigned N = 4;
        const auto lane_id = threadIdx.x % warp_size;

        if (range == 1) {
            for (std::uint32_t b = 2; b <= N; b <<= 1) {
                for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
        #pragma unroll
                for (std::uint32_t i = 0; i < N; i++) {
                    std::uint32_t j = i ^ c;
                    if (i >= j) continue;
                    const auto line_id = i + (N * lane_id);
                    const auto p       = static_cast<bool>(line_id & b) == static_cast<bool>(line_id & c);
                    swap_if_needed(k[i], v[i], k[j], v[j], p);
                }
                }
            }
            return;
        }

        const std::uint32_t b = range;
        for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
        #pragma unroll
            for (std::uint32_t i = 0; i < N; i++) {
                swap_if_needed(k[i], v[i], c, p);
            }
        }
        const auto p = ((lane_id & b) == 0);
        for (std::uint32_t c = N / 2; c >= 1; c >>= 1) {
        #pragma unroll
            for (std::uint32_t i = 0; i < N; i++) {
                std::uint32_t j = i ^ c;
                if (i >= j) continue;
                swap_if_needed(k[i], v[i], k[j], v[j], p);
            }
        }
    }
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_merge(K k[N], V v[N], unsigned range, const bool asc = true)
{
  warp_merge_core<K, V, N, warp_size>(k, v, range, asc);
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_sort(K k[N], V v[N], const bool asc = true)
{
  for (std::uint32_t range = 1; range <= warp_size; range <<= 1) {
    warp_merge<K, V, N, warp_size>(k, v, range, asc);
  }
}

template <unsigned N_1, unsigned N_2>
__device__ void candidate_by_bitonic_sort
(
    INDEX_T* candidate_indices,
    DISTANCE_T* candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE
)
{
    const unsigned lane_id = threadIdx.x % 32;
    const unsigned warp_id = threadIdx.x / 32;

    if (warp_id > 0) { return; }
    if (CANDIDATE_BUFFER_SIZE != 64) 
    {
        printf("CANDIDATE_BUFFER_SIZE must be 64\n");
        assert(false);
    }
    DISTANCE_T key_1[N_1];
    INDEX_T val_1[N_1];

    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = lane_id + (32 * i);
        if (j < CANDIDATE_BUFFER_SIZE)
        {
            key_1[i] = candidate_distances[j];
            val_1[i] = candidate_indices[j];
        }
        else
        {
            key_1[i] = std::numeric_limits<DISTANCE_T>::max();
            val_1[i] = std::numeric_limits<INDEX_T>::max();
        }
    }
    warp_sort<float, uint32_t, N_1>(key_1, val_1);
    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = (N_1 * lane_id) + i;
        if (j < CANDIDATE_BUFFER_SIZE){
            candidate_distances[j] = key_1[i];
            candidate_indices[j]   = val_1[i];
        }
    }
}

template <unsigned N_1, unsigned N_2>
__device__ void candidate_by_bitonic_sort_inverse
(
    INDEX_T* candidate_indices,
    DISTANCE_T* candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE
)
{
    const unsigned lane_id = threadIdx.x % 32;
    const unsigned warp_id = threadIdx.x / 32;

    if (warp_id > 0) { return; }
    if (CANDIDATE_BUFFER_SIZE != 64) 
    {
        printf("CANDIDATE_BUFFER_SIZE must be 64\n");
        assert(false);
    }
    DISTANCE_T key_1[N_1];
    INDEX_T val_1[N_1];

    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = lane_id + (32 * i);
        if (j < CANDIDATE_BUFFER_SIZE)
        {
            key_1[i] = candidate_distances[j];
            val_1[i] = candidate_indices[j];
        }
        else
        {
            key_1[i] = std::numeric_limits<DISTANCE_T>::max();
            val_1[i] = std::numeric_limits<INDEX_T>::max();
        }
    }
    warp_sort<float, uint32_t, N_1>(key_1, val_1);
    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = CANDIDATE_BUFFER_SIZE - 1 - ( (N_1 * lane_id) + i );
        if (j < CANDIDATE_BUFFER_SIZE){
            candidate_distances[j] = key_1[i];
            candidate_indices[j]   = val_1[i];
        }
    }
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_by_bitonic_sort
(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    uint32_t RESULT_BUFFER_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t INTERNAL_TOPK,
    bool first
)
{
    const unsigned lane_id = threadIdx.x % 32;
    const unsigned warp_id = threadIdx.x / 32;

    if (warp_id > 0) { return; }
    if (CANDIDATE_BUFFER_SIZE != 64) 
    {
        printf("CANDIDATE_BUFFER_SIZE must be 64\n");
        assert(false);
    }
    DISTANCE_T key_1[N_1];
    INDEX_T val_1[N_1];
    auto candidate_distances = result_distances_ptr + INTERNAL_TOPK;
    auto candidate_indices = result_indices_ptr + INTERNAL_TOPK;
    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = lane_id + (32 * i);
        if (j < CANDIDATE_BUFFER_SIZE)
        {
            key_1[i] = candidate_distances[j];
            val_1[i] = candidate_indices[j];
        }
        else
        {
            key_1[i] = std::numeric_limits<DISTANCE_T>::max();
            val_1[i] = std::numeric_limits<INDEX_T>::max();
        }
    }
    warp_sort<float, uint32_t, N_1>(key_1, val_1);
    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = (N_1 * lane_id) + i;
        if (j < CANDIDATE_BUFFER_SIZE && j < INTERNAL_TOPK){
            candidate_distances[j] = key_1[i];
            candidate_indices[j]   = val_1[i];
        }
    }

    DISTANCE_T key_2[N_2];
    INDEX_T val_2[N_2];
    if (first) {
        for (unsigned i = 0; i < N_2; i++) {
            unsigned j = lane_id + (32 * i);
            if (j < INTERNAL_TOPK) {
                key_2[i] = result_distances_ptr[j];
                val_2[i] = result_indices_ptr[j];
            } else {
                key_2[i] = std::numeric_limits<DISTANCE_T>::max();
                val_2[i] = std::numeric_limits<INDEX_T>::max();
            }
        }
        warp_sort<float, uint32_t, N_2>(key_2, val_2);
    }
    else
    {
        for (unsigned i = 0; i < N_2; i++) {
            unsigned j = (N_2 * lane_id) + i;
            if (j < INTERNAL_TOPK) {
                key_2[i] = result_distances_ptr[j];
                val_2[i] = result_indices_ptr[j];
            } else {
                key_2[i] = std::numeric_limits<DISTANCE_T>::max();
                val_2[i] = std::numeric_limits<INDEX_T>::max();
            }
        }
    }
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = (N_2 * lane_id) + i;
      unsigned k = INTERNAL_TOPK - 1 - j;
      if (k >= INTERNAL_TOPK || k >= CANDIDATE_BUFFER_SIZE) continue;
      auto candidate_key = candidate_distances[k];
      if (key_2[i] > candidate_key) {
        key_2[i] = candidate_key;
        val_2[i] = candidate_indices[k];
      }
    }
    warp_merge<float, uint32_t, N_2>(key_2, val_2, 32);
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = (N_2 * lane_id) + i;
      if (j < INTERNAL_TOPK) {
        result_distances_ptr[j] = key_2[i];
        result_indices_ptr[j]   = val_2[i];
      }
    }
}

/////////////////////////////////////////////////////// PQ distance computation ///////////////////////////////////////////////////////

template<uint32_t VECTOR_DIM>
__device__ DISTANCE_T compute_similarity_pq
(
    uint8_t* d_pq_codes,
    INDEX_T child_id,
    bool valid_child,
    float* dist_table         
)
{
    DISTANCE_T norm2 = 0.0f;
    const unsigned lane_id = threadIdx.x % 32;   // we use only first 8 lanes per team
    if (valid_child && lane_id < PQ_M) {
        uint8_t code = d_pq_codes[child_id * PQ_M + lane_id];
        norm2 = dist_table[lane_id * PQ_Ks + code];
    }
    // reduce within warp (only first 8 lanes have meaningful values)
    unsigned full_mask = 0xffffffff;
    for (uint32_t offset = PQ_M / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(full_mask, norm2, offset);
    }
    return norm2;
}


template<uint32_t VECTOR_DIM, uint32_t TEAM_SIZE>
__device__ void compute_distance_pq_batch
(
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    uint8_t* d_pq_codes,
    float* dist_table,
    uint32_t CANDIDATE_BUFFER_SIZE
)
{
    for (uint32_t tid = threadIdx.x; tid < CANDIDATE_BUFFER_SIZE * TEAM_SIZE; tid += blockDim.x) {
        const auto i = tid / TEAM_SIZE;
        const bool valid_i = (i < CANDIDATE_BUFFER_SIZE);
        INDEX_T child_id = valid_i ? candidate_indices_ptr[i] : std::numeric_limits<INDEX_T>::max();
        DISTANCE_T norm2 = compute_similarity_pq<VECTOR_DIM>(d_pq_codes, child_id, child_id != std::numeric_limits<INDEX_T>::max(), dist_table);
        const unsigned lane_id = threadIdx.x % TEAM_SIZE;
        if (valid_i && lane_id == 0) {
            candidate_distances_ptr[i] = (child_id == std::numeric_limits<INDEX_T>::max()) ? FLT_MAX : norm2;
        }
    }
}

/////////////////////////////////////////////////////// helpers for seed initialization ///////////////////////////////////////////////////////

template <uint32_t VECTOR_DIM, uint32_t TEAM_SIZE>
__device__ void compute_distance_to_maped_top10_nodes_pq
(
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    DATA_T* query_ptr,
    DATA_T* dataset_ptr,
    INDEX_T* graph_ptr,
    INDEX_T* top10_ptr,
    INDEX_T* seed_map_ptr,
    uint32_t GRAPH_DEGREE,
    uint32_t DATASET_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    INDEX_T* visited_hash_ptr,
    uint32_t BITLEN,
    uint32_t SEED_CONFIG,
    uint32_t HASH_TABLE_CONFIG,
    uint32_t SEED_TOPK_SIZE,
    uint8_t* d_pq_codes,
    float* dist_table
)
{
    const INDEX_T invalid_index = std::numeric_limits<INDEX_T>::max();
    INDEX_T top1 = top10_ptr[blockIdx.y * SEED_TOPK_SIZE];
    INDEX_T parent_id = seed_map_ptr[top1];
    for (uint32_t i = threadIdx.x; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
        INDEX_T child_id = graph_ptr[(static_cast<int64_t>(GRAPH_DEGREE) * parent_id) + i];
        if (HASH_TABLE_CONFIG) {
            if (child_id != invalid_index) {
                if (hashtable_insert(visited_hash_ptr, BITLEN, child_id) == 0) {
                    child_id = invalid_index;
                    candidate_distances_ptr[i] = FLT_MAX;
                } else {
                    candidate_distances_ptr[i] = 0;
                }
            }
            candidate_indices_ptr[i] = child_id;
        }
    }
    __syncthreads();
    compute_distance_pq_batch<VECTOR_DIM, TEAM_SIZE>(candidate_indices_ptr, candidate_distances_ptr, d_pq_codes, dist_table, CANDIDATE_BUFFER_SIZE);
}

template <uint32_t VECTOR_DIM, uint32_t TEAM_SIZE>
__device__ void compute_distance_to_random_nodes_pq
(
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    DATA_T* query_ptr,
    DATA_T* dataset_ptr,
    uint32_t DATASET_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t NUM_DIST,
    INDEX_T* visited_hash_ptr,
    uint32_t BITLEN,
    uint32_t HASH_TABLE_CONFIG,
    uint8_t* d_pq_codes,
    float* dist_table
)
{
    uint32_t max_i = CANDIDATE_BUFFER_SIZE;
    if (max_i % (32 / TEAM_SIZE)) { max_i += (32 / TEAM_SIZE) - (max_i % (32 / TEAM_SIZE)); }

    for (uint32_t i = threadIdx.x / TEAM_SIZE; i < max_i; i += blockDim.x / TEAM_SIZE) {
        const bool valid_i = (i < CANDIDATE_BUFFER_SIZE);
        INDEX_T best_index_team_local;
        DISTANCE_T best_norm2_team_local = std::numeric_limits<DISTANCE_T>::max();
        for (uint32_t j = 0; j < NUM_DIST; j++) {
            INDEX_T seed_index;
            if (valid_i) {
                uint32_t gid = threadIdx.x + (blockDim.x * (i + (CANDIDATE_BUFFER_SIZE * j)));
                curandState state;
                curand_init(clock64(), gid, 0, &state);
                seed_index = curand_uniform(&state) * DATASET_SIZE;
            }
            DISTANCE_T norm2 = compute_similarity_pq<VECTOR_DIM>(d_pq_codes, seed_index, valid_i, dist_table);
            if (valid_i && (norm2 < best_norm2_team_local)) {
                best_norm2_team_local = norm2;
                best_index_team_local = seed_index;
            }
        }
        const unsigned lane_id = threadIdx.x % TEAM_SIZE;
        if (valid_i && lane_id == 0) {
            if (HASH_TABLE_CONFIG) {
                if (hashtable_insert(visited_hash_ptr, BITLEN, best_index_team_local)) {
                    candidate_distances_ptr[i] = best_norm2_team_local;
                    candidate_indices_ptr[i]   = best_index_team_local;
                } else {
                    candidate_distances_ptr[i] = std::numeric_limits<DISTANCE_T>::max();
                    candidate_indices_ptr[i]   = std::numeric_limits<INDEX_T>::max();
                }
            } else {
                candidate_distances_ptr[i] = best_norm2_team_local;
                candidate_indices_ptr[i]   = best_index_team_local;
            }
        }
    }
}

/////////////////////////////////////////////////////// pick next parents (unchanged) ///////////////////////////////////////////////////////

__device__ void pickup_next_parents
(
    uint32_t* terminate_flag,
    INDEX_T* const parent_list_buffer,
    INDEX_T* const result_indices_buffer,
    const uint32_t INTERNAL_TOPK,
    const uint32_t SEARCH_WIDTH
)
{
    constexpr INDEX_T index_msb_1_mask = 0x80000000;
    for (std::uint32_t i = threadIdx.x; i < SEARCH_WIDTH; i += 32) {
        parent_list_buffer[i] = std::numeric_limits<INDEX_T>::max();
    }
    std::uint32_t itopk_max = INTERNAL_TOPK;
    if (itopk_max % 32) { itopk_max += 32 - (itopk_max % 32); }
    std::uint32_t num_new_parents = 0;
    
    for (std::uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
        INDEX_T index;
        int new_parent = 0;
        if (j < INTERNAL_TOPK) {
            index = result_indices_buffer[j];
            if ((index & index_msb_1_mask) == 0) {
                new_parent = 1;
            }
        }
        const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
        if (new_parent) {
            const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_new_parents;
            if (i < SEARCH_WIDTH) {
                parent_list_buffer[i] = j;
                result_indices_buffer[j] |= index_msb_1_mask;
            }
        }
        num_new_parents += __popc(ballot_mask);
        if (num_new_parents >= SEARCH_WIDTH) { break; }
    }
    if (threadIdx.x == 0 && (num_new_parents == 0)) { *terminate_flag = 1; }
}

/////////////////////////////////////////////////////// direction pruning with fallback ///////////////////////////////////////////////////////

template <uint32_t VECTOR_DIM, uint32_t TEAM_SIZE, uint32_t INTERNAL_TOPK>
__device__ inline void compute_distance_to_child_nodes_team_with_direction_pq
(
    INDEX_T* parent_buffer,
    INDEX_T* internal_topk_list,
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    DATA_T* query_ptr,
    DATA_T* d_dataset_ptr,
    INDEX_T* d_graph_ptr,
    uint32_t* d_sign_bit_ptr,
    uint32_t GRAPH_DEGREE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t PRUNE_CONFIG,
    float PRUNE_RATIO,
    uint8_t* d_pq_codes,
    float* dist_table
)
{
    constexpr INDEX_T index_msb_1_mask = 0x80000000;
    const INDEX_T invalid_index = std::numeric_limits<INDEX_T>::max();
    const INDEX_T smem_parent_id = parent_buffer[0];
    const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;

    // Pre-compute sign bits of query relative to parent for SIGN_BIT_PRUNE
    constexpr uint32_t packed_vector_dim_size = (VECTOR_DIM + 31) / 32;
    __shared__ int32_t query_sign_bit[packed_vector_dim_size];

    if (PRUNE_CONFIG == SIGN_BIT_PRUNE) {
        uint32_t VECTOR_DIM_32 = (VECTOR_DIM + 31) & ~31;
        for (uint32_t i = threadIdx.x; i < VECTOR_DIM_32; i += blockDim.x) {
            const bool valid_i = i < VECTOR_DIM;
            uint32_t sign_bit = (query_ptr[i] > d_dataset_ptr[parent_id * VECTOR_DIM + i]) ? 1 : 0;
            sign_bit <<= (31 - (threadIdx.x % 32));
            unsigned active = __ballot_sync(0xffffffff, valid_i);
            __syncwarp(active);
            for (int offset = 16; offset > 0; offset >>= 1) {
                sign_bit |= __shfl_xor_sync(0xffffffff, sign_bit, offset);
            }
            if (valid_i) {
                query_sign_bit[i / 32] = sign_bit;
            }
            __syncwarp(active);
        }
        __syncthreads();
    }


    for (uint32_t i = threadIdx.x; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
        INDEX_T child_id = d_graph_ptr[(static_cast<int64_t>(GRAPH_DEGREE) * parent_id) + i];
        // Remove duplicates already in internal topk
        for (int32_t k = 0; k < INTERNAL_TOPK; ++k) {
            INDEX_T topk_id = internal_topk_list[k] & ~index_msb_1_mask;
            if (child_id == topk_id) {
                child_id = invalid_index;
                break;
            }
        }
        candidate_indices_ptr[i] = child_id;
        float score = -1e9f;
        if (child_id != invalid_index) {
            if (PRUNE_CONFIG == SIGN_BIT_PRUNE) {
                int direction = 0;
                uint32_t sign_bit_vector_size = VECTOR_DIM * GRAPH_DEGREE / 32; // each row has this many uint32_t
                for (int j = 0; j < packed_vector_dim_size; ++j) {
                    direction -= __popcll(query_sign_bit[j] ^ d_sign_bit_ptr[parent_id * sign_bit_vector_size + i * packed_vector_dim_size + j]);
                }
                score = static_cast<float>(direction);
            } else if (PRUNE_CONFIG == RANDOM_PRUNE) {
                uint32_t gid = threadIdx.x + blockDim.x * i;
                curandState state;
                curand_init(clock64(), gid, 0, &state);
                score = (curand_uniform(&state) - 0.5f) * VECTOR_DIM;
            }
        }
        candidate_distances_ptr[i] = score;
    }
    __syncthreads();

   
    candidate_by_bitonic_sort_inverse<2, INTERNAL_TOPK / 32>
        (candidate_indices_ptr, candidate_distances_ptr, CANDIDATE_BUFFER_SIZE);
    __syncthreads();

   
    int keep = max(1, static_cast<int>(CANDIDATE_BUFFER_SIZE * PRUNE_RATIO));
    __shared__ int selected_cnt;
    if (threadIdx.x == 0) {
        selected_cnt = 0;
        for (int i = 0; i < keep; ++i) {
            if (candidate_indices_ptr[i] != invalid_index) ++selected_cnt;
        }
    }
    __syncthreads();

    const int fallback_threshold = 4;  // if fewer than 4 valid nodes, compute all
    if (selected_cnt < fallback_threshold) {
        keep = CANDIDATE_BUFFER_SIZE;
    }

   
    for (uint32_t tid = threadIdx.x; tid < CANDIDATE_BUFFER_SIZE * TEAM_SIZE; tid += blockDim.x) {
        uint32_t i = tid / TEAM_SIZE;
        bool compute = (i < keep);
        INDEX_T child_id = compute ? candidate_indices_ptr[i] : invalid_index;
        float dist = FLT_MAX;
        if (compute && child_id != invalid_index) {
            dist = compute_similarity_pq<VECTOR_DIM>(d_pq_codes, child_id, true, dist_table);
        }
        unsigned lane_id = threadIdx.x % TEAM_SIZE;
        if (lane_id == 0) {
            if (compute && child_id != invalid_index) {
                candidate_distances_ptr[i] = dist;
            } else {
                candidate_distances_ptr[i] = FLT_MAX;
            }
        }
    }
    __syncthreads();
}


template <uint32_t VECTOR_DIM, uint32_t TEAM_SIZE, uint32_t INTERNAL_TOPK>
__device__ void compute_distance_to_child_nodes_pq
(
    INDEX_T* parent_buffer,
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    INDEX_T* internal_topk_list,
    DATA_T* query_ptr,
    DATA_T* d_dataset_ptr,
    INDEX_T* d_graph_ptr,
    uint32_t GRAPH_DEGREE,
    uint32_t SEARCH_WIDTH,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t NUM_DIST,
    INDEX_T* visited_hash_ptr,
    uint32_t BITLEN,
    uint32_t* counter,
    uint32_t THRESHOLD_CONFIG,
    uint32_t FULL_COMPUTE_RATIO,
    uint32_t HASH_TABLE_CONFIG,
    uint8_t* d_pq_codes,
    float* dist_table
)
{
    constexpr INDEX_T index_msb_1_mask = 0x80000000;
    const INDEX_T invalid_index = std::numeric_limits<INDEX_T>::max();

    for (uint32_t i = threadIdx.x; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x) {
        const INDEX_T smem_parent_id = parent_buffer[0];
        const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
        INDEX_T child_id = d_graph_ptr[(static_cast<int64_t>(GRAPH_DEGREE) * parent_id) + i];
        if (!HASH_TABLE_CONFIG) {
            for (int32_t k = 0; k < INTERNAL_TOPK; ++k) {
                INDEX_T topk_id = internal_topk_list[k] & ~index_msb_1_mask;
                if (child_id == topk_id) {
                    child_id = invalid_index;
                    break;
                }
            }
            candidate_indices_ptr[i] = child_id;
            candidate_distances_ptr[i] = (child_id != invalid_index) ? 0.0f : FLT_MAX;
        } else {
            if (child_id != invalid_index) {
                if (hashtable_insert(visited_hash_ptr, BITLEN, child_id) == 0) {
                    child_id = invalid_index;
                    candidate_distances_ptr[i] = FLT_MAX;
                } else {
                    candidate_distances_ptr[i] = 0.0f;
                }
            }
            candidate_indices_ptr[i] = child_id;
        }
    }
    __syncthreads();
    compute_distance_pq_batch<VECTOR_DIM, TEAM_SIZE>(candidate_indices_ptr, candidate_distances_ptr, d_pq_codes, dist_table, CANDIDATE_BUFFER_SIZE);
}

/////////////////////////////////////////////////////// main search kernel ///////////////////////////////////////////////////////

template<uint32_t VECTOR_DIM, uint32_t INTERNAL_TOPK>
__global__ void search_kernel
(
    INDEX_T* d_graph_ptr,
    DATA_T* d_dataset_ptr,
    DATA_T* d_queries_ptr,
    INDEX_T* d_top10_ptr,
    INDEX_T* d_seed_map_ptr,
    uint32_t* d_sign_bit_ptr,
    INDEX_T* d_results_ptr,
    DISTANCE_T* d_distances_ptr,
    uint32_t NUM_QUERIES,
    uint32_t TOPK,
    uint32_t SEARCH_WIDTH,
    uint32_t MAX_ITER,
    uint32_t MIN_ITER,
    uint32_t DATASET_SIZE,
    uint32_t TEAM_SIZE,
    uint32_t GRAPH_DEGREE,
    uint32_t NUM_DIST,
    uint32_t BLOCK_SIZE,
    uint32_t BITLEN,
    uint32_t SMALL_HASH_RESET_INTERVAL,
    uint32_t PRUNE_CONFIG,
    float ITERATION_DIRECTION_RATIO,
    float PRUNE_RATIO,
    uint32_t THRESHOLD_CONFIG,
    uint32_t FULL_COMPUTE_RATIO,
    uint32_t SEED_CONFIG,
    uint32_t HASH_TABLE_CONFIG,
    uint32_t SEED_TOPK_SIZE,
    float* d_pq_codebook,
    uint8_t* d_pq_codes
)
{
    static_assert(VECTOR_DIM % PQ_M == 0, "VECTOR_DIM must be multiple of PQ_M");
    constexpr int dsub = VECTOR_DIM / PQ_M;

    if (TEAM_SIZE != 8) {
        printf("TEAM_SIZE must be 8 for PQ mode\n");
        assert(false);
    }

    const uint32_t query_id = blockIdx.y;

    extern __shared__ uint32_t smem[];
    uint32_t RESULT_BUFFER_SIZE = INTERNAL_TOPK + SEARCH_WIDTH * GRAPH_DEGREE;
    uint32_t CANDIDATE_BUFFER_SIZE = SEARCH_WIDTH * GRAPH_DEGREE;
    uint32_t QUERY_BUFFER_SIZE = VECTOR_DIM;
    uint32_t hash_table_size = 1 << BITLEN;

   
    auto query_buffer = reinterpret_cast<DATA_T*>(smem);
    auto result_indices_buffer = reinterpret_cast<INDEX_T*>(query_buffer + QUERY_BUFFER_SIZE);
    auto result_distances_buffer = reinterpret_cast<DISTANCE_T*>(result_indices_buffer + RESULT_BUFFER_SIZE);
    auto visited_hash_buffer = reinterpret_cast<INDEX_T*>(result_distances_buffer + RESULT_BUFFER_SIZE);
    auto parent_list_buffer = reinterpret_cast<INDEX_T*>(visited_hash_buffer + hash_table_size);
    auto terminate_flag = reinterpret_cast<uint32_t*>(parent_list_buffer + SEARCH_WIDTH);
    terminate_flag[0] = 0;
    auto counter = reinterpret_cast<uint32_t*>(terminate_flag + 1);
    // dist_table placed after counter, ensure alignment (float)
    float* dist_table = reinterpret_cast<float*>(counter + 1);

    if (HASH_TABLE_CONFIG) {
        hashtable_init(visited_hash_buffer, BITLEN);
    }

   
    for (uint32_t i = threadIdx.x; i < VECTOR_DIM; i += blockDim.x) {
        query_buffer[i] = d_queries_ptr[query_id * VECTOR_DIM + i];
    }

    
    const int total_dist_entries = PQ_M * PQ_Ks;
    for (int idx = threadIdx.x; idx < total_dist_entries; idx += blockDim.x) {
        int m = idx / PQ_Ks;
        int ks = idx % PQ_Ks;
        const float* center = d_pq_codebook + m * PQ_Ks * dsub + ks * dsub;
        float dist = 0.0f;
        for (int d = 0; d < dsub; ++d) {
            float diff = query_buffer[m * dsub + d] - center[d];
            dist += diff * diff;
        }
        dist_table[m * PQ_Ks + ks] = dist;
    }
    __syncthreads();

    // Initialize result buffers
    for (uint32_t i = threadIdx.x; i < RESULT_BUFFER_SIZE; i += blockDim.x) {
        result_indices_buffer[i] = std::numeric_limits<INDEX_T>::max();
        result_distances_buffer[i] = std::numeric_limits<DISTANCE_T>::max();
    }
    __syncthreads();


    if (SEED_CONFIG == NO_SEED) {
        compute_distance_to_random_nodes_pq<VECTOR_DIM, 8>
        (
            result_indices_buffer + INTERNAL_TOPK,
            result_distances_buffer + INTERNAL_TOPK,
            query_buffer,
            d_dataset_ptr,
            DATASET_SIZE,
            CANDIDATE_BUFFER_SIZE,
            NUM_DIST,
            visited_hash_buffer,
            BITLEN,
            HASH_TABLE_CONFIG,
            d_pq_codes,
            dist_table
        );
    } else {
        compute_distance_to_maped_top10_nodes_pq<VECTOR_DIM, 8>
        (
            result_indices_buffer + INTERNAL_TOPK,
            result_distances_buffer + INTERNAL_TOPK,
            query_buffer,
            d_dataset_ptr,
            d_graph_ptr,
            d_top10_ptr,
            d_seed_map_ptr,
            GRAPH_DEGREE,
            DATASET_SIZE,
            CANDIDATE_BUFFER_SIZE,
            visited_hash_buffer,
            BITLEN,
            SEED_CONFIG,
            HASH_TABLE_CONFIG,
            SEED_TOPK_SIZE,
            d_pq_codes,
            dist_table
        );
    }
    __syncthreads();

    uint32_t iter = 0;
    while (1) {
        if (HASH_TABLE_CONFIG && (iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
            hashtable_init(visited_hash_buffer, BITLEN);
            __syncthreads();
        }

        // Sort internal topk and merge candidates
        topk_by_bitonic_sort<2, INTERNAL_TOPK / 32>
            (
                result_indices_buffer,
                result_distances_buffer,
                RESULT_BUFFER_SIZE,
                CANDIDATE_BUFFER_SIZE,
                INTERNAL_TOPK,
                (iter == 0)
            );
        __syncthreads();

        if (iter + 1 == MAX_ITER) break;

        
        if (threadIdx.x < 32) {
            pickup_next_parents
                (
                    terminate_flag,
                    parent_list_buffer,
                    result_indices_buffer,
                    INTERNAL_TOPK,
                    SEARCH_WIDTH
                );
        }

        if (HASH_TABLE_CONFIG && (iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
            const unsigned first_tid = ((blockDim.x <= 32) ? 0 : 32);
            hashtable_restore
                (
                    visited_hash_buffer,
                    BITLEN,
                    result_indices_buffer,
                    INTERNAL_TOPK,
                    first_tid
                );
            __syncthreads();
        }

        if (*terminate_flag && iter >= MIN_ITER) break;

        if (PRUNE_CONFIG != NO_PRUNE && iter < MAX_ITER * ITERATION_DIRECTION_RATIO) {
            compute_distance_to_child_nodes_team_with_direction_pq<VECTOR_DIM, 8, INTERNAL_TOPK>
            (
                parent_list_buffer,
                result_indices_buffer,
                result_indices_buffer + INTERNAL_TOPK,
                result_distances_buffer + INTERNAL_TOPK,
                query_buffer,
                d_dataset_ptr,
                d_graph_ptr,
                d_sign_bit_ptr,
                GRAPH_DEGREE,
                CANDIDATE_BUFFER_SIZE,
                PRUNE_CONFIG,
                PRUNE_RATIO,
                d_pq_codes,
                dist_table
            );
        } else {
            compute_distance_to_child_nodes_pq<VECTOR_DIM, 8, INTERNAL_TOPK>
            (
                parent_list_buffer,
                result_indices_buffer + INTERNAL_TOPK,
                result_distances_buffer + INTERNAL_TOPK,
                result_indices_buffer,
                query_buffer,
                d_dataset_ptr,
                d_graph_ptr,
                GRAPH_DEGREE,
                SEARCH_WIDTH,
                CANDIDATE_BUFFER_SIZE,
                NUM_DIST,
                visited_hash_buffer,
                BITLEN,
                counter,
                THRESHOLD_CONFIG,
                FULL_COMPUTE_RATIO,
                HASH_TABLE_CONFIG,
                d_pq_codes,
                dist_table
            );
        }
        __syncthreads();
        iter++;
    }


    topk_by_bitonic_sort<2, INTERNAL_TOPK / 32>
            (
                result_indices_buffer,
                result_distances_buffer,
                RESULT_BUFFER_SIZE,
                CANDIDATE_BUFFER_SIZE,
                INTERNAL_TOPK,
                false
            );
    __syncthreads();


    for (uint32_t i = threadIdx.x; i < TOPK; i += blockDim.x) {
        d_results_ptr[query_id * TOPK + i] = result_indices_buffer[i] & ~0x80000000;
        d_distances_ptr[query_id * TOPK + i] = result_distances_buffer[i];
    }
}

/////////////////////////////////////////////////////// Host interface ///////////////////////////////////////////////////////

torch::Tensor search(
    torch::Tensor graph, 
    torch::Tensor dataset, 
    torch::Tensor queries, 
    torch::Tensor top10,
    torch::Tensor seed_map,
    torch::Tensor sign_bit, 
    torch::Tensor configs,
    torch::Tensor results_distances,
    torch::Tensor pq_codebook,
    torch::Tensor pq_codes
) 
{
    if (!graph.is_cuda() || !dataset.is_cuda() || !queries.is_cuda() ||
        !top10.is_cuda() || !seed_map.is_cuda() || !sign_bit.is_cuda() ||
        !configs.is_cuda() || !pq_codebook.is_cuda() || !pq_codes.is_cuda()) {
        throw std::runtime_error("All input tensors must be on GPU.");
    }

    uint32_t NUM_QUERIES = configs[0].item<int>();
    uint32_t TOPK = configs[1].item<int>();
    uint32_t SEARCH_WIDTH = configs[2].item<int>();
    uint32_t MAX_ITER = configs[3].item<int>();
    uint32_t MIN_ITER = configs[4].item<int>();
    uint32_t INTERNAL_TOPK = configs[5].item<int>();
    uint32_t VECTOR_DIM = configs[6].item<int>();
    uint32_t DATASET_SIZE = configs[7].item<int>();
    uint32_t TEAM_SIZE = configs[8].item<int>();
    uint32_t GRAPH_DEGREE = configs[9].item<int>();
    uint32_t NUM_DIST = configs[10].item<int>();
    uint32_t BLOCK_SIZE = configs[11].item<int>();
    uint32_t BITLEN = configs[12].item<int>();
    uint32_t SMALL_HASH_RESET_INTERVAL = configs[13].item<int>();
    uint32_t SHARED_MEM_SIZE = configs[14].item<int>();
    uint32_t PRUNE_CONFIG = configs[15].item<int>();
    float ITERATION_DIRECTION_RATIO = configs[16].item<float>();
    float PRUNE_RATIO = configs[17].item<float>();
    uint32_t THRESHOLD_CONFIG = configs[18].item<int>();
    uint32_t FULL_COMPUTE_RATIO = configs[19].item<int>();
    uint32_t SEED_CONFIG = configs[20].item<int>();
    uint32_t HASH_TABLE_CONFIG = configs[21].item<int>();
    uint32_t SEED_TOPK_SIZE = (configs.size(0) > 22) ? configs[22].item<int>() : 1;

    torch::Tensor results_indices = torch::zeros({NUM_QUERIES, TOPK}, torch::device(queries.device()).dtype(torch::kUInt32));

    INDEX_T* d_graph_ptr = graph.data_ptr<INDEX_T>();
    DATA_T* d_dataset_ptr = dataset.data_ptr<DATA_T>();
    DATA_T* d_queries_ptr = queries.data_ptr<DATA_T>();
    INDEX_T* d_top10_ptr = top10.data_ptr<INDEX_T>();
    INDEX_T* d_seed_map_ptr = seed_map.data_ptr<INDEX_T>();
    uint32_t* d_sign_bit_ptr = sign_bit.data_ptr<uint32_t>();
    INDEX_T* d_results_indices_ptr = results_indices.data_ptr<INDEX_T>();
    DISTANCE_T* d_results_distances_ptr = results_distances.data_ptr<DISTANCE_T>();
    float* d_pq_codebook_ptr = pq_codebook.data_ptr<float>();
    uint8_t* d_pq_codes_ptr = pq_codes.data_ptr<uint8_t>();

    dim3 thread_dims(BLOCK_SIZE, 1, 1);
    dim3 block_dims(1, NUM_QUERIES, 1);

    // Instantiate based on VECTOR_DIM and INTERNAL_TOPK
    // Add more cases as needed for other dimensions.
    if (VECTOR_DIM == 128) {
        if (INTERNAL_TOPK == 64) {
            search_kernel<128, 64><<<block_dims, thread_dims, SHARED_MEM_SIZE>>>(
                d_graph_ptr, d_dataset_ptr, d_queries_ptr, d_top10_ptr, d_seed_map_ptr, d_sign_bit_ptr,
                d_results_indices_ptr, d_results_distances_ptr,
                NUM_QUERIES, TOPK, SEARCH_WIDTH, MAX_ITER, MIN_ITER, DATASET_SIZE, TEAM_SIZE,
                GRAPH_DEGREE, NUM_DIST, BLOCK_SIZE, BITLEN, SMALL_HASH_RESET_INTERVAL,
                PRUNE_CONFIG, ITERATION_DIRECTION_RATIO, PRUNE_RATIO,
                THRESHOLD_CONFIG, FULL_COMPUTE_RATIO, SEED_CONFIG, HASH_TABLE_CONFIG, SEED_TOPK_SIZE,
                d_pq_codebook_ptr, d_pq_codes_ptr);
        } else if (INTERNAL_TOPK == 128) {
            search_kernel<128, 128><<<block_dims, thread_dims, SHARED_MEM_SIZE>>>(
                d_graph_ptr, d_dataset_ptr, d_queries_ptr, d_top10_ptr, d_seed_map_ptr, d_sign_bit_ptr,
                d_results_indices_ptr, d_results_distances_ptr,
                NUM_QUERIES, TOPK, SEARCH_WIDTH, MAX_ITER, MIN_ITER, DATASET_SIZE, TEAM_SIZE,
                GRAPH_DEGREE, NUM_DIST, BLOCK_SIZE, BITLEN, SMALL_HASH_RESET_INTERVAL,
                PRUNE_CONFIG, ITERATION_DIRECTION_RATIO, PRUNE_RATIO,
                THRESHOLD_CONFIG, FULL_COMPUTE_RATIO, SEED_CONFIG, HASH_TABLE_CONFIG, SEED_TOPK_SIZE,
                d_pq_codebook_ptr, d_pq_codes_ptr);
        } else {
            printf("Invalid INTERNAL_TOPK for VECTOR_DIM=128\n");
            assert(false);
        }
    } else {
        printf("Unsupported VECTOR_DIM for PQ mode. Add new case in host code.\n");
        assert(false);
    }

    return results_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("search", &search, "A function that performs search on the graph using PQ distance with direction pruning fallback");
}
