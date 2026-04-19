#ifdef TICS_HAS_GPU_BROAD_PHASE

#include "tics_internal.h"

#include <broad_phase_cuda.h>
#include <stb_ds.h>

#include <assert.h>
#include <stddef.h>

// ================================================================================================
// GPU Broad Phase Adapter
// ================================================================================================
// This file is the only place that knows about CUDA. It translates between tics' GPU-agnostic API
// (gpu_*) and the CUDA-specific implementation (cuda_*).

typedef struct cuda_state_all {
	cuda_shared_state* shared_state;

	cuda_state_brute_force* state_brute_force;
	cuda_state_grid_a* state_grid_a;
	cuda_state_grid_b* state_grid_b;
} cuda_state_all;

void* gpu_broad_phase_create(void) {
	cuda_state_all* s = (cuda_state_all*)calloc(1, sizeof(cuda_state_all));
	s->shared_state = cuda_shared_state_create();
	s->state_brute_force = cuda_state_brute_force_create();
	s->state_grid_a = cuda_state_grid_a_create();
	s->state_grid_b = cuda_state_grid_b_create();
	return s;
}

void gpu_broad_phase_destroy(void* state) {
	cuda_state_all* s = (cuda_state_all*)state;
	cuda_shared_state_destroy(s->shared_state);
	cuda_state_brute_force_destroy(s->state_brute_force);
	cuda_state_grid_a_destroy(s->state_grid_a);
	cuda_state_grid_b_destroy(s->state_grid_b);
	free(s);
}

static broad_phase_pair* convert_pairs(cuda_pair* cu_pairs, size_t count) {
	broad_phase_pair* pairs = NULL;
	if (count > 0) {
		arrsetlen(pairs, count);
		for (size_t i = 0; i < count; ++i) {
			pairs[i].a.type = RIGID_BODY;
			pairs[i].a.index = cu_pairs[i].a_index;
			pairs[i].b.type = cu_pairs[i].b_type;
			pairs[i].b.index = cu_pairs[i].b_index;
		}
		free(cu_pairs);
	}
	return pairs;
}

// Compile-time assertions: Ensure packed_aabb and cuda_aabb have the same layout
_Static_assert(sizeof(packed_aabb) == sizeof(cuda_aabb),
			   "packed_aabb and cuda_aabb must have identical size");
_Static_assert(offsetof(packed_aabb, min_x) == offsetof(cuda_aabb, min_x),
			   "packed_aabb and cuda_aabb layout mismatch");
_Static_assert(offsetof(packed_aabb, max_z) == offsetof(cuda_aabb, max_z),
			   "packed_aabb and cuda_aabb layout mismatch");

broad_phase_pair* gpu_broad_phase_run_grid_a(void* gpu_state, packed_aabb* packed_rigid_proxies,
											 size_t rigid_count, packed_aabb* packed_static_proxies,
											 size_t static_count, bool statics_changed) {
	size_t count = 0;
	cuda_pair* cu_pairs = cuda_broad_phase_grid_a(
		((cuda_state_all*)gpu_state)->shared_state, ((cuda_state_all*)gpu_state)->state_grid_a,
		(const cuda_aabb*)packed_rigid_proxies, (int)rigid_count,
		(const cuda_aabb*)packed_static_proxies, (int)static_count, statics_changed, &count);
	return convert_pairs(cu_pairs, count);
}

broad_phase_pair* gpu_broad_phase_run_grid_b_half_shell(void* gpu_state,
														packed_aabb* packed_rigid_proxies,
														size_t rigid_count,
														packed_aabb* packed_static_proxies,
														size_t static_count, bool statics_changed) {
	size_t count = 0;
	cuda_pair* cu_pairs = cuda_broad_phase_grid_b(
		((cuda_state_all*)gpu_state)->shared_state, ((cuda_state_all*)gpu_state)->state_grid_b,
		(const cuda_aabb*)packed_rigid_proxies, (int)rigid_count,
		(const cuda_aabb*)packed_static_proxies, (int)static_count, statics_changed, &count, true);
	return convert_pairs(cu_pairs, count);
}

broad_phase_pair* gpu_broad_phase_run_grid_b_naive(void* gpu_state,
												   packed_aabb* packed_rigid_proxies,
												   size_t rigid_count,
												   packed_aabb* packed_static_proxies,
												   size_t static_count, bool statics_changed) {
	size_t count = 0;
	cuda_pair* cu_pairs = cuda_broad_phase_grid_b(
		((cuda_state_all*)gpu_state)->shared_state, ((cuda_state_all*)gpu_state)->state_grid_b,
		(const cuda_aabb*)packed_rigid_proxies, (int)rigid_count,
		(const cuda_aabb*)packed_static_proxies, (int)static_count, statics_changed, &count, false);
	return convert_pairs(cu_pairs, count);
}

broad_phase_pair* gpu_broad_phase_run_brute_force(void* gpu_state,
												  packed_aabb* packed_rigid_proxies,
												  size_t rigid_count,
												  packed_aabb* packed_static_proxies,
												  size_t static_count, bool statics_changed) {
	size_t count = 0;
	cuda_pair* cu_pairs = cuda_broad_phase_brute_force(
		((cuda_state_all*)gpu_state)->shared_state, ((cuda_state_all*)gpu_state)->state_brute_force,
		(const cuda_aabb*)packed_rigid_proxies, (int)rigid_count,
		(const cuda_aabb*)packed_static_proxies, (int)static_count, statics_changed, &count);
	return convert_pairs(cu_pairs, count);
}

#endif
