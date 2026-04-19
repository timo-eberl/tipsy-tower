#include "blick_adapter.h"
#include "tics_internal.h"
#include "tics_math.h"

#include <stb_ds.h>

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// Handles world creation and configuration, memory management, and the administration of bodies and
// shapes (creation, destruction, storage).

tics_world* tics_world_create(tics_world_desc desc) {
	// calloc to zero-initialize the memory, ensuring stb_ds pointers are NULL
	tics_world* world = (tics_world*)calloc(1, sizeof(tics_world));
	if (!world) return NULL;

	// config
	world->gravity = desc.gravity;
	world->air_fric_lin = desc.air_friction_linear;
	world->air_fric_ang = desc.air_friction_angular;

	// stb_ds arrays and maps start as NULL, which is valid.

	// Initialize counters to 1 (0 is reserved for invalid handles)
	world->body_id_counter = 1;
	world->shape_id_counter = 1;

	world->rigid_bodies_dirty = true;
	world->static_bodies_dirty = true;

#ifdef TICS_HAS_GPU_BROAD_PHASE
	world->gpu_state = gpu_broad_phase_create();
#endif

	// Force the thread pool to spin up immediately - otherwise a lag spike might happen when this
	// happens the first time during the simulation
#pragma omp parallel
	{
		// We do a trivial operation to ensure the compiler doesn't optimize it away.
		int id = omp_get_thread_num();
		(void)id;
	}

	BLICK_INIT();
	tics_transform identity_transform = {.position = {0}, .rotation = {0, 0, 0, 1}};
	// BLICK_TRANSFORM(0, identity_transform, 100.0);

	return world;
}

void tics_world_destroy(tics_world* world) {
	assert(world);
	if (!world) return;

	BLICK_SHUTDOWN();

	// Free convex collision data
	if (world->shapes) {
		size_t count = arrlen(world->shapes);
		for (size_t i = 0; i < count; ++i) {
			if (world->shapes[i].type == TICS_SHAPE_CONVEX) {
				if (world->shapes[i].data.convex.vertices) {
					free(world->shapes[i].data.convex.vertices);
				}
			}
		}
	}

	// Free stb_ds structures
	arrfree(world->rigid_bodies);
	arrfree(world->static_bodies);
	arrfree(world->shapes);
	arrfree(world->typed_proxies);
	arrfree(world->typed_proxy_map);
	arrfree(world->packed_rigid_proxies);
	arrfree(world->packed_static_proxies);
	hmfree(world->body_map);
	hmfree(world->shape_map);

#ifdef TICS_HAS_GPU_BROAD_PHASE
	gpu_broad_phase_destroy(world->gpu_state);
#endif

	free(world);
}

tics_shape_id tics_create_shape(tics_world* world, tics_shape_desc desc) {
	assert(world);

	shape_data sd;
	sd.type = desc.type;

	switch (desc.type) {
	case TICS_SHAPE_SPHERE:
		sd.data.sphere.center = desc.data.sphere.center;
		sd.data.sphere.radius = desc.data.sphere.radius;
		break;
	case TICS_SHAPE_CONVEX:
		// We copy the vertex data and remove duplicate vertices
		if (desc.data.convex.vertices && desc.data.convex.vertex_count > 0) {
			// Allocate worst-case size first (assuming no duplicates)
			size_t max_size = sizeof(tics_vec3) * desc.data.convex.vertex_count;
			sd.data.convex.vertices = (tics_vec3*)malloc(max_size);

			if (sd.data.convex.vertices) {
				int unique_count = 0;
				for (int i = 0; i < desc.data.convex.vertex_count; ++i) {
					tics_vec3 v = desc.data.convex.vertices[i];
					bool is_duplicate = false;

					// Check if exact vertex already exists in our new list
					for (int j = 0; j < unique_count; ++j) {
						if (memcmp(&sd.data.convex.vertices[j], &v, sizeof(tics_vec3)) == 0) {
							is_duplicate = true;
							break;
						}
					}

					if (!is_duplicate) { sd.data.convex.vertices[unique_count++] = v; }
				}

				// Resize to fit actual count to save memory
				if (unique_count < desc.data.convex.vertex_count) {
					tics_vec3* shrunk = (tics_vec3*)realloc(sd.data.convex.vertices,
															sizeof(tics_vec3) * unique_count);
					if (shrunk) sd.data.convex.vertices = shrunk;
				}
				sd.data.convex.count = unique_count;
			}
			else { sd.data.convex.count = 0; }
		}
		else {
			sd.data.convex.vertices = NULL;
			sd.data.convex.count = 0;
		}
		break;
	default:
		assert(false);
		return 0;
	}

	tics_shape_id id = world->shape_id_counter;
	world->shape_id_counter++;

	sd.id = id;

	// Add to array
	arrput(world->shapes, sd);
	// Add ID -> Index mapping
	size_t index = arrlen(world->shapes) - 1;
	hmput(world->shape_map, id, index);

	return id;
}

void tics_destroy_shape(tics_world* world, tics_shape_id shape) {
	assert(world);

	ptrdiff_t map_idx = hmgeti(world->shape_map, shape);
	if (map_idx == -1) return;

	size_t index_to_remove = world->shape_map[map_idx].value;

	// Handle resource cleanup for convex shape
	if (world->shapes[index_to_remove].type == TICS_SHAPE_CONVEX) {
		if (world->shapes[index_to_remove].data.convex.vertices) {
			free(world->shapes[index_to_remove].data.convex.vertices);
		}
	}

	// Swap and Pop Logic for Shapes
	size_t last_index = arrlen(world->shapes) - 1;
	if (index_to_remove != last_index) {
		// Move last element to hole
		world->shapes[index_to_remove] = world->shapes[last_index];

		// Update the map for the moved shape.
		for (size_t i = 0; i < hmlen(world->shape_map); ++i) {
			if (world->shape_map[i].value == last_index) {
				world->shape_map[i].value = index_to_remove;
				break;
			}
		}
	}
	arrsetlen(world->shapes, last_index);

	// Remove from map
	hmdel(world->shape_map, shape);
}

static body_ref get_body(const tics_world* world, tics_body_id id) {
	// we get a compiler error when using a const world because of hmgeti, so we cast to non const
	// and trust
	tics_world* non_const_world = (tics_world*)world;
	ptrdiff_t idx = hmgeti(non_const_world->body_map, id);
	if (idx == -1) {
		assert(false && "Body doesn't exist");
		return (body_ref){.type = -1};
	}

	body_ref ref = world->body_map[idx].value;

	if (ref.type == RIGID_BODY) {
		assert(ref.index < (size_t)arrlen(world->rigid_bodies) && "Invalid index");
	}
	else if (ref.type == STATIC_BODY) {
		assert(ref.index < (size_t)arrlen(world->static_bodies) && "Invalid index");
	}
	else { assert(false && "Body type not implemented"); }

	return ref;
}

tics_body_id tics_world_add_static_body(tics_world* world, tics_static_body_desc desc) {
	assert(world);

	// Look up shape
	ptrdiff_t shape_map_idx = hmgeti(world->shape_map, desc.shape);
	if (shape_map_idx == -1) return 0;

	size_t shape_index = world->shape_map[shape_map_idx].value;

	tics_body_id id = world->body_id_counter;
	world->body_id_counter++;

	static_body_data sb;
	sb.id = id;
	// Copy shape data for cache locality (except mesh pointer which is shared)
	sb.shape = world->shapes[shape_index];
	sb.transform = desc.transform;
	// ensure the rotation is a normalized quaternion, even for bad input data
	sb.transform.rotation = quat_normalize(sb.transform.rotation);
	sb.elasticity = desc.elasticity;

	// Calculate AABB once when adding static body and use it later in broad phase
	sb.aabb = calculate_aabb(&sb.shape, sb.transform);

	arrput(world->static_bodies, sb);
	size_t index = arrlen(world->static_bodies) - 1;

	body_ref ref = {STATIC_BODY, index};
	hmput(world->body_map, id, ref);

	world->static_bodies_dirty = true;

	BLICK_DRAW_SHAPE(0, sb.shape, sb.transform, 0xFF444444, false);
	BLICK_DRAW_SHAPE(0, sb.shape, sb.transform, 0xFF000000, true);

	return id;
}

tics_body_id tics_world_add_rigid_body(tics_world* world, tics_rigid_body_desc desc) {
	assert(world);

	ptrdiff_t shape_map_idx = hmgeti(world->shape_map, desc.shape);
	if (shape_map_idx == -1) return 0;

	size_t shape_index = world->shape_map[shape_map_idx].value;

	tics_body_id id = world->body_id_counter;
	world->body_id_counter++;

	rigid_body_data rb;
	rb.id = id;
	rb.shape = world->shapes[shape_index];
	rb.transform = desc.transform;
	// ensure the rotation is a normalized quaternion, even for bad input data
	rb.transform.rotation = quat_normalize(rb.transform.rotation);
	rb.linear_velocity = desc.linear_velocity;
	rb.angular_velocity = desc.angular_velocity;
	assert(desc.mass > 0.0f);
	rb.mass = desc.mass;
	rb.inv_mass = 1.0f / desc.mass;
	rb.elasticity = desc.elasticity;
	rb.gravity_scale = desc.gravity_scale;

	// Calculate Inverse Inertia: Appriximate all shapes as a solid sphere
	float r_sq = 1.0f;
	if (rb.shape.type == TICS_SHAPE_SPHERE) {
		float r = rb.shape.data.sphere.radius;
		r_sq = r * r;
	}
	else if (rb.shape.type == TICS_SHAPE_CONVEX) {
		// use maximum distance to center for the radius (approximation)
		float max_sq = 0.0f;
		for (size_t i = 0; i < rb.shape.data.convex.count; ++i) {
			float d2 = vec3_length_sq(rb.shape.data.convex.vertices[i]);
			if (d2 > max_sq) max_sq = d2;
		}
		r_sq = max_sq;
	}
	// Solid Sphere Inertia: I = 0.4 * mass * r^2
	rb.inv_inertia = 1.0f / (0.4 * desc.mass * r_sq);

	arrput(world->rigid_bodies, rb);
	size_t index = arrlen(world->rigid_bodies) - 1;

	body_ref ref = {RIGID_BODY, index};
	hmput(world->body_map, id, ref);

	world->rigid_bodies_dirty = true;

	return id;
}
void tics_world_remove_body(tics_world* world, tics_body_id id) {
	assert(world);

	body_ref ref = get_body(world, id);

	if (ref.type == RIGID_BODY) {
		size_t remove_idx = ref.index;
		size_t last_idx = arrlen(world->rigid_bodies) - 1;

		if (remove_idx != last_idx) {
			// Swap with last
			rigid_body_data* last_body = &world->rigid_bodies[last_idx];
			rigid_body_data* target = &world->rigid_bodies[remove_idx];

			// Update the map for the swapped body
			tics_body_id moved_id = last_body->id;
			ptrdiff_t moved_map_idx = hmgeti(world->body_map, moved_id);
			if (moved_map_idx != -1) { world->body_map[moved_map_idx].value.index = remove_idx; }

			*target = *last_body; // Move data
		}
		arrsetlen(world->rigid_bodies, last_idx);
	}
	else if (ref.type == STATIC_BODY) {
		size_t remove_idx = ref.index;
		size_t last_idx = arrlen(world->static_bodies) - 1;

		if (remove_idx != last_idx) {
			static_body_data* last_body = &world->static_bodies[last_idx];
			static_body_data* target = &world->static_bodies[remove_idx];

			tics_body_id moved_id = last_body->id;
			ptrdiff_t moved_map_idx = hmgeti(world->body_map, moved_id);
			if (moved_map_idx != -1) { world->body_map[moved_map_idx].value.index = remove_idx; }

			*target = *last_body;
		}
		arrsetlen(world->static_bodies, last_idx);
	}
	else { assert(false && "Body type not implemented"); }

	world->rigid_bodies_dirty = ref.type == RIGID_BODY;
	world->static_bodies_dirty = ref.type == STATIC_BODY;

	hmdel(world->body_map, id);
}

tics_transform tics_body_get_transform(const tics_world* world, tics_body_id id) {
	assert(world);
	body_ref ref = get_body(world, id);

	if (ref.type == RIGID_BODY) { return world->rigid_bodies[ref.index].transform; }
	else if (ref.type == STATIC_BODY) { return world->static_bodies[ref.index].transform; }
	else {
		assert(false && "Body type not implemented");
		return (tics_transform){.position = {0}, .rotation = {0, 0, 0, 1}};
	}
}

tics_vec3 tics_body_get_velocity(const tics_world* world, tics_body_id id) {
	assert(world);
	body_ref ref = get_body(world, id);

	if (ref.type == RIGID_BODY) { return world->rigid_bodies[ref.index].linear_velocity; }
	else if (ref.type == STATIC_BODY) { return (tics_vec3){0}; }
	else {
		assert(false && "Body type not implemented");
		return (tics_vec3){0};
	}
}

void tics_body_set_velocity(tics_world* world, tics_body_id id, tics_vec3 velocity) {
	assert(world);
	body_ref ref = get_body(world, id);

	if (ref.type == RIGID_BODY) { world->rigid_bodies[ref.index].linear_velocity = velocity; }
	// Static bodies don't have velocity, so this is a no-op for them
}

void tics_debug_upload_shape_mesh(tics_shape_id id, const tics_vec3* vertices,
								  const uint32_t* indices, uint32_t i_count) {
	BLICK_UPLOAD_MESH_INDEXED(id, vertices, indices, i_count);
}
