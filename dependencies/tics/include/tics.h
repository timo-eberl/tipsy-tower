#ifndef TICS_H
#define TICS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// clang-format doesn't allow one-line structs
// clang-format off

typedef struct { float x, y, z; } tics_vec3;
typedef struct { float x, y, z, w; } tics_quat; // x,y,z,w

// Opaque handle to the simulation world
typedef struct tics_world tics_world;
// Handle for all types of bodies. 0 is always invalid.
typedef uint32_t tics_body_id;
// Handle for collision shapes. 0 is always invalid.
// Using handles allows sharing one convex data buffer among many bodies.
typedef uint32_t tics_shape_id;

// Transform consisting of position and rotation (quaternion). Scaling is unsupported as the scale
// of a rigid body can per definition not change.
typedef struct { tics_vec3 position; tics_quat rotation; } tics_transform;
typedef enum { TICS_SHAPE_SPHERE, TICS_SHAPE_CONVEX } tics_shape_type;

// Configuration used to initialize the world
typedef struct {
	// Global acceleration vector applied to all dynamic bodies, e.g. {0, -9.81, 0}
	tics_vec3 gravity;
	// Global linear damping coefficient [0.0 - 1.0]. Simulates resistance to translation.
	float air_friction_linear;
	// Global angular damping coefficient [0.0 - 1.0]. Simulates resistance to rotation.
	float air_friction_angular;
} tics_world_desc;
// Configuration used to create a shape resource
typedef struct {
	tics_shape_type type;
	union {
		struct { tics_vec3 center; float radius; } sphere;
		// Vertices will be copied on creation
		struct { const tics_vec3* vertices; size_t vertex_count; } convex;
	} data;
} tics_shape_desc;
// Configuration for creating a Static Body (Ground, Walls)
typedef struct {
	tics_transform transform;
	tics_shape_id shape; // Reference to a pre-created shape
	float elasticity;	 // [0.0 - 1.0]
} tics_static_body_desc;
// Configuration for creating a Rigid Body (Moving objects)
typedef struct {
	tics_transform transform;
	tics_shape_id shape; // Reference to a pre-created shape

	tics_vec3 linear_velocity;
	// Angular velocity vector (radians per second): Direction is the axis, magnitude is the speed.
	tics_vec3 angular_velocity;

	float mass;
	float elasticity; // [0.0 - 1.0]
	float gravity_scale;
} tics_rigid_body_desc;

// clang-format on

// Create a new physics world. Returns NULL on failure.
tics_world* tics_world_create(tics_world_desc desc);
// Destroy the world and free all internal resources/bodies.
void tics_world_destroy(tics_world* world);

// Steps the simulation forward by delta (in seconds).
void tics_world_step(tics_world* world, float delta);

// Creates a shape resource. For convex shapes, exact duplicate vertices are automatically removed.
// This is essential for efficiently importing flat-shaded or hard-edged meshes, where multiple
// vertices often exist at the same position to support distinct normals/UVs, whereas the physics
// shape requires unique positions only. Returns 0 on failure.
tics_shape_id tics_create_shape(tics_world* world, tics_shape_desc desc);
// Destroys a shape. Note: Do not destroy a shape while it is in use by a body.
void tics_destroy_shape(tics_world* world, tics_shape_id shape);

// Adds a static body. Returns 0 on failure.
tics_body_id tics_world_add_static_body(tics_world* world, tics_static_body_desc desc);
// Adds a rigid body. Returns 0 on failure.
tics_body_id tics_world_add_rigid_body(tics_world* world, tics_rigid_body_desc desc);
// Removes and destroys a body. The ID becomes invalid.
void tics_world_remove_body(tics_world* world, tics_body_id id);

// Get the transform. Useful for rendering synchronization.
// Returns identity if ID is invalid.
tics_transform tics_body_get_transform(const tics_world* world, tics_body_id id);

// Get the world space linear velocity at the bodies center.
// Returns (0,0,0) for static bodies or if ID is invalid.
tics_vec3 tics_body_get_velocity(const tics_world* world, tics_body_id id);

// Set the world space linear velocity at the bodies center.
// This is a no-op for static bodies or if ID is invalid.
void tics_body_set_velocity(tics_world* world, tics_body_id id, tics_vec3 velocity);

// Uploads indexed mesh geometry belonging to a shape.
// Only serves visualization inside the debug viewer.
// This is a no-op if the library was built without debug visualization support.
void tics_debug_upload_shape_mesh(tics_shape_id id, const tics_vec3* vertices,
								  const uint32_t* indices, uint32_t i_count);

#ifdef __cplusplus
}
#endif

#endif // TICS_H
