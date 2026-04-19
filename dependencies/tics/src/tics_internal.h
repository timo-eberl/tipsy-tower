#ifndef TICS_INTERNAL_H
#define TICS_INTERNAL_H

#include "tics.h"

#ifdef _OPENMP
#include <omp.h>
#else
// Fallbacks if OpenMP isn't available (e.g. on web)
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#endif

// Internal runtime data for shapes and bodies differ from the descriptors that are used to
// initialize them.

// clang-format off

// Internal runtime storage for a shape
typedef struct {
	tics_shape_type type;
	union {
		struct { tics_vec3 center; float radius; } sphere;
		struct { tics_vec3* vertices; size_t count; } convex;
	} data;
	tics_shape_id id; // Back-reference to ID, needed for debug drawing
} shape_data;

// Axis-aligned bounding box
typedef struct { tics_vec3 min; tics_vec3 max; } aabb;
// Axis-aligned bounding box with different memory layout
typedef struct { float min_x, max_x, min_y, max_y, min_z, max_z; } packed_aabb;

typedef struct {
	// store shape data directly, because the shape data is small and looking up the shape in a map
	// is slow. the mesh data (which might be big) will still be shared.
	shape_data shape;
	tics_transform transform;
	aabb aabb;		 // Computed once at initialization
	tics_body_id id; // Back-reference to ID, needed for swap-and-pop updates
	float elasticity;
} static_body_data;

typedef struct {
	// store shape data directly, because the shape data is small and looking up the shape in a map
	// is slow. the mesh data (which might be big) will still be shared.
	shape_data shape;
	tics_transform transform;

	tics_vec3 linear_velocity;  // world space
	tics_vec3 angular_velocity; // world space

	tics_body_id id; // Back-reference to ID, needed for swap-and-pop updates

	float mass;
	float inv_mass; // Pre-calculated 1.0f/mass for solvers
	float inv_inertia;
	float elasticity;
	float gravity_scale;
} rigid_body_data;

// If we were to use a bare enum, it would be 4 byte
typedef uint8_t body_type;
enum { STATIC_BODY = 0, RIGID_BODY = 1 };
// Holds type and index into either rigid_bodies or static_bodies array
typedef struct { body_type type; size_t index; } body_ref;
typedef struct { tics_body_id key; body_ref value; } body_map_entry;
// Holds index into shapes array
typedef struct { tics_shape_id key; size_t value; } shape_map_entry;

// clang-format on

typedef struct {
	// a and b are the points where each shape penetrates the other most
	tics_vec3 point_a;
	tics_vec3 point_b;
	tics_vec3 normal; // penetration vector direction
	float depth;	  // penetration vector length
	bool has_collision;
} collision_result;

typedef struct {
	body_ref body_a_ref;
	body_ref body_b_ref;

	collision_result result;

	// The total impulse applied over all iterations in one frame.
	// Initialized from the persistent collision hash map before the solver starts.
	float accumulated_impulse;
	float target_velocity; // Pre-calculated before iterations
	float effective_mass;  // Pre-calculated before iterations
} collision;

// Unique identifier for a pair of bodies
typedef struct {
	tics_body_id id_a;
	tics_body_id id_b;
} manifold_key;

// The Value: Data we need to persist across frames
typedef struct {
	manifold_key key; // Needed for stb_ds if using hmput/hmget with structs

	float accumulated_impulse;

	// We store the local anchor points to detect if the contact has jumped to a different location
	tics_vec3 local_point_a;
	tics_vec3 local_point_b;
} manifold_cache_entry;

// Broad phase Proxy stripped of all physics properties (velocity, mass, etc).
typedef struct {
	aabb aabb;
	// id into rigid_bodies or static_bodies. Type is implied by which array this proxy resides in.
	uint32_t index;
} broad_phase_proxy;

// Alternative broad phase Proxy with type information
typedef struct {
	aabb aabb;
	uint32_t index;
	body_type type;
} broad_phase_proxy_typed;

struct tics_world {
	// Config
	tics_vec3 gravity;
	float air_fric_lin;
	float air_fric_ang;

	// --- Dense Data Arrays (stb_ds arrays) ---

	rigid_body_data* rigid_bodies;
	static_body_data* static_bodies;
	shape_data* shapes;

	// dirty flags, will be cleared at the end of the frame.

	// True when rigid bodies are added or removed.
	// When false, rigid bodies might still have moved.
	bool rigid_bodies_dirty;
	// True when static bodies are added or removed.
	// When false, it is guaranteed that static bodies did not move.
	bool static_bodies_dirty;

	// --- Lookups (stb_ds hash maps) ---

	// We use a unified map for all body types.
	// Here's a table outlining when a lookup is required:
	// | Operation          | Input          | Map Read?                        |
	// | :----------------- | :------------- | :------------------------------- |
	// | API Call           | `tics_body_id` | YES                              |
	// | Constraints Solver | `tics_body_id` | NO (can be cached in constraint) |
	// | Dynamics           | body arrays    | NO                               |
	// | Collision Det.     | body arrays    | NO                               |
	// | Event Callback     | rb             | NO (can be cached in rb)         |

	// ID -> {Type, Index}
	body_map_entry* body_map;
	// map for shapes: ID -> Index
	shape_map_entry* shape_map;

	// Map that stores previous collisions
	manifold_cache_entry* manifold_map;

	// --- ID Generation ---
	// Strictly Increasing IDs: This effectively eliminates "ABA problems" (where you access a
	// reused slot thinking it's the old object) without needing generation counters in the index.
	// Initialized to 1 (0 is invalid)

	uint32_t body_id_counter;
	uint32_t shape_id_counter;

	// Broadphase state, used by Sweep and Prune
	broad_phase_proxy_typed* typed_proxies; // Persistent stb_ds array
	uint32_t* typed_proxy_map;				// Maps Rigid Body Index -> typed_proxies Index

	packed_aabb* packed_rigid_proxies;	// Persistent stb_ds array. Order mirrors rigid_bodies.
	packed_aabb* packed_static_proxies; // Persistent stb_ds array. Order mirrors static_bodies.

#ifdef TICS_HAS_GPU_BROAD_PHASE
	// Broadphase state, used by GPU broad phase
	void* gpu_state; // Opaque pointer to internal state (avoids header dependency)
#endif
};

// The output of the broadphase. Represents a potential collision.
// We output body_ref here so the Narrowphase knows exactly which arrays to look into to find the
// shape data.
typedef struct {
	body_ref a;
	body_ref b;
} broad_phase_pair;

aabb calculate_aabb(const shape_data* shape, tics_transform t);

// Proxy Builders
// These functions iterate over the world bodies, compute/fetch the AABB, and return a new dynamic
// array (stb_ds) of proxies. Separation allows us to treat Static bodies as passive in the
// broadphase.
broad_phase_proxy* build_rigid_proxies(const tics_world* world);
broad_phase_proxy* build_static_proxies(const tics_world* world);

void update_typed_proxies(tics_world* world);

// Builds the rigid body packed AABB array into world->packed_rigid_proxies.
// When static_bodies_dirty is true also builds world->packed_static_proxies.
void update_packed_proxies(tics_world* world);

// Broad phase collision detection - Multiple versions
// Takes two lists to enable optimizations (we do not need to check static vs static).

broad_phase_pair* broad_phase_naive(const broad_phase_proxy* rigids, size_t rigid_count,
									const broad_phase_proxy* statics, size_t static_count);

broad_phase_pair* broad_phase_naive_parallel(const broad_phase_proxy* rigids, size_t rigid_count,
											 const broad_phase_proxy* statics, size_t static_count);

// Sweep and Prune (modifies typed_proxies)
broad_phase_pair* broad_phase_sap(broad_phase_proxy_typed* proxies, size_t count,
								  uint32_t* proxy_map);

#ifdef TICS_HAS_GPU_BROAD_PHASE
// GPU broad phase adapter functions (GPU-agnostic API)
// These functions abstract away GPU details from the rest of tics.

// Creates GPU-side state. Returns opaque handle. Call once at world creation.
void* gpu_broad_phase_create(void);
// Destroys GPU-side state. Call once at world destruction.
void gpu_broad_phase_destroy(void* state);
// Runs GPU broad phase. Returns stb_dsy array of pairs; caller frees.
broad_phase_pair* gpu_broad_phase_run_grid_a(void* gpu_state, packed_aabb* packed_rigid_proxies,
											 size_t rigid_count, packed_aabb* packed_static_proxies,
											 size_t static_count, bool statics_changed);
broad_phase_pair* gpu_broad_phase_run_grid_b_half_shell(void* gpu_state,
														packed_aabb* packed_rigid_proxies,
														size_t rigid_count,
														packed_aabb* packed_static_proxies,
														size_t static_count, bool statics_changed);
broad_phase_pair* gpu_broad_phase_run_grid_b_naive(void* gpu_state,
												   packed_aabb* packed_rigid_proxies,
												   size_t rigid_count,
												   packed_aabb* packed_static_proxies,
												   size_t static_count, bool statics_changed);
broad_phase_pair* gpu_broad_phase_run_brute_force(void* gpu_state,
												  packed_aabb* packed_rigid_proxies,
												  size_t rigid_count,
												  packed_aabb* packed_static_proxies,
												  size_t static_count, bool statics_changed);

#endif

// Narrow phase collision detection
// Takes the list of pairs found by the broadphase. Requires pointers to the body arrays to
// resolve the indices in 'broad_phase_pair' to actual shape data for the geometric checks.
collision* narrow_phase(const broad_phase_pair* pairs, size_t pair_count,
						const rigid_body_data* r_bodies, const static_body_data* s_bodies);

collision_result collision_test(const shape_data* a, tics_transform at, const shape_data* b,
								tics_transform bt);

// Applies semi-implicit euler. Does not match the mathematically correct solution (it will loose
// energy). Other solutions that do (e.g. velocity verlet integration) are impractical for a
// discrete physics simulation.
void apply_gravity_and_air_friction(tics_world* world, float delta);

// Calculates the instantaneous linear velocity of a specific point on the rigid body.
// The result accounts for both the body's linear velocity and the tangential velocity.
// Input and output are in world space.
tics_vec3 get_velocity_at_point(rigid_body_data* rb, tics_vec3 point);

// 'impulse' and 'position' are in world space.
// To apply a linear impulse, apply it at the object center.
void rigid_body_apply_impulse(rigid_body_data* rb, tics_vec3 impulse, tics_vec3 position);

void prepare_velocity_solver(tics_world* world, collision* collisions);

// Calculates and applies instantaneous impulses to handle momentum transfer, restitution, and
// contact friction. This function modifies the bodies linear and angular velocities to prevent
// them from moving deeper into an intersection during the following integration step.
void resolve_velocities(tics_world* world, collision* collisions);

// Directly translates (teleports) bodies to correct geometric overlaps. It modifies the positions
// directly to enforce non-penetration without adding energy.
void resolve_penetrations(tics_world* world, collision* collisions);

// Applies velocities (linear and angular) to position and rotation.
void apply_velocities(tics_world* world, float delta);

#endif // TICS_INTERNAL_H
