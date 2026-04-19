#include "models_data.h"
#include "sokol_util.h"
#include "tics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SOKOL_IMPL
#if defined(__EMSCRIPTEN__)
#define SOKOL_GLES3
#elif defined(_WIN32)
#define SOKOL_D3D11
#elif defined(__APPLE__)
#define SOKOL_METAL
#else
#define SOKOL_GLCORE
#endif

#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"

#define PHYSICS_TIMESTEP (1.0f / 120.0f)
#define MAX_FRAME_TIME 0.25f

#define SPHERE_RADIUS 1.0f
#define DYNAMIC_BODIES 1000
#define GROUND_EXTENT 5
#define MAX_STATIC_GROUNDS (GROUND_EXTENT * GROUND_EXTENT)

#if defined(__EMSCRIPTEN__) || defined(__ANDROID__)
#define SHADER_PREFIX "#version 300 es\n"
#else
#define SHADER_PREFIX "#version 330\n"
#endif

// The vertex shader passes the local position to the fragment shader.
// Because the sphere is centered at the origin, the local position serves as the normal direction.
static const char* vs_source = SHADER_PREFIX 
	"uniform mat4 mvp;\n"
	"uniform mat4 model;\n"
	"layout(location=0) in vec3 position;\n"
	"out vec3 v_local_pos;\n"
	"out vec3 v_world_normal;\n"
	"void main() {\n"
	"  gl_Position = mvp * vec4(position, 1.0);\n"
	"  v_local_pos = position;\n"
	"  v_world_normal = mat3(model) * position;\n"
	"}\n";

static const char* fs_source = SHADER_PREFIX 
	"precision highp float;\n"
	"uniform vec4 color;\n"
	"in vec3 v_local_pos;\n"
	"in vec3 v_world_normal;\n"
	"out vec4 frag_color;\n"
	"void main() {\n"
	"  vec3 normal = normalize(v_world_normal);\n"
	"  vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));\n"
	"  float n_dot_l = max(dot(normal, light_dir), 0.15);\n"
	"  vec3 grid = step(fract(v_local_pos * 4.0), vec3(0.08));\n"
	"  float is_grid = max(max(grid.x, grid.y), grid.z);\n"
	"  vec3 final_color = mix(color.rgb, vec3(0.05), is_grid * 0.4) * n_dot_l;\n"
	"  frag_color = vec4(final_color, color.a);\n"
	"}\n";

typedef struct {
	su_mat4 mvp;
	su_mat4 model;
} vs_params_t;

typedef struct {
	float color[4];
} fs_params_t;

typedef struct {
	tics_body_id body;
} game_entity;

static struct {
	tics_world* world;
	float accumulator;

	su_input input;
	su_camera camera;

	sg_pass_action pass_action;
	sg_pipeline pip;
	sg_bindings bind_sphere;

	game_entity entities[DYNAMIC_BODIES];
	int entity_count;

	su_vec3 ground_positions[MAX_STATIC_GROUNDS];
	int ground_count;

	tics_shape_id sh_sphere;
	float max_tower_height;
} state;

static su_mat4 mat4_from_tics(tics_transform t) {
	float x = t.rotation.x, y = t.rotation.y, z = t.rotation.z, w = t.rotation.w;
	float x2 = x + x, y2 = y + y, z2 = z + z;
	float xx = x * x2, xy = x * y2, xz = x * z2;
	float yy = y * y2, yz = y * z2, zz = z * z2;
	float wx = w * x2, wy = w * y2, wz = w * z2;

	su_mat4 m = {0};
	m.m[0] = 1.0f - (yy + zz); m.m[1] = xy + wz; m.m[2] = xz - wy;
	m.m[4] = xy - wz; m.m[5] = 1.0f - (xx + zz); m.m[6] = yz + wx;
	m.m[8] = xz + wy; m.m[9] = yz - wx; m.m[10] = 1.0f - (xx + yy);
	m.m[12] = t.position.x; m.m[13] = t.position.y; m.m[14] = t.position.z;
	m.m[15] = 1.0f;
	return m;
}

static sg_buffer make_vbuf(const tics_vec3* data, size_t count) {
	return sg_make_buffer(&(sg_buffer_desc){
		.usage = {.vertex_buffer = true, .immutable = true},
		.data = (sg_range){data, count * sizeof(tics_vec3)}
	});
}

static sg_buffer make_ibuf(const uint32_t* data, size_t count) {
	return sg_make_buffer(&(sg_buffer_desc){
		.usage = {.index_buffer = true, .immutable = true},
		.data = (sg_range){data, count * sizeof(uint32_t)}
	});
}

static void spawn_sphere(void) {
	if (state.entity_count >= DYNAMIC_BODIES) return;

	su_vec3 forward = su_vec3_normalize(su_vec3_sub(state.camera.target, state.camera.position));
	su_vec3 spawn_pos = su_vec3_add(state.camera.position, su_vec3_scale(forward, 4.0f));

	tics_body_id body = tics_world_add_rigid_body(state.world, (tics_rigid_body_desc){
		.shape = state.sh_sphere,
		.mass = 2.0f,
		.elasticity = 0.8f,
		.gravity_scale = 1.0f,
		.transform = {
			.position = {spawn_pos.x, spawn_pos.y, spawn_pos.z},
			.rotation = {0.0f, 0.0f, 0.0f, 1.0f}
		}
	});

	state.entities[state.entity_count++] = (game_entity){ .body = body };
}

static void init(void) {
	sg_setup(&(sg_desc){.environment = sglue_environment()});
	
	state.pass_action = (sg_pass_action){
		.colors[0] = {
			.load_action = SG_LOADACTION_CLEAR,
			.clear_value = {0.08f, 0.08f, 0.09f, 1.0f}
		}
	};

	state.camera.position = (su_vec3){0.0f, 5.0f, 15.0f};
	state.camera.target = (su_vec3){0.0f, 2.0f, 0.0f};
	state.camera.up = (su_vec3){0.0f, 1.0f, 0.0f};

	sg_shader shd = sg_make_shader(&(sg_shader_desc){
		.vertex_func.source = vs_source,
		.fragment_func.source = fs_source,
		.uniform_blocks[0] = {
			.stage = SG_SHADERSTAGE_VERTEX,
			.size = sizeof(vs_params_t),
			.glsl_uniforms = {
				[0] = {.type = SG_UNIFORMTYPE_MAT4, .glsl_name = "mvp"},
				[1] = {.type = SG_UNIFORMTYPE_MAT4, .glsl_name = "model"}
			}
		},
		.uniform_blocks[1] = {
			.stage = SG_SHADERSTAGE_FRAGMENT,
			.size = sizeof(fs_params_t),
			.glsl_uniforms[0] = {.type = SG_UNIFORMTYPE_FLOAT4, .glsl_name = "color"}
		}
	});

	state.pip = sg_make_pipeline(&(sg_pipeline_desc){
		.shader = shd,
		.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3,
		.index_type = SG_INDEXTYPE_UINT32,
		.depth = {.compare = SG_COMPAREFUNC_LESS_EQUAL, .write_enabled = true},
		.cull_mode = SG_CULLMODE_BACK,
		.face_winding = SG_FACEWINDING_CCW,
	});

	state.bind_sphere.vertex_buffers[0] = 
		make_vbuf(icosphere_Icosphere_vertices, icosphere_vertex_buffer_sizes[0]);
	state.bind_sphere.index_buffer = 
		make_ibuf(icosphere_Icosphere_indices, icosphere_index_buffer_sizes[0]);

	state.world = tics_world_create((tics_world_desc){
		.gravity = {0.0f, -9.81f, 0.0f},
		.air_friction_linear = 0.1f,
		.air_friction_angular = 0.8f
	});

	state.sh_sphere = tics_create_shape(state.world, (tics_shape_desc){
		.type = TICS_SHAPE_SPHERE,
		.data.sphere = {.center = {0, 0, 0}, .radius = SPHERE_RADIUS}
	});

	tics_debug_upload_shape_mesh(state.sh_sphere, icosphere_Icosphere_vertices,
								 icosphere_Icosphere_indices, icosphere_index_buffer_sizes[0]);

	// Generate regular square ground grid
	float spacing = 2.0f * SPHERE_RADIUS;
	int half_ext = GROUND_EXTENT / 2;

	for (int z = -half_ext; z <= half_ext; z++) {
		for (int x = -half_ext; x <= half_ext; x++) {
			if (state.ground_count >= MAX_STATIC_GROUNDS) break;
			
			su_vec3 pos = {x * spacing, 0.0f, z * spacing};
			
			state.ground_positions[state.ground_count++] = pos;

			tics_world_add_static_body(state.world, (tics_static_body_desc){
				.transform = {.position = {pos.x, pos.y, pos.z}, .rotation = {0, 0, 0, 1}},
				.shape = state.sh_sphere,
				.elasticity = 0.8f
			});
		}
	}
}

static void frame(void) {
	float dt = (float)sapp_frame_duration();
	float clamped_dt = dt > MAX_FRAME_TIME ? MAX_FRAME_TIME : dt;

	su_camera_navigate(&state.camera, &state.input, clamped_dt);

	state.accumulator += clamped_dt;
	while (state.accumulator >= PHYSICS_TIMESTEP) {
		tics_world_step(state.world, PHYSICS_TIMESTEP);
		state.accumulator -= PHYSICS_TIMESTEP;
	}

	float current_highest_y = 0.0f;

	for (int i = 0; i < state.entity_count;) {
		tics_transform tf = tics_body_get_transform(state.world, state.entities[i].body);

		if (tf.position.y < -30.0f) {
			tics_world_remove_body(state.world, state.entities[i].body);
			state.entities[i] = state.entities[state.entity_count - 1];
			state.entity_count--;
		} else {
			if (tf.position.y > current_highest_y) {
				current_highest_y = tf.position.y;
			}
			i++;
		}
	}

	float frame_tower_height = current_highest_y > 0.0f ? current_highest_y + SPHERE_RADIUS : 0.0f;
	if (frame_tower_height > state.max_tower_height) {
		state.max_tower_height = frame_tower_height;
		printf("New Tower Height Record: %.2f\n", state.max_tower_height);
	}

	float w = (float)sapp_width();
	float h = (float)sapp_height();
	su_mat4 proj = su_mat4_persp(45.0f * (3.14159f / 180.0f), w / h, 0.1f, 1000.0f);
	su_mat4 view = su_mat4_look_at(state.camera.position, state.camera.target, state.camera.up);
	su_mat4 vp = su_mat4_mul(proj, view);

	sg_begin_pass(&(sg_pass){.action = state.pass_action, .swapchain = sglue_swapchain()});
	sg_apply_pipeline(state.pip);
	sg_apply_bindings(&state.bind_sphere);

	fs_params_t fs_static = {.color = {0.15f, 0.4f, 0.8f, 1.0f}};
	sg_apply_uniforms(1, &SG_RANGE(fs_static));

	for (int i = 0; i < state.ground_count; i++) {
		su_vec3 pos = state.ground_positions[i];
		su_mat4 model = mat4_from_tics((tics_transform){{pos.x, pos.y, pos.z}, {0, 0, 0, 1}});
		vs_params_t vs = {.mvp = su_mat4_mul(vp, model), .model = model};
		
		sg_apply_uniforms(0, &SG_RANGE(vs));
		sg_draw(0, icosphere_index_buffer_sizes[0], 1);
	}

	fs_params_t fs_dyn = {.color = {0.95f, 0.5f, 0.1f, 1.0f}};
	sg_apply_uniforms(1, &SG_RANGE(fs_dyn));

	for (int i = 0; i < state.entity_count; i++) {
		tics_transform tf = tics_body_get_transform(state.world, state.entities[i].body);
		su_mat4 model = mat4_from_tics(tf);
		vs_params_t vs = {.mvp = su_mat4_mul(vp, model), .model = model};

		sg_apply_uniforms(0, &SG_RANGE(vs));
		sg_draw(0, icosphere_index_buffer_sizes[0], 1);
	}

	sg_end_pass();
	sg_commit();

	su_input_end_frame(&state.input);
}

static void event(const sapp_event* ev) {
	su_input_update(&state.input, ev);

	if (ev->type == SAPP_EVENTTYPE_KEY_DOWN && !ev->key_repeat) {
		if (ev->key_code == SAPP_KEYCODE_SPACE) {
			spawn_sphere();
		}
	}
}

static void cleanup(void) {
	tics_world_destroy(state.world);
	sg_shutdown();
}

sapp_desc sokol_main(int argc, char* argv[]) {
	(void)argc;
	(void)argv;
	return (sapp_desc){
		.init_cb = init,
		.frame_cb = frame,
		.cleanup_cb = cleanup,
		.event_cb = event,
		.width = 1280,
		.height = 720,
		.window_title = "Tics Tower Builder",
		.icon.sokol_default = true
	};
}
