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
#include "sokol_debugtext.h"

#define PHYSICS_TIMESTEP (1.0f / 120.0f)
#define MAX_FRAME_TIME 0.25f

#define SPHERE_RADIUS 1.0f
#define DYNAMIC_BODIES 1000
#define MAX_STATIC_GROUNDS 10

#define ELASTICITY 0.9f

#if defined(__EMSCRIPTEN__) || defined(__ANDROID__)
#define SHADER_PREFIX "#version 300 es\n"
#else
#define SHADER_PREFIX "#version 330\n"
#endif

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
	"  float n_dot_l = max(dot(normal, light_dir), 0.25);\n"
	"  vec3 grid = step(fract(v_local_pos * 4.0), vec3(0.08));\n"
	"  float is_grid = max(max(grid.x, grid.y), grid.z);\n"
	"  vec3 final_color = mix(color.rgb, vec3(1), is_grid * 0.5) * n_dot_l;\n"
	"  frag_color = vec4(final_color, color.a);\n"
	"}\n";

typedef struct {
	su_mat4 mvp;
	su_mat4 model;
} vs_params_t;

typedef struct {
	float color[4];
} fs_params_t;

static const char* vs_grid_source = SHADER_PREFIX 
	"uniform mat4 mvp;\n"
	"layout(location=0) in vec3 position;\n"
	"out vec3 v_pos;\n"
	"void main() {\n"
	"  gl_Position = mvp * vec4(position, 1.0);\n"
	"  v_pos = position;\n"
	"}\n";

static const char* fs_grid_source = SHADER_PREFIX 
	"precision highp float;\n"
	"in vec3 v_pos;\n"
	"out vec4 frag_color;\n"
	"void main() {\n"
	"  vec2 coord = v_pos.xz * 0.5;\n"
	"  vec2 grid = abs(fract(coord + 0.5) - 0.5) / fwidth(coord);\n"
	"  float line = min(grid.x, grid.y);\n"
	"  float alpha = 1.0 - min(line, 1.0);\n"
	"  float dist = length(v_pos.xz);\n"
	"  alpha *= clamp(1.0 - dist / 50.0, 0.0, 1.0);\n"
	"  frag_color = vec4(1.0, 1.0, 1.0, alpha * 0.3);\n"
	"}\n";

typedef struct {
	su_mat4 mvp;
} vs_grid_params_t;

typedef struct {
	tics_body_id body;
} game_entity;

typedef enum {
	SPAWN_MODE_NONE = 0,
	SPAWN_MODE_DYNAMIC = 1,
	SPAWN_MODE_STATIC = 2,
	SPAWN_MODE_DELETE = 3
} spawn_mode_t;

static struct {
	tics_world* world;
	float accumulator;

	su_input input;
	su_camera camera;

	sg_pass_action pass_action;
	sg_pipeline pip;
	sg_bindings bind_sphere;

	sg_pipeline pip_grid;
	sg_bindings bind_grid;

	game_entity entities[DYNAMIC_BODIES];
	int entity_count;

	su_vec3 ground_positions[MAX_STATIC_GROUNDS];
	tics_body_id ground_bodies[MAX_STATIC_GROUNDS];
	int ground_count;

	tics_shape_id sh_sphere;
	float max_tower_height;
	
	spawn_mode_t spawn_mode;
	int target_type; // 0 = none, 1 = dynamic, 2 = static
	int target_index;
	bool ui_visible;
} state;

static bool ray_sphere_intersect(su_vec3 ray_o, su_vec3 ray_d, su_vec3 sphere_c, float r,
								 float* out_t) {
	su_vec3 oc = su_vec3_sub(ray_o, sphere_c);
	float b = 2.0f * su_vec3_dot(oc, ray_d);
	float c = su_vec3_dot(oc, oc) - (r * r);
	float discriminant = b * b - 4.0f * c;
	if (discriminant < 0.0f) return false;
	
	float t = (-b - sqrtf(discriminant)) / 2.0f;
	if (t < 0.0f) t = (-b + sqrtf(discriminant)) / 2.0f;
	if (t < 0.0f) return false;
	
	*out_t = t;
	return true;
}

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

static su_vec3 get_dynamic_spawn_position(void) {
	su_vec3 forward = su_vec3_normalize(su_vec3_sub(state.camera.target, state.camera.position));
	return su_vec3_add(state.camera.position, su_vec3_scale(forward, 6.0f));
}

// Calculates the intersection of the camera's forward vector with the XZ plane (y=0)
static bool get_static_spawn_position(su_vec3* out_pos) {
	su_vec3 forward = su_vec3_normalize(su_vec3_sub(state.camera.target, state.camera.position));
	
	// If looking upwards or parallel to the horizon, it will never intersect the ground
	if (forward.y >= -0.0001f) return false;
	
	float t = -state.camera.position.y / forward.y;
	if (t < 0.0f) return false; // Safety check, intersection is behind camera

	*out_pos = su_vec3_add(state.camera.position, su_vec3_scale(forward, t));
	return true;
}

static void spawn_dynamic_sphere(void) {
	if (state.entity_count >= DYNAMIC_BODIES) return;

	su_vec3 spawn_pos = get_dynamic_spawn_position();

	tics_body_id body = tics_world_add_rigid_body(state.world, (tics_rigid_body_desc){
		.shape = state.sh_sphere,
		.mass = 2.0f,
		.elasticity = ELASTICITY,
		.gravity_scale = 1.0f,
		.transform = {
			.position = {spawn_pos.x, spawn_pos.y, spawn_pos.z},
			.rotation = {0.0f, 0.0f, 0.0f, 1.0f}
		}
	});

	state.entities[state.entity_count++] = (game_entity){ .body = body };
}

static void spawn_static_sphere(void) {
	if (state.ground_count >= MAX_STATIC_GROUNDS) return;
	
	su_vec3 spawn_pos;
	if (!get_static_spawn_position(&spawn_pos)) return;

	state.ground_positions[state.ground_count] = spawn_pos;

	tics_body_id body = tics_world_add_static_body(state.world, (tics_static_body_desc){
		.transform = {.position = {spawn_pos.x, 0.0f, spawn_pos.z}, .rotation = {0, 0, 0, 1}},
		.shape = state.sh_sphere,
		.elasticity = ELASTICITY
	});

	state.ground_bodies[state.ground_count++] = body;
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
	state.spawn_mode = SPAWN_MODE_DYNAMIC;
	state.ui_visible = true;

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
		.colors[0].blend = {
			.enabled = true,
			.src_factor_rgb = SG_BLENDFACTOR_SRC_ALPHA,
			.dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
			.src_factor_alpha = SG_BLENDFACTOR_ONE,
			.dst_factor_alpha = SG_BLENDFACTOR_ZERO
		}
	});

	state.bind_sphere.vertex_buffers[0] = make_vbuf(sphere_vertices, sphere_vertices_length);
	state.bind_sphere.index_buffer = make_ibuf(sphere_indices, sphere_indices_length);

	static const float quad_verts[] = {
		-100.0f, -1.0f, -100.0f,
		 100.0f, -1.0f, -100.0f,
		-100.0f, -1.0f,  100.0f,
		 100.0f, -1.0f,  100.0f
	};
	state.bind_grid.vertex_buffers[0] = sg_make_buffer(&(sg_buffer_desc){
		.usage = {.vertex_buffer = true, .immutable = true},
		.data = (sg_range){quad_verts, sizeof(quad_verts)}
	});

	sg_shader shd_grid = sg_make_shader(&(sg_shader_desc){
		.vertex_func.source = vs_grid_source,
		.fragment_func.source = fs_grid_source,
		.uniform_blocks[0] = {
			.stage = SG_SHADERSTAGE_VERTEX,
			.size = sizeof(vs_grid_params_t),
			.glsl_uniforms = {
				[0] = {.type = SG_UNIFORMTYPE_MAT4, .glsl_name = "mvp"}
			}
		}
	});

	state.pip_grid = sg_make_pipeline(&(sg_pipeline_desc){
		.shader = shd_grid,
		.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3,
		.primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP,
		.depth = {.compare = SG_COMPAREFUNC_LESS_EQUAL, .write_enabled = false},
		.colors[0].blend = {
			.enabled = true,
			.src_factor_rgb = SG_BLENDFACTOR_SRC_ALPHA,
			.dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
			.src_factor_alpha = SG_BLENDFACTOR_ONE,
			.dst_factor_alpha = SG_BLENDFACTOR_ZERO
		}
	});

	state.world = tics_world_create((tics_world_desc){
		.gravity = {0.0f, -9.81f, 0.0f},
		.air_friction_linear = 0.0f,
		.air_friction_angular = 0.5f
	});

	state.sh_sphere = tics_create_shape(state.world, (tics_shape_desc){
		.type = TICS_SHAPE_SPHERE,
		.data.sphere = {.center = {0, 0, 0}, .radius = SPHERE_RADIUS}
	});

	// Generate 2x2 regular square ground grid centered around origin
	float spacing = 2.0f * SPHERE_RADIUS;
	float offset = spacing / 2.0f;
	
	for (int z = 0; z < 2; z++) {
		for (int x = 0; x < 2; x++) {
			su_vec3 pos = {(x * spacing) - offset, 0.0f, (z * spacing) - offset};
			
			state.ground_positions[state.ground_count] = pos;

			tics_body_id body = tics_world_add_static_body(state.world, (tics_static_body_desc){
				.transform = {.position = {pos.x, pos.y, pos.z}, .rotation = {0, 0, 0, 1}},
				.shape = state.sh_sphere,
				.elasticity = ELASTICITY
			});

			state.ground_bodies[state.ground_count++] = body;
		}
	}

	sdtx_setup(&(sdtx_desc_t){
		.fonts[0] = sdtx_font_oric()
	});
}

static void frame(void) {
	float dt = (float)sapp_frame_duration();
	float clamped_dt = dt > MAX_FRAME_TIME ? MAX_FRAME_TIME : dt;

	if (sapp_mouse_locked()) {
		su_camera_navigate(&state.camera, &state.input, clamped_dt);
	}

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

	state.target_type = 0;
	state.target_index = -1;
	
	if (state.spawn_mode == SPAWN_MODE_DELETE) {
		su_vec3 ray_o = state.camera.position;
		su_vec3 ray_d = su_vec3_normalize(su_vec3_sub(state.camera.target, state.camera.position));
		float min_t = 1e9f;
		
		for (int i = 0; i < state.entity_count; i++) {
			tics_transform tf = tics_body_get_transform(state.world, state.entities[i].body);
			float t;
			if (ray_sphere_intersect(ray_o, ray_d, (su_vec3){tf.position.x, tf.position.y, tf.position.z}, SPHERE_RADIUS, &t)) {
				if (t < min_t) { min_t = t; state.target_type = 1; state.target_index = i; }
			}
		}
		
		for (int i = 0; i < state.ground_count; i++) {
			float t;
			if (ray_sphere_intersect(ray_o, ray_d, state.ground_positions[i], SPHERE_RADIUS, &t)) {
				if (t < min_t) { min_t = t; state.target_type = 2; state.target_index = i; }
			}
		}
	}

	sg_begin_pass(&(sg_pass){.action = state.pass_action, .swapchain = sglue_swapchain()});
	sg_apply_pipeline(state.pip);
	sg_apply_bindings(&state.bind_sphere);

	for (int i = 0; i < state.ground_count; i++) {
		su_vec3 pos = state.ground_positions[i];
		su_mat4 model = mat4_from_tics((tics_transform){{pos.x, pos.y, pos.z}, {0, 0, 0, 1}});
		vs_params_t vs = {.mvp = su_mat4_mul(vp, model), .model = model};
		sg_apply_uniforms(0, &SG_RANGE(vs));

		fs_params_t fs_static = {.color = {0.15f, 0.4f, 0.8f, 1.0f}};
		if (state.spawn_mode == SPAWN_MODE_DELETE && state.target_type == 2 && state.target_index == i) {
			fs_static.color[0] = 1.0f; fs_static.color[1] = 0.2f; fs_static.color[2] = 0.2f;
		}
		sg_apply_uniforms(1, &SG_RANGE(fs_static));
		sg_draw(0, sphere_indices_length, 1);
	}

	for (int i = 0; i < state.entity_count; i++) {
		tics_transform tf = tics_body_get_transform(state.world, state.entities[i].body);
		su_mat4 model = mat4_from_tics(tf);
		vs_params_t vs = {.mvp = su_mat4_mul(vp, model), .model = model};
		sg_apply_uniforms(0, &SG_RANGE(vs));

		fs_params_t fs_dyn = {.color = {0.95f, 0.5f, 0.1f, 1.0f}};
		if (state.spawn_mode == SPAWN_MODE_DELETE && state.target_type == 1 && state.target_index == i) {
			fs_dyn.color[0] = 1.0f; fs_dyn.color[1] = 0.2f; fs_dyn.color[2] = 0.2f;
		}
		sg_apply_uniforms(1, &SG_RANGE(fs_dyn));
		sg_draw(0, sphere_indices_length, 1);
	}

	// Render Grid
	sg_apply_pipeline(state.pip_grid);
	sg_apply_bindings(&state.bind_grid);
	vs_grid_params_t vs_grid = {.mvp = vp};
	sg_apply_uniforms(0, &SG_RANGE(vs_grid));
	sg_draw(0, 4, 1);

	// Render Preview Sphere
	sg_apply_pipeline(state.pip);
	sg_apply_bindings(&state.bind_sphere);

	// Render Preview Sphere
	if (state.spawn_mode == SPAWN_MODE_DYNAMIC || state.spawn_mode == SPAWN_MODE_STATIC) {
		su_vec3 preview_pos = {0};
		bool render_preview = false;
		fs_params_t fs_preview;

		if (state.spawn_mode == SPAWN_MODE_DYNAMIC) {
			preview_pos = get_dynamic_spawn_position();
			fs_preview.color[0] = 0.95f; fs_preview.color[1] = 0.5f; fs_preview.color[2] = 0.1f; fs_preview.color[3] = 0.4f;
			render_preview = true;
		} else if (state.spawn_mode == SPAWN_MODE_STATIC) {
			if (get_static_spawn_position(&preview_pos)) {
				fs_preview.color[0] = 0.15f; fs_preview.color[1] = 0.4f; fs_preview.color[2] = 0.8f; fs_preview.color[3] = 0.4f;
				render_preview = true;
			}
		}

		if (render_preview) {
			su_mat4 preview_model = mat4_from_tics((tics_transform){
				{preview_pos.x, preview_pos.y, preview_pos.z}, {0, 0, 0, 1}
			});
			vs_params_t vs_preview = {.mvp = su_mat4_mul(vp, preview_model), .model = preview_model};

			sg_apply_uniforms(0, &SG_RANGE(vs_preview));
			sg_apply_uniforms(1, &SG_RANGE(fs_preview));
			sg_draw(0, sphere_indices_length, 1);
		}
	}

	// --- Render UI Overlay ---
	if (state.ui_visible) {
		sdtx_canvas(w * 0.5f, h * 0.5f); // Scale up text x2
		sdtx_origin(1.0f, 1.0f); // Margin
		sdtx_home();
		
		sdtx_color3b(255, 255, 255);
		sdtx_printf("Max Tower Height: %.2f\n\n", state.max_tower_height);
		
		sdtx_puts("Controls:\n");
		
		if (state.spawn_mode == SPAWN_MODE_DYNAMIC) sdtx_color3b(242, 128, 25);
		else sdtx_color3b(150, 150, 150);
		sdtx_puts(" [1] Spawn Dynamic\n");
		
		if (state.spawn_mode == SPAWN_MODE_STATIC) sdtx_color3b(38, 102, 204);
		else sdtx_color3b(150, 150, 150);
		sdtx_puts(" [2] Spawn Static\n");
		
		if (state.spawn_mode == SPAWN_MODE_DELETE) sdtx_color3b(255, 51, 51);
		else sdtx_color3b(150, 150, 150);
		sdtx_puts(" [3] Delete Mode\n");
		
		sdtx_color3b(255, 255, 255);
		sdtx_puts(" [Space] Execute Action\n");
		sdtx_puts(" [Tab]   Toggle UI\n");
		
		sdtx_draw();
	}

	sg_end_pass();
	sg_commit();

	su_input_end_frame(&state.input);
}

static void event(const sapp_event* ev) {
	su_input_update(&state.input, ev);

	// Request pointer lock and keyboard focus on left click
	if (ev->type == SAPP_EVENTTYPE_MOUSE_DOWN && ev->mouse_button == SAPP_MOUSEBUTTON_LEFT) {
		sapp_lock_mouse(true);
	}

	if (ev->type == SAPP_EVENTTYPE_KEY_DOWN && !ev->key_repeat) {
		if (ev->key_code == SAPP_KEYCODE_TAB) {
			state.ui_visible = !state.ui_visible;
		} else if (ev->key_code == SAPP_KEYCODE_1) {
			state.spawn_mode = SPAWN_MODE_DYNAMIC;
		} else if (ev->key_code == SAPP_KEYCODE_2) {
			state.spawn_mode = SPAWN_MODE_STATIC;
		} else if (ev->key_code == SAPP_KEYCODE_3) {
			state.spawn_mode = SPAWN_MODE_DELETE;
		} else if (ev->key_code == SAPP_KEYCODE_SPACE) {
			if (state.spawn_mode == SPAWN_MODE_DYNAMIC) {
				spawn_dynamic_sphere();
			} else if (state.spawn_mode == SPAWN_MODE_STATIC) {
				spawn_static_sphere();
			} else if (state.spawn_mode == SPAWN_MODE_DELETE) {
				if (state.target_type == 1 && state.target_index >= 0) {
					tics_world_remove_body(state.world, state.entities[state.target_index].body);
					state.entities[state.target_index] = state.entities[state.entity_count - 1];
					state.entity_count--;
					state.target_index = -1;
				} else if (state.target_type == 2 && state.target_index >= 0) {
					tics_world_remove_body(state.world, state.ground_bodies[state.target_index]);
					state.ground_positions[state.target_index] = state.ground_positions[state.ground_count - 1];
					state.ground_bodies[state.target_index] = state.ground_bodies[state.ground_count - 1];
					state.ground_count--;
					state.target_index = -1;
				}
			}
		} else if (ev->key_code == SAPP_KEYCODE_ESCAPE) {
			#ifndef __EMSCRIPTEN__
			sapp_lock_mouse(false); // Restore cursor (Browser handle this by themself)
			#endif
		}
	}
}

static void cleanup(void) {
	sdtx_shutdown();
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
