#ifndef SOKOL_UTIL_H
#define SOKOL_UTIL_H

#include "sokol_app.h"
#include <stdbool.h>

// --- MATH TYPES ---
typedef struct {
	float x, y;
} su_vec2;
typedef struct {
	float x, y, z;
} su_vec3;
typedef struct {
	float m[16];
} su_mat4;

// --- MATH FUNCTIONS ---
su_vec3 su_vec3_add(su_vec3 a, su_vec3 b);
su_vec3 su_vec3_sub(su_vec3 a, su_vec3 b);
su_vec3 su_vec3_scale(su_vec3 v, float s);
float su_vec3_dot(su_vec3 a, su_vec3 b);
su_vec3 su_vec3_cross(su_vec3 a, su_vec3 b);
su_vec3 su_vec3_normalize(su_vec3 v);
float su_vec3_length(su_vec3 v);
su_vec3 su_vec3_rotate_axis_angle(su_vec3 v, su_vec3 axis, float angle);

su_mat4 su_mat4_mul(su_mat4 a, su_mat4 b);
su_mat4 su_mat4_look_at(su_vec3 eye, su_vec3 center, su_vec3 up);
su_mat4 su_mat4_persp(float fov_y, float aspect, float near_z, float far_z);

// --- INPUT TRACKING ---
typedef struct {
	bool keys_down[512];
	bool rmb_down;
	su_vec2 mouse_delta;
	float mouse_wheel;
} su_input;

void su_input_update(su_input* input, const sapp_event* ev);
void su_input_end_frame(su_input* input);

// --- CAMERA LOGIC ---
typedef struct {
	su_vec3 position;
	su_vec3 target;
	su_vec3 up;
} su_camera;

void su_camera_navigate(su_camera* cam, const su_input* input, float dt_seconds);

#endif // SOKOL_UTIL_H
