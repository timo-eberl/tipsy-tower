#include "sokol_util.h"
#include <math.h>

// --- MATH FUNCTIONS ---
// clang-format off
su_vec3 su_vec3_add(su_vec3 a, su_vec3 b) { return (su_vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
su_vec3 su_vec3_sub(su_vec3 a, su_vec3 b) { return (su_vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
su_vec3 su_vec3_scale(su_vec3 v, float s) { return (su_vec3){v.x * s, v.y * s, v.z * s}; }
float su_vec3_dot(su_vec3 a, su_vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float su_vec3_length(su_vec3 v) { return sqrtf(su_vec3_dot(v, v)); }
// clang-format on
su_vec3 su_vec3_cross(su_vec3 a, su_vec3 b) {
	return (su_vec3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
su_vec3 su_vec3_normalize(su_vec3 v) {
	float len = su_vec3_length(v);
	if (len > 0.00001f) return su_vec3_scale(v, 1.0f / len);
	return v;
}

su_vec3 su_vec3_rotate_axis_angle(su_vec3 v, su_vec3 axis, float angle) {
	float s = sinf(angle), c = cosf(angle);
	float dot = su_vec3_dot(axis, v);
	su_vec3 cross = su_vec3_cross(axis, v);
	return (su_vec3){v.x * c + cross.x * s + axis.x * dot * (1.0f - c),
					 v.y * c + cross.y * s + axis.y * dot * (1.0f - c),
					 v.z * c + cross.z * s + axis.z * dot * (1.0f - c)};
}

su_mat4 su_mat4_mul(su_mat4 a, su_mat4 b) {
	su_mat4 res = {0};
	for (int c = 0; c < 4; c++) {
		for (int r = 0; r < 4; r++) {
			res.m[c * 4 + r] = a.m[0 * 4 + r] * b.m[c * 4 + 0] + a.m[1 * 4 + r] * b.m[c * 4 + 1] +
							   a.m[2 * 4 + r] * b.m[c * 4 + 2] + a.m[3 * 4 + r] * b.m[c * 4 + 3];
		}
	}
	return res;
}

su_mat4 su_mat4_look_at(su_vec3 eye, su_vec3 center, su_vec3 up) {
	su_vec3 f = su_vec3_normalize(su_vec3_sub(center, eye));
	su_vec3 s = su_vec3_normalize(su_vec3_cross(f, up));
	su_vec3 u = su_vec3_cross(s, f);

	su_mat4 m = {0};
	m.m[0] = s.x;
	m.m[4] = s.y;
	m.m[8] = s.z;
	m.m[12] = -su_vec3_dot(s, eye);
	m.m[1] = u.x;
	m.m[5] = u.y;
	m.m[9] = u.z;
	m.m[13] = -su_vec3_dot(u, eye);
	m.m[2] = -f.x;
	m.m[6] = -f.y;
	m.m[10] = -f.z;
	m.m[14] = su_vec3_dot(f, eye);
	m.m[15] = 1.0f;
	return m;
}

su_mat4 su_mat4_persp(float fov_y, float aspect, float n, float f) {
	su_mat4 m = {0};
	float t = tanf(fov_y * 0.5f);
	m.m[0] = 1.0f / (t * aspect);
	m.m[5] = 1.0f / t;
	m.m[10] = -(f + n) / (f - n);
	m.m[11] = -1.0f;
	m.m[14] = -(2.0f * f * n) / (f - n);
	return m;
}

// --- INPUT TRACKING ---
void su_input_update(su_input* input, const sapp_event* ev) {
	if (ev->type == SAPP_EVENTTYPE_KEY_DOWN && ev->key_code < 512)
		input->keys_down[ev->key_code] = true;
	if (ev->type == SAPP_EVENTTYPE_KEY_UP && ev->key_code < 512)
		input->keys_down[ev->key_code] = false;

	if (ev->type == SAPP_EVENTTYPE_MOUSE_DOWN && ev->mouse_button == SAPP_MOUSEBUTTON_RIGHT) {
		input->rmb_down = true;
	}
	if (ev->type == SAPP_EVENTTYPE_MOUSE_UP && ev->mouse_button == SAPP_MOUSEBUTTON_RIGHT) {
		input->rmb_down = false;
	}

	if (ev->type == SAPP_EVENTTYPE_MOUSE_MOVE) {
		input->mouse_delta.x += ev->mouse_dx;
		input->mouse_delta.y += ev->mouse_dy;
	}
}

void su_input_end_frame(su_input* input) {
	input->mouse_delta.x = 0;
	input->mouse_delta.y = 0;
	input->mouse_wheel = 0;
}

// --- CAMERA LOGIC ---
void su_camera_navigate(su_camera* cam, const su_input* input, float dt_seconds) {
	// Look around
	{
		float dx = input->mouse_delta.x * 0.003f;
		float dy = input->mouse_delta.y * 0.003f;

		su_vec3 fwd = su_vec3_normalize(su_vec3_sub(cam->target, cam->position));
		// Yaw
		fwd = su_vec3_rotate_axis_angle(fwd, (su_vec3){0, 1, 0}, -dx);
		// Pitch
		su_vec3 right = su_vec3_cross(fwd, cam->up);
		float current_angle = acosf(fmaxf(-1.0f, fminf(1.0f, su_vec3_dot(fwd, cam->up))));
		float target_angle = current_angle + dy;
		float clamped_angle = fmaxf(0.001f, fminf(3.14159f - 0.001f, target_angle));
		float apply_pitch = -(clamped_angle - current_angle);
		fwd = su_vec3_rotate_axis_angle(fwd, right, apply_pitch);

		cam->target = su_vec3_add(cam->position, fwd);
	}

	// Movement
	static float speed = 10.0f;
	speed *= (1.0f + 0.1f * input->mouse_wheel);
	if (speed < 0.1f) speed = 0.1f;
	if (speed > 500.0f) speed = 500.0f;

	su_vec3 dir = {0};
	su_vec3 fwd = su_vec3_normalize(su_vec3_sub(cam->target, cam->position));
	su_vec3 right = su_vec3_normalize(su_vec3_cross(fwd, cam->up));

	if (input->keys_down[SAPP_KEYCODE_W]) dir = su_vec3_add(dir, fwd);
	if (input->keys_down[SAPP_KEYCODE_S]) dir = su_vec3_sub(dir, fwd);
	if (input->keys_down[SAPP_KEYCODE_D]) dir = su_vec3_add(dir, right);
	if (input->keys_down[SAPP_KEYCODE_A]) dir = su_vec3_sub(dir, right);
	if (input->keys_down[SAPP_KEYCODE_E]) dir.y += 1.0f;
	if (input->keys_down[SAPP_KEYCODE_Q]) dir.y -= 1.0f;

	if (su_vec3_length(dir) > 0) {
		dir = su_vec3_scale(su_vec3_normalize(dir), speed * dt_seconds);
		cam->position = su_vec3_add(cam->position, dir);
		cam->target = su_vec3_add(cam->target, dir);
	}
}
