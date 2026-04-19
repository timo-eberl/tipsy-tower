#ifndef TICS_MATH_H
#define TICS_MATH_H

#include <math.h>

#include "tics.h"

// Technically not required since there are only typedefs and static inline functions here, but
// maybe that'll change
#ifdef __cplusplus
extern "C" {
#endif

// header-only math for best performance

// --- Vector Implementation ---

static inline tics_vec3 vec3_add(tics_vec3 a, tics_vec3 b) {
	return (tics_vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline tics_vec3 vec3_sub(tics_vec3 a, tics_vec3 b) {
	return (tics_vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline tics_vec3 vec3_mul_f(tics_vec3 v, float s) {
	return (tics_vec3){v.x * s, v.y * s, v.z * s};
}

static inline float vec3_dot(tics_vec3 a, tics_vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline tics_vec3 vec3_cross(tics_vec3 a, tics_vec3 b) {
	return (tics_vec3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

static inline float vec3_length_sq(tics_vec3 v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

static inline float vec3_length(tics_vec3 v) {
	return sqrt(vec3_length_sq(v));
}

static inline tics_vec3 vec3_normalize(tics_vec3 v) {
	float len = vec3_length(v);
	if (len > 0.00001f) {
		float inv = 1.0f / len;
		return (tics_vec3){v.x * inv, v.y * inv, v.z * inv};
	}
	return (tics_vec3){0, 0, 0};
}

static inline tics_vec3 vec3_negate(tics_vec3 v) {
	return (tics_vec3){-v.x, -v.y, -v.z};
}

// --- Quaternion Implementation ---

// Grassman product (standard quaternion multiplication)
static inline tics_quat quat_mul(tics_quat a, tics_quat b) {
	return (tics_quat){a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
					   a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
					   a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
					   a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

static inline tics_quat quat_normalize(tics_quat q) {
	float len_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
	if (len_sq > 0.00001f) {
		float inv = 1.0f / sqrt(len_sq);
		return (tics_quat){q.x * inv, q.y * inv, q.z * inv, q.w * inv};
	}
	return (tics_quat){0, 0, 0, 1};
}

// axis must be normalized
static inline tics_quat quat_from_axis_angle(tics_vec3 axis, float angle) {
	float half_angle = angle * 0.5f;
	float s = sinf(half_angle);
	return (tics_quat){axis.x * s, axis.y * s, axis.z * s, cosf(half_angle)};
}

// Rotate vector v by quaternion q: v' = q * v * q_inv
// Optimized version: v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
static inline tics_vec3 quat_rotate_vec3(tics_vec3 v, tics_quat q) {
	tics_vec3 q_xyz = {q.x, q.y, q.z};
	tics_vec3 t = vec3_cross(q_xyz, v);
	t = vec3_add(t, t); // 2 * cross(q.xyz, v)

	// v + q.w * t + cross(q.xyz, t)
	tics_vec3 term1 = vec3_mul_f(t, q.w);
	tics_vec3 term2 = vec3_cross(q_xyz, t);
	return vec3_add(v, vec3_add(term1, term2));
}

// Simple linear interpolation followed by normalize
static inline tics_quat quat_nlerp(tics_quat a, tics_quat b, float t) {
	tics_quat res;
	float one_minus_t = 1.0f - t;
	res.x = a.x * one_minus_t + b.x * t;
	res.y = a.y * one_minus_t + b.y * t;
	res.z = a.z * one_minus_t + b.z * t;
	res.w = a.w * one_minus_t + b.w * t;
	return quat_normalize(res);
}

// Spherical linear interpolation
static inline tics_quat quat_slerp(tics_quat a, tics_quat b, float t) {
	float cos_theta = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

	// Ensure shortest path
	if (cos_theta < 0.0f) {
		b.x = -b.x;
		b.y = -b.y;
		b.z = -b.z;
		b.w = -b.w;
		cos_theta = -cos_theta;
	}

	// Fallback to nlerp if angle is too small (prevents division by zero)
	if (cos_theta > 0.9995f) { return quat_nlerp(a, b, t); }

	float theta = acosf(cos_theta);
	float sin_theta = sinf(theta);

	float w_a = sinf((1.0f - t) * theta) / sin_theta;
	float w_b = sinf(t * theta) / sin_theta;

	tics_quat res;
	res.x = a.x * w_a + b.x * w_b;
	res.y = a.y * w_a + b.y * w_b;
	res.z = a.z * w_a + b.z * w_b;
	res.w = a.w * w_a + b.w * w_b;

	return quat_normalize(res);
}

// Scales rotation magnitude by interpolating (slerp) from identity to q
static inline tics_quat quat_scale(tics_quat q, float scale) {
	return quat_slerp((tics_quat){0, 0, 0, 1}, q, scale);
}

// inverse rotation (conjugate for unit quaternions)
static inline tics_quat quat_inverse(tics_quat q) {
	return (tics_quat){-q.x, -q.y, -q.z, q.w};
}

// Transform a World Point into Local Space
static inline tics_vec3 world_to_local(tics_transform t, tics_vec3 world_point) {
	tics_vec3 rel = vec3_sub(world_point, t.position);
	tics_quat inv_rot = quat_inverse(t.rotation); // assumes unit quaternion
	return quat_rotate_vec3(rel, inv_rot);
}

#ifdef __cplusplus
}
#endif

#endif // TICS_MATH_H
