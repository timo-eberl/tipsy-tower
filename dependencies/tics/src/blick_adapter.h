#ifndef BLICK_ADAPTER_H
#define BLICK_ADAPTER_H

// Internal adapter for the 'blick' visualization library.

#ifdef TICS_ENABLE_BLICK

#include "tics.h"

#include <blick.h>

#include <stdio.h>

// --- Type Conversion Helpers (Private) ---
#define _BLICK_V3(v) ((blick_vec3){(v).x, (v).y, (v).z})
#define _BLICK_Q(q) ((blick_quat){(q).x, (q).y, (q).z, (q).w})

// --- Lifecycle ---

#define BLICK_INIT() blick_init("blick_viewer")
#define BLICK_SHUTDOWN() blick_shutdown()
#define BLICK_REFRESH() blick_refresh()
#define BLICK_CLEAR(mask) blick_clear(mask)
#define BLICK_TRIM_LAYER(l, max_count) blick_trim_layer(l, max_count)

#define BLICK_UPLOAD_MESH_INDEXED(id, verts, indices, i_count)                                     \
	blick_upload_mesh_indexed(id, (const blick_vec3*)(verts), indices, i_count)

// --- Primitives ---
// l = layer id (0-15)

#define BLICK_LINE(l, s, e, c) blick_record_line(l, _BLICK_V3(s), _BLICK_V3(e), c)
#define BLICK_ARROW(l, s, e, c) blick_record_arrow(l, _BLICK_V3(s), _BLICK_V3(e), c)
#define BLICK_POINT(l, p, r, c) blick_record_point(l, _BLICK_V3(p), r, c)
#define BLICK_AABB(l, min, max, c) blick_record_aabb(l, _BLICK_V3(min), _BLICK_V3(max), c)
#define BLICK_TRIANGLE(l, a, b, c_pos, col)                                                        \
	blick_record_triangle(l, _BLICK_V3(a), _BLICK_V3(b), _BLICK_V3(c_pos), col)
#define BLICK_TRANSFORM(l, t, s)                                                                   \
	blick_record_transform(l, _BLICK_V3((t).position), _BLICK_Q((t).rotation), s)
#define BLICK_SPHERE(l, t, r, c, wire)                                                             \
	blick_record_sphere(l, _BLICK_V3((t).position), _BLICK_Q((t).rotation), r, c, wire)
#define BLICK_TEXT(l, p, txt, c) blick_record_text(l, _BLICK_V3(p), txt, c)
#define BLICK_MESH(l, id, t, color, wire)                                                          \
	blick_record_mesh(l, id, _BLICK_V3((t).position), _BLICK_Q((t).rotation), color, wire)

// --- Text Helpers ---

// Prints an integer: BLICK_TEXT_INT(layer, pos, 42, color)
#define BLICK_TEXT_INT(l, p, val, c)                                                               \
	do {                                                                                           \
		char _b[32];                                                                               \
		snprintf(_b, sizeof(_b), "%d", (int)(val));                                                \
		blick_record_text(l, _BLICK_V3(p), _b, c);                                                 \
	} while (0)

// Prints a float with specific precision: BLICK_TEXT_FLOAT(layer, pos, 3.14159, 2, color) -> "3.14"
#define BLICK_TEXT_FLOAT(l, p, val, prec, c)                                                       \
	do {                                                                                           \
		char _b[32];                                                                               \
		snprintf(_b, sizeof(_b), "%.*f", prec, (float)(val));                                      \
		blick_record_text(l, _BLICK_V3(p), _b, c);                                                 \
	} while (0)

// Prints a vec3: BLICK_TEXT_VEC3(layer, pos, my_vec, 2, color) -> "(1.00 2.50 -0.50)"
#define BLICK_TEXT_VEC3(l, p, v, prec, c)                                                          \
	do {                                                                                           \
		char _b[64];                                                                               \
		snprintf(_b, sizeof(_b), "(%.*f %.*f %.*f)", (int)(prec), (float)(v).x, (int)(prec),       \
				 (float)(v).y, (int)(prec), (float)(v).z);                                         \
		blick_record_text(l, _BLICK_V3(p), _b, c);                                                 \
	} while (0)

#define BLICK_DRAW_SHAPE(layer, shape, xform, color, wire)                                         \
	do {                                                                                           \
		if ((shape).type == TICS_SHAPE_CONVEX) {                                                   \
			BLICK_MESH((layer), (shape).id, (xform), (color), (wire));                             \
		}                                                                                          \
		else if ((shape).type == TICS_SHAPE_SPHERE) {                                              \
			tics_transform _bx = (xform);                                                          \
			_bx.position = vec3_add(_bx.position,                                                  \
									quat_rotate_vec3((shape).data.sphere.center, _bx.rotation));   \
			BLICK_SPHERE((layer), _bx, (shape).data.sphere.radius, (color), (wire));               \
		}                                                                                          \
	} while (0)

#else

// --- No-Op Implementations ---

#define BLICK_INIT() ((void)0)
#define BLICK_SHUTDOWN() ((void)0)
#define BLICK_REFRESH() ((void)0)
#define BLICK_CLEAR(...) ((void)0)
#define BLICK_TRIM_LAYER(...) ((void)0)

#define BLICK_UPLOAD_MESH_INDEXED(...) ((void)0)

#define BLICK_LINE(...) ((void)0)
#define BLICK_ARROW(...) ((void)0)
#define BLICK_POINT(...) ((void)0)
#define BLICK_AABB(...) ((void)0)
#define BLICK_TRIANGLE(...) ((void)0)
#define BLICK_TRANSFORM(...) ((void)0)
#define BLICK_SPHERE(...) ((void)0)
#define BLICK_TEXT(...) ((void)0)
#define BLICK_MESH(...) ((void)0)

#define BLICK_TEXT_INT(...) ((void)0)
#define BLICK_TEXT_FLOAT(...) ((void)0)
#define BLICK_TEXT_VEC3(...) ((void)0)

#define BLICK_DRAW_SHAPE(...) ((void)0)

#endif // TICS_ENABLE_BLICK

#endif // BLICK_ADAPTER_H
