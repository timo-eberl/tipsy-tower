#include "blick_adapter.h"
#include "tics_internal.h"
#include "tics_math.h"

#include <stb_ds.h>

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

// Ensures the pair is ordered deterministically before collision detection.
// Order:
// 1. By Type (RIGID_BODY, STATIC_BODY)
// 2. By Index (Low < High)
static inline void canonicalize_pair(broad_phase_pair* p) {
	int swap = 0;

	// Rule 1: Sort by Type (Rigid=1 Static=0)
	if (p->a.type < p->b.type) { swap = 1; }
	// Rule 2: If types are identical, Sort by Index
	else if (p->a.type == p->b.type && p->a.index > p->b.index) { swap = 1; }

	if (swap) {
		body_ref temp = p->a;
		p->a = p->b;
		p->b = temp;
	}
}

static int compare_collisions(const void* lhs, const void* rhs);

collision* narrow_phase(const broad_phase_pair* pairs, size_t pair_count,
						const rigid_body_data* r_bodies, const static_body_data* s_bodies) {

	// Early exit if broadphase found nothing
	if (pair_count == 0) return NULL;

	// Setup per-thread storage
	int max_threads = omp_get_max_threads();
	collision** thread_buffers = calloc(max_threads, sizeof(collision*));

	// Uncomment this to disable multi-threading
	// omp_set_num_threads(1);

	// Parallel loop over potential collision pairs
	// We use 'static' because chunks have roughly the same workload
#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < pair_count; ++i) {
		int tid = omp_get_thread_num();

		broad_phase_pair p = pairs[i];
		canonicalize_pair(&p);

		// A
		const shape_data* shape_a;
		tics_transform trans_a;
		if (p.a.type == RIGID_BODY) {
			shape_a = &r_bodies[p.a.index].shape;
			trans_a = r_bodies[p.a.index].transform;
		}
		else {
			shape_a = &s_bodies[p.a.index].shape;
			trans_a = s_bodies[p.a.index].transform;
		}
		// B
		const shape_data* shape_b;
		tics_transform trans_b;
		if (p.b.type == RIGID_BODY) {
			shape_b = &r_bodies[p.b.index].shape;
			trans_b = r_bodies[p.b.index].transform;
		}
		else {
			shape_b = &s_bodies[p.b.index].shape;
			trans_b = s_bodies[p.b.index].transform;
		}

		// --- Actual Geometric Test ---
		collision_result res = collision_test(shape_a, trans_a, shape_b, trans_b);

		if (res.has_collision) {
			collision col;
			col.body_a_ref = p.a;
			col.body_b_ref = p.b;
			col.result = res;

			// Write to thread-local buffer
			arrput(thread_buffers[tid], col);
		}
	} // implicit barrier

	// Merge Phase
	// Calculate total collisions to allocate exact memory once
	size_t total_count = 0;
	for (int i = 0; i < max_threads; ++i) {
		total_count += arrlen(thread_buffers[i]);
	}

	collision* collisions = NULL;
	arrsetlen(collisions, total_count);

	size_t offset = 0;
	for (int i = 0; i < max_threads; ++i) {
		size_t count = arrlen(thread_buffers[i]);
		if (count > 0) {
			memcpy(collisions + offset, thread_buffers[i], count * sizeof(collision));
			offset += count;
		}
		arrfree(thread_buffers[i]);
	}
	free(thread_buffers);

	// Sort the collisions so the order is exactly the same -> Determinism with Multi-Threading
	size_t col_size = arrlen(collisions);
	if (col_size > 0) { qsort(collisions, col_size, sizeof(collision), compare_collisions); }

	return collisions;
}

static int compare_collisions(const void* lhs, const void* rhs) {
	const collision* a = (const collision*)lhs;
	const collision* b = (const collision*)rhs;

	// 1. Compare Body A Type
	if (a->body_a_ref.type != b->body_a_ref.type)
		return (int)a->body_a_ref.type - (int)b->body_a_ref.type;

	// 2. Compare Body A Index
	if (a->body_a_ref.index != b->body_a_ref.index)
		return (a->body_a_ref.index < b->body_a_ref.index) ? -1 : 1;

	// 3. Compare Body B Type
	if (a->body_b_ref.type != b->body_b_ref.type)
		return (int)a->body_b_ref.type - (int)b->body_b_ref.type;

	// 4. Compare Body B Index
	if (a->body_b_ref.index != b->body_b_ref.index)
		return (a->body_b_ref.index < b->body_b_ref.index) ? -1 : 1;

	return 0;
}

// support point on minkowski difference
// point on shape b is calculated as a - m
typedef struct {
	tics_vec3 m; // minkowski difference
	tics_vec3 a; // corresponding point on shape a
} mink_support;

typedef struct {
	tics_vec3 normal;
	float distance;
} face_plane;

typedef struct {
	uint32_t a;
	uint32_t b;
} edge;

// A support function takes a direction d and returns a point on the boundary of a shape "furthest"
// in direction d
static tics_vec3 support_point_mesh(const shape_data* c, tics_transform t, tics_vec3 d) {
	assert(c->type == TICS_SHAPE_CONVEX);

	tics_vec3 local_d = quat_rotate_vec3(d, quat_inverse(t.rotation));

	const tics_vec3* vertices = c->data.convex.vertices;
	size_t count = c->data.convex.count;

	// find the support point in local space
	float support_point_dot = -FLT_MAX;
	tics_vec3 support = {0, 0, 0};

	for (size_t i = 0; i < count; ++i) {
		float p_dot_d = vec3_dot(vertices[i], local_d);
		if (p_dot_d > support_point_dot) {
			support_point_dot = p_dot_d;
			support = vertices[i];
		}
	}

	// this fails if the center position of a mesh is not inside the mesh
	assert(support_point_dot >= 0.0);

	support = quat_rotate_vec3(support, t.rotation);
	support = vec3_add(support, t.position);

	return support;
}

static mink_support support_point_on_minkowski_diff_mesh_mesh(const shape_data* ca,
															  tics_transform ta,
															  const shape_data* cb,
															  tics_transform tb, tics_vec3 d) {
	assert(ca->type == TICS_SHAPE_CONVEX);
	assert(cb->type == TICS_SHAPE_CONVEX);

	mink_support point;
	point.a = support_point_mesh(ca, ta, d);
	// point.b = support_point_mesh(cb, tb, - d);
	// point.m = point.a - point.b;
	tics_vec3 b_supp = support_point_mesh(cb, tb, vec3_negate(d));
	point.m = vec3_sub(point.a, b_supp);

	return point;
}

static void add_if_unique_edge(edge** edges, uint32_t edge_a, uint32_t edge_b) {
	size_t count = arrlen(*edges);
	ptrdiff_t found_idx = -1;

	for (size_t i = 0; i < count; ++i) {
		if ((*edges)[i].a == edge_b && (*edges)[i].b == edge_a) {
			found_idx = i;
			break;
		}
	}

	// edge was already present -> remove it
	if (found_idx != -1) { arrdel(*edges, found_idx); }
	else {
		edge new_edge = {edge_a, edge_b};
		arrput(*edges, new_edge);
	}
}

// Collision detection is based on the GJK Algorithm.
//   First implementation is based on https://youtu.be/ajv46BSqcK4
//   However that implementation didn't cover all cases for 3D that caused cycling.
//   Added improvements based on https://gist.github.com/vurtun/29727217c269a2fbf4c0ed9a1d11cb40
// If a collision is found the EPA algorithm is used to get detailed collision information.
static collision_result collision_test_convex_convex(const shape_data* as, tics_transform ta,
													 const shape_data* bs, tics_transform tb) {
	assert(as->type == TICS_SHAPE_CONVEX);
	assert(bs->type == TICS_SHAPE_CONVEX);

	collision_result result = {0};

	// first direction is arbitrary - we use the direction from the origin of one shape to the other
	tics_vec3 d = vec3_sub(tb.position, ta.position);
	// if this is zero, we use a fallback
	if (vec3_length_sq(d) == 0) d = (tics_vec3){1, 0, 0};

	// Can be a point, line segment, triangle or polyhedron
	mink_support simplex[4] = {0};
	int count = 0; // simplex size

	while (true) {
		// find the next support point
		simplex[count] = support_point_on_minkowski_diff_mesh_mesh(as, ta, bs, tb, d);
		// if the new support point does not "pass" the origin, the shapes do not intersect
		if (vec3_dot(simplex[count].m, d) < 0.001f) { return result; }
		count++;

		// For case 1,2,3 the simplex stays the same size or is expanded. For case 4, the simplex
		// may be reduced (special case for GJK in 3D).
		switch (count) {
		// point
		case 1: {
			// the second direction is towards the origin
			d = vec3_negate(simplex[0].m);
		} break;
		// line segment
		case 2: {
			// We could check if the origin lies exactly on the line AB.
			// However, we just "forward" that problem to be handled correctly in case 3.

			// A = most recently added vertex, O = Origin
			tics_vec3 AB = vec3_sub(simplex[0].m, simplex[1].m);
			tics_vec3 AO = vec3_negate(simplex[1].m);
			// triple product: vector perpendicular to AB pointing toward the origin
			d = vec3_cross(vec3_cross(AB, AO), AB);
		} break;
		// triangle
		case 3: {
			// We could check if the origin lies on the line AB or AC.
			// We "forward" that problem

			// A = most recently added vertex, O = Origin
			tics_vec3 AB = vec3_sub(simplex[1].m, simplex[2].m);
			tics_vec3 AC = vec3_sub(simplex[0].m, simplex[2].m);
			tics_vec3 AO = vec3_negate(simplex[2].m);

			// triple products to define regions R_AB and R_AC
			tics_vec3 ABC_normal = vec3_cross(AB, AC);
			tics_vec3 AB_normal = vec3_cross(vec3_negate(ABC_normal), AB); // (AC x AB) x AB
			tics_vec3 AC_normal = vec3_cross(ABC_normal, AC);			   // (AB x AC) x AC

			if (vec3_dot(AB_normal, AO) > 0) {
				// We are in region AB
				// Remove current C, shift the array so newest vertex is at simplex[2]
				simplex[0] = simplex[1];
				simplex[1] = simplex[2];
				d = AB_normal;
				count = 2;
			}
			else if (vec3_dot(AC_normal, AO) > 0) {
				// We are in region AC
				// Remove current B, shift the array so newest vertex is at simplex[2]
				simplex[1] = simplex[2];
				d = AC_normal;
				count = 2;
			}
			else {
				// We are in region ABC. Check if the origin is above or below ABC and move on.

				// We could check if the origin lies exactly on the plane ABC.
				// We "forward" that problem

				float abc_dot_ao = vec3_dot(ABC_normal, AO);
				if (vec3_dot(ABC_normal, AO) > 0) {
					// above ABC
					d = ABC_normal;
				}
				else {
					// below ABC
					// swap current C and B (change winding order), so we are above ABC again
					mink_support B = simplex[1];
					simplex[1] = simplex[0];
					simplex[0] = B;
					d = vec3_negate(ABC_normal);
				}
			}
		} break;
		// tetrahedron
		case 4: {
			// We could check if the origin lies exactly on any relevant line/plane.
			// We "forward" that problem

			mink_support A = simplex[3];
			mink_support B = simplex[2];
			mink_support C = simplex[1];
			mink_support D = simplex[0];

			tics_vec3 AB = vec3_sub(B.m, A.m);
			tics_vec3 AC = vec3_sub(C.m, A.m);
			tics_vec3 AD = vec3_sub(D.m, A.m);
			tics_vec3 AO = vec3_negate(A.m);

			tics_vec3 ABC_normal = vec3_cross(AB, AC);
			tics_vec3 ACD_normal = vec3_cross(AC, AD);
			tics_vec3 ADB_normal = vec3_cross(AD, AB);

			// Barycentric Coordinate Determinants
			// We calculate cross products of the vertices (relative to origin).
			// These represent the normals of the sub-triangles formed by the Origin and the Edge.
			// By dotting these with the Face Normal, we find the signed volume/area contribution.
			// if (Normal . (A x B)) < 0, the origin is on the 'outside' of edge AB relative to the
			// face.

			tics_vec3 OAB_normal = vec3_cross(A.m, B.m);
			tics_vec3 OAC_normal = vec3_cross(A.m, C.m);
			tics_vec3 OAD_normal = vec3_cross(A.m, D.m);
			tics_vec3 OBC_normal = vec3_cross(B.m, C.m);
			tics_vec3 OCD_normal = vec3_cross(C.m, D.m);
			tics_vec3 ODB_normal = vec3_cross(D.m, B.m);

			// Barycentrics for Face ABC
			// w_abc: vertex C's contribution. Negative -> outside ABC across edge AB.
			// v_abc: vertex B's contribution. Negative -> outside ABC across edge AC.
			// u_abc: vertex A's contribution. Negative -> outside ABC across edge BC.
			float w_abc = vec3_dot(ABC_normal, OAB_normal);
			// Note: OAC_normal is A->C, winding flip for AC edge check inside ABC
			float v_abc = vec3_dot(ABC_normal, vec3_negate(OAC_normal));
			float u_abc = vec3_dot(ABC_normal, OBC_normal);

			// Barycentrics for Face ACD
			// v_acd: D's contribution
			// u_acd: C's contribution
			// w_acd: A's contribution
			float v_acd = vec3_dot(ACD_normal, OAC_normal);
			float u_acd = vec3_dot(ACD_normal, vec3_negate(OAD_normal));
			float w_acd = vec3_dot(ACD_normal, OCD_normal);

			// Barycentrics for Face ADB
			// u_adb: B's contribution
			// w_adb: D's contribution
			// v_adb: A's contribution
			float u_adb = vec3_dot(ADB_normal, OAD_normal);
			float w_adb = vec3_dot(ADB_normal, vec3_negate(OAB_normal));
			float v_adb = vec3_dot(ADB_normal, ODB_normal);

			// Note: We only check faces connected to A (ABC, ACD, ADB) because the origin can not
			// be in the region of BDC. If it were, the "pass origin" check would have failed
			// earlier. We need to check all edge regions.

			// --- CHECK EDGES CONNECTED TO A ---
			// An edge is the closest feature if the origin is outside of the edges two triangles
			// across said edge.

			// Check Edge AB
			// w_abc <= 0: outside ABC across edge AB
			// w_adb <= 0: outside ADB across edge AB
			if (w_abc <= 0 && w_adb <= 0) {
				// Reduce to Line AB
				simplex[0] = B;
				simplex[1] = A;
				count = 2;
				d = vec3_cross(vec3_cross(AB, AO), AB);
				break;
			}
			// Check Edge AC
			if (v_abc <= 0 && v_acd <= 0) {
				// Reduce to Line AC
				simplex[0] = C;
				simplex[1] = A;
				count = 2;
				d = vec3_cross(vec3_cross(AC, AO), AC);
				break;
			}
			// Check Edge AD
			if (u_acd <= 0 && u_adb <= 0) {
				// Reduce to Line AD
				simplex[0] = D;
				simplex[1] = A;
				count = 2;
				d = vec3_cross(vec3_cross(AD, AO), AD);
				break;
			}

			// --- CHECK FACES ---
			// To be in a Face Voronoi region, the origin must be in front of the face plane (dot >
			// 0) AND inside the triangular prism defined by the edges (all barycentrics > 0).
			// Checking only the dot product does not suffice, because multiple dot products can be
			// positive.

			if (vec3_dot(ABC_normal, AO) > 0 && w_abc > 0 && v_abc > 0 && u_abc > 0) {
				// We are strictly in region ABC
				simplex[0] = C;
				simplex[1] = B;
				simplex[2] = A;
				count = 3;
				d = ABC_normal;
				break;
			}
			if (vec3_dot(ACD_normal, AO) > 0 && v_acd > 0 && u_acd > 0 && w_acd > 0) {
				// We are strictly in region ACD
				simplex[0] = D;
				simplex[1] = C;
				simplex[2] = A;
				count = 3;
				d = ACD_normal;
				break;
			}
			if (vec3_dot(ADB_normal, AO) > 0 && u_adb > 0 && w_adb > 0 && v_adb > 0) {
				// We are strictly in region ADB
				simplex[0] = B;
				simplex[1] = D;
				simplex[2] = A;
				count = 3;
				d = ADB_normal;
				break;
			}

			// --- CHECK BASE EDGES ---
			// For these edges, we must verify the origin is outside BOTH adjacent faces (The upper
			// face and BDC). Those edges are not connected to A, so we theoretically only need to
			// check that condition for the triangle that is connected to A, because the other
			// triangle is BDC and we can not be in its voronoi region. However we need to keep in
			// mind that it's possible that the origin lies exactly on BDC. In that case we will not
			// reduce, but instead continue to EPA. We could have handled the "exaclty on" cases
			// beforehand separately, but since we didn't we need to deal with it here.

			// Calculate Base Face Normal (BDC): BC x BD
			tics_vec3 BCD_normal = vec3_cross(vec3_sub(C.m, B.m), vec3_sub(D.m, B.m));
			// Barycentrics for Face BDC (Base)
			// d_bcd: vertex D's contribution. Negative -> outside BDC across edge BC.
			// b_bcd: vertex B's contribution. Negative -> outside BDC across edge CD.
			// c_bcd: vertex C's contribution. Negative -> outside BDC across edge DB.
			float d_bcd = vec3_dot(BCD_normal, OBC_normal);
			float b_bcd = vec3_dot(BCD_normal, OCD_normal);
			float c_bcd = vec3_dot(BCD_normal, ODB_normal);

			// Check Edge BC
			// u_abc <= 0: outside ABC across edge BC
			// d_bcd <= 0: outside BDC across edge BC
			if (u_abc <= 0 && d_bcd <= 0) {
				// Reduce to Line BC
				simplex[0] = C;
				simplex[1] = B;
				count = 2;
				tics_vec3 BC = vec3_sub(C.m, B.m);
				d = vec3_cross(vec3_cross(BC, vec3_negate(B.m)), BC);
				break;
			}
			// Check Edge CD
			// w_acd <= 0: outside ACD across edge CD
			// b_bcd <= 0: outside BDC across edge CD
			if (w_acd <= 0 && b_bcd <= 0) {
				// Reduce to Line CD
				simplex[0] = D;
				simplex[1] = C;
				count = 2;
				tics_vec3 CD = vec3_sub(D.m, C.m);
				d = vec3_cross(vec3_cross(CD, vec3_negate(C.m)), CD);
				break;
			}
			// Check Edge DB
			// v_adb <= 0: outside ADB across edge DB
			// c_bcd <= 0: outside BDC across edge DB
			if (v_adb <= 0 && c_bcd <= 0) {
				// Reduce to Line DB
				simplex[0] = B;
				simplex[1] = D;
				count = 2;
				tics_vec3 DB = vec3_sub(B.m, D.m);
				d = vec3_cross(vec3_cross(DB, vec3_negate(D.m)), DB);
				break;
			}

			// Collision detected!
			result.has_collision = true;

			// EPA (Expanding Polytope Algorithm): GJK Extension for collision information
			// We want to find the normal of the collision.
			//
			// normal of collision = b - a
			// if a and b are each the furthest points of the one shape into the other. This
			// normal is the normal of the face of the minkowski difference that is closest to
			// the origin.
			//
			// Problem: The simplex we found in which the origin lies is a subspace of the
			// minkowski difference. It does not necessarily contain the required face.
			//
			// Solution: We are adding vertices to the simplex (making it a polytope) until we
			// find the shortest normal from a face that is on the original mesh

			// we find the face that is closest
			// then we try to expand the polytope in the direction of the faces normal
			// if we were able to expand - repeat
			// if not, we found the closest face

			// initialize the polytope with the data from the simplex
			mink_support* polytope_positions = NULL;
			arrput(polytope_positions, simplex[0]);
			arrput(polytope_positions, simplex[1]);
			arrput(polytope_positions, simplex[2]);
			arrput(polytope_positions, simplex[3]);

			// order the vertices of the triangles so that the normals are always pointing
			// outwards
			uint32_t* polytope_indices = NULL;
			// clang-format off
			// 0,1,2 ; 0,3,1 ; 0,2,3 ; 1,3,2
			arrput(polytope_indices,0); arrput(polytope_indices,1); arrput(polytope_indices,2);
			arrput(polytope_indices,0); arrput(polytope_indices,3); arrput(polytope_indices,1);
			arrput(polytope_indices,0); arrput(polytope_indices,2); arrput(polytope_indices,3);
			arrput(polytope_indices,1); arrput(polytope_indices,3); arrput(polytope_indices,2);
			// clang-format on

			// calculate face normals (normal, distance)
			// and find the face closest to the origin
			face_plane* polytope_normals = NULL;
			float closest_distance = FLT_MAX;
			size_t closest_index = 0;

			size_t num_faces = arrlen(polytope_indices) / 3;
			for (size_t k = 0; k < num_faces; k++) {
				tics_vec3 a = polytope_positions[polytope_indices[k * 3 + 0]].m;
				tics_vec3 b = polytope_positions[polytope_indices[k * 3 + 1]].m;
				tics_vec3 c = polytope_positions[polytope_indices[k * 3 + 2]].m;

				tics_vec3 normal = vec3_normalize(vec3_cross(vec3_sub(b, a), vec3_sub(c, a)));
				float distance = vec3_dot(normal, a); // works with any vertex of the plane

				face_plane plane = {normal, distance};
				arrput(polytope_normals, plane);

				if (distance < closest_distance) {
					closest_distance = distance;
					closest_index = k;
				}
			}

			while (true) {
				// search for a new support point in the direction of the normal of the closest
				// face
				d = polytope_normals[closest_index].normal;
				mink_support new_supp_p =
					support_point_on_minkowski_diff_mesh_mesh(as, ta, bs, tb, d);
				float support_distance = vec3_dot(d, new_supp_p.m);

				// check if the support point lies on the same plane as the closest face
				// if it does, the polytype cannot be further expanded
				if (fabsf(support_distance - closest_distance) <= 0.001f) {
					break; // cannot be expanded - found the closest face!
				}

				// expand the polytope by adding the support point
				// to make sure the polytope stays convex, we remove all faces that point
				// towards the support point and create new faces afterwards

				edge* unique_edges = NULL;

				size_t k = 0;
				while (k < arrlen(polytope_normals)) {
					// check if the support point is in front of the triangle
					tics_vec3 face_normal = polytope_normals[k].normal;
					tics_vec3 p_on_face = polytope_positions[polytope_indices[k * 3]].m;
					float dotp = vec3_dot(face_normal, vec3_sub(new_supp_p.m, p_on_face));

					if (dotp > 0) {
						// if it is, collect all unique edges
						add_if_unique_edge(&unique_edges, polytope_indices[k * 3 + 0],
										   polytope_indices[k * 3 + 1]);
						add_if_unique_edge(&unique_edges, polytope_indices[k * 3 + 1],
										   polytope_indices[k * 3 + 2]);
						add_if_unique_edge(&unique_edges, polytope_indices[k * 3 + 2],
										   polytope_indices[k * 3 + 0]);

						// Remove this face (indices and normal)
						arrdel(polytope_indices, k * 3); // arrdel removes 1 item
						arrdel(polytope_indices, k * 3);
						arrdel(polytope_indices, k * 3);
						arrdel(polytope_normals, k);
					}
					else {
						// Only move to the next index if we didn't remove the current one
						k++;
					}
				}

				// create new vertex and faces
				uint32_t new_vertex_index = (uint32_t)arrlen(polytope_positions);
				arrput(polytope_positions, new_supp_p);

				for (size_t k = 0; k < arrlen(unique_edges); k++) {
					uint32_t edge_index_a = unique_edges[k].a;
					uint32_t edge_index_b = unique_edges[k].b;

					arrput(polytope_indices, edge_index_a);
					arrput(polytope_indices, edge_index_b);
					arrput(polytope_indices, new_vertex_index);

					tics_vec3 a = polytope_positions[edge_index_a].m;
					tics_vec3 b = polytope_positions[edge_index_b].m;
					tics_vec3 c = polytope_positions[new_vertex_index].m;

					tics_vec3 normal = vec3_normalize(vec3_cross(vec3_sub(b, a), vec3_sub(c, a)));
					float distance = vec3_dot(normal, a);

					if (distance < 0) {
						normal = vec3_negate(normal);
						distance = -distance;
					}

					face_plane plane = {normal, distance};
					arrput(polytope_normals, plane);
				}

				arrfree(unique_edges);

				// (re)iterate over all faces and find the closest
				closest_distance = FLT_MAX;
				closest_index = 0;
				for (size_t k = 0; k < arrlen(polytope_normals); k++) {
					float distance = polytope_normals[k].distance;
					if (distance < closest_distance) {
						closest_distance = distance;
						closest_index = k;
					}
				}
			}

			tics_vec3 result_normal = polytope_normals[closest_index].normal;
			result.normal = vec3_negate(result_normal);
			result.depth = closest_distance;

			// Algorithm that finds the collision points on the original shapes a and b

			// get vertices of face the farthest from the origin in minkowski space
			mink_support a = polytope_positions[polytope_indices[closest_index * 3 + 0]];
			mink_support b = polytope_positions[polytope_indices[closest_index * 3 + 1]];
			mink_support c = polytope_positions[polytope_indices[closest_index * 3 + 2]];

			// first, we find the closest point to the origin of the face in minkowski space
			tics_vec3 p = vec3_mul_f(result_normal, polytope_normals[closest_index].distance);

			// now, we calculate the barycentric coordinates of this point on the minkowski
			// space face the areas of the triangles BCP,CAP,ABP are proportional to the
			// barycentric coordinates u,v,w

			float bcp_area = vec3_length(vec3_cross(vec3_sub(p, b.m), vec3_sub(p, c.m)));
			float cap_area = vec3_length(vec3_cross(vec3_sub(p, c.m), vec3_sub(p, a.m)));
			float abp_area = vec3_length(vec3_cross(vec3_sub(p, a.m), vec3_sub(p, b.m)));

			float face_area = cap_area + abp_area + bcp_area;
			// barycentric coordinates
			float u = bcp_area / face_area; // a
			float v = cap_area / face_area; // b
			float w = abp_area / face_area; // c

			// reconstruct p to see if the barycentric coordinates are correct:
			// p_reconstructed = ( a.m * u + b.m * v + c.m * w );
			// reconstructed_distance = length(p_reconstructed - p);
			// TODO Fix: sometimes the values are off, because p does not lie on the plane abc
			// TODO Check if this is still the case

			// now, we reconstruct the collision points of the original shapes a and b
			tics_vec3 term_a = vec3_mul_f(a.a, u);
			tics_vec3 term_b = vec3_mul_f(b.a, v);
			tics_vec3 term_c = vec3_mul_f(c.a, w);

			result.point_a = vec3_add(vec3_add(term_a, term_b), term_c);
			result.point_b = vec3_add(result.point_a, vec3_mul_f(result.normal, result.depth));

			// cleanup
			arrfree(polytope_positions);
			arrfree(polytope_indices);
			arrfree(polytope_normals);

			return result;
		} break;
		} // switch
	}
}

static collision_result collision_test_sphere_sphere(const shape_data* as, tics_transform ta,
													 const shape_data* bs, tics_transform tb) {
	assert(as->type == TICS_SHAPE_SPHERE);
	assert(bs->type == TICS_SHAPE_SPHERE);

	collision_result result = {0};

	// Calculate global center positions
	// Apply rotation to the local center offset, then add to body position
	tics_vec3 center_a =
		vec3_add(ta.position, quat_rotate_vec3(as->data.sphere.center, ta.rotation));
	tics_vec3 center_b =
		vec3_add(tb.position, quat_rotate_vec3(bs->data.sphere.center, tb.rotation));

	float radius_a = as->data.sphere.radius;
	float radius_b = bs->data.sphere.radius;
	float radius_sum = radius_a + radius_b;

	// Vector from A to B
	tics_vec3 delta = vec3_sub(center_b, center_a);
	float dist_sq = vec3_length_sq(delta);

	// Early exit: no collision if distance squared > radius sum squared
	if (dist_sq > radius_sum * radius_sum) { return result; }

	result.has_collision = true;
	float distance = sqrtf(dist_sq);

	// Handle degenerate case: spheres are at the exact same position
	if (distance < 0.0001f) {
		result.depth = radius_sum;
		result.normal = (tics_vec3){0.0f, 1.0f, 0.0f}; // Arbitrary normal (Up)
	}
	else {
		result.depth = radius_sum - distance;
		result.normal = vec3_mul_f(delta, -1.0f / distance); // Normalized center_b -> center_a
	}

	// Point on Surface A closest to B
	result.point_a = vec3_add(center_a, vec3_mul_f(result.normal, -radius_a));
	// Point on Surface B closest to A (center_b - normal * radius_b)
	result.point_b = vec3_add(center_b, vec3_mul_f(result.normal, radius_b));

	return result;
}

// function type for a collision test function
typedef collision_result (*collision_test_func)(const shape_data*, tics_transform,
												const shape_data*, tics_transform);

collision_result collision_test(const shape_data* as, tics_transform at, const shape_data* bs,
								tics_transform bt) {
	// a collision table as described by valve in this pdf on page 33
	// https://media.steampowered.com/apps/valve/2015/DirkGregorius_Contacts.pdf

#define XXX NULL // Unreachable/Invalid

	static const collision_test_func function_table[2][2] = {
		// clang-format off
		// Sphere                                Convex
		{ collision_test_sphere_sphere /*TODO*/, NULL /*TODO*/                }, // Sphere
		{ XXX,                                   collision_test_convex_convex }, // Convex
		// clang-format on
	};

	// make sure the colliders are in the correct order
	// example: (convex, sphere) gets swapped to (sphere, convex)
	bool swap = as->type > bs->type;

	const shape_data* sorted_a = swap ? bs : as;
	const shape_data* sorted_b = swap ? as : bs;
	tics_transform sorted_at = swap ? bt : at;
	tics_transform sorted_bt = swap ? at : bt;

	// pick the function that matches the collider types from the table
	collision_test_func func = function_table[sorted_a->type][sorted_b->type];
	// check if collision test function is defined for the given colliders
	assert(func != NULL);

	collision_result result = func(sorted_a, sorted_at, sorted_b, sorted_bt);

	// BLICK_CLEAR(0b1000000);
	// // draw shapes in different colors
	// BLICK_DRAW_SHAPE(6, *sorted_a, sorted_at, 0xFFFF5555, true);
	// BLICK_DRAW_SHAPE(6, *sorted_a, sorted_at, 0x22FF5555, false);
	// BLICK_DRAW_SHAPE(6, *sorted_b, sorted_bt, 0xFF55FF55, true);
	// BLICK_DRAW_SHAPE(6, *sorted_b, sorted_bt, 0x2255FF55, false);
	// // draw a red arrow between collision points (might be very small)
	// BLICK_ARROW(6, result.point_a, result.point_b, 0xFF0000FF);
	// // draw two yellow lines with a fixed length extending in both directions of the arrow
	// tics_vec3 target_a = vec3_add(result.point_a, vec3_mul_f(result.normal, -0.1f));
	// tics_vec3 target_b = vec3_add(result.point_b, vec3_mul_f(result.normal, 0.1f));
	// BLICK_LINE(6, result.point_a, target_a, 0xFF00FFFF);
	// BLICK_LINE(6, result.point_b, target_b, 0xFF00FFFF);
	// BLICK_REFRESH();

	// if we swapped the input colliders, we need to invert the collision data
	if (swap) {
		result.normal = vec3_negate(result.normal);
		tics_vec3 temp = result.point_a;
		result.point_a = result.point_b;
		result.point_b = temp;
	}

	return result;
};
