/*
The MIT License (MIT)
Copyright (c) 2012-Present, Syoyo Fujita and many contributors.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

//
// version 2.0.0 : Add new object oriented API. 1.x API is still provided.
//                 * Support line primitive.
//                 * Support points primitive.
//                 * Support multiple search path for .mtl(v1 API).
//                 * Support vertex weight `vw`(as an tinyobj extension)
//                 * Support escaped whitespece in mtllib
//                 * Add robust triangulation using Mapbox earcut(TINYOBJLOADER_USE_MAPBOX_EARCUT).
// version 1.4.0 : Modifed ParseTextureNameAndOption API
// version 1.3.1 : Make ParseTextureNameAndOption API public
// version 1.3.0 : Separate warning and error message(breaking API of LoadObj)
// version 1.2.3 : Added color space extension('-colorspace') to tex opts.
// version 1.2.2 : Parse multiple group names.
// version 1.2.1 : Added initial support for line('l') primitive(PR #178)
// version 1.2.0 : Hardened implementation(#175)
// version 1.1.1 : Support smoothing groups(#162)
// version 1.1.0 : Support parsing vertex color(#144)
// version 1.0.8 : Fix parsing `g` tag just after `usemtl`(#138)
// version 1.0.7 : Support multiple tex options(#126)
// version 1.0.6 : Add TINYOBJLOADER_USE_DOUBLE option(#124)
// version 1.0.5 : Ignore `Tr` when `d` exists in MTL(#43)
// version 1.0.4 : Support multiple filenames for 'mtllib'(#112)
// version 1.0.3 : Support parsing texture options(#85)
// version 1.0.2 : Improve parsing speed by about a factor of 2 for large
// files(#105)
// version 1.0.1 : Fixes a shape is lost if obj ends with a 'usemtl'(#104)
// version 1.0.0 : Change data structure. Change license from BSD to MIT.
//

//
// Use this in *one* .cc
//   #define TINYOBJLOADER_IMPLEMENTATION
//   #include "tiny_obj_loader.h"
//

#ifndef TINY_OBJ_LOADER_H_
#define TINY_OBJ_LOADER_H_

#include "material.h"
#include <limits>

// https://en.wikipedia.org/wiki/Wavefront_.obj_file says ...
//
//  -blendu on | off                       # set horizontal texture blending
//  (default on)
//  -blendv on | off                       # set vertical texture blending
//  (default on)
//  -boost real_value                      # boost mip-map sharpness
//  -mm base_value gain_value              # modify texture map values (default
//  0 1)
//                                         #     base_value = brightness,
//                                         gain_value = contrast
//  -o u [v [w]]                           # Origin offset             (default
//  0 0 0)
//  -s u [v [w]]                           # Scale                     (default
//  1 1 1)
//  -t u [v [w]]                           # Turbulence                (default
//  0 0 0)
//  -texres resolution                     # texture resolution to create
//  -clamp on | off                        # only render texels in the clamped
//  0-1 range (default off)
//                                         #   When unclamped, textures are
//                                         repeated across a surface,
//                                         #   when clamped, only texels which
//                                         fall within the 0-1
//                                         #   range are rendered.
//  -bm mult_value                         # bump multiplier (for bump maps
//  only)
//
//  -imfchan r | g | b | m | l | z         # specifies which channel of the file
//  is used to
//                                         # create a scalar or bump texture.
//                                         r:red, g:green,
//                                         # b:blue, m:matte, l:luminance,
//                                         z:z-depth..
//                                         # (the default for bump is 'l' and
//                                         for decal is 'm')
//  bump -imfchan r bumpmap.tga            # says to use the red channel of
//  bumpmap.tga as the bumpmap
//
// For reflection maps...
//
//   -type sphere                           # specifies a sphere for a "refl"
//   reflection map
//   -type cube_top    | cube_bottom |      # when using a cube map, the texture
//   file for each
//         cube_front  | cube_back   |      # side of the cube is specified
//         separately
//         cube_left   | cube_right
//
// TinyObjLoader extension.
//
//   -colorspace SPACE                      # Color space of the texture. e.g.
//   'sRGB` or 'linear'
//

struct index_t {
  int vertex_index;
  int normal_index;
  int texcoord_index;
};

struct mesh_t {
  std::vector<index_t> indices;
  std::vector<unsigned char> num_face_vertices;          
  // The number of vertices per
  // face. 3 = triangle, 4 = quad,
  // ... Up to 255 vertices per face.
  std::vector<int> material_ids;  // per-face material ID
};

struct shape_t {
  std::string name;
  mesh_t mesh;
};

// Vertex attributes
struct attrib_t {
  std::vector<real_t> vertices;  // 'v'(xyz)

  // For backward compatibility, we store vertex weight in separate array.
  std::vector<real_t> vertex_weights;  // 'v'(w)
  std::vector<real_t> normals;         // 'vn'
  std::vector<real_t> texcoords;       // 'vt'(uv)

  // For backward compatibility, we store texture coordinate 'w' in separate
  // array.
  std::vector<real_t> texcoord_ws;  // 'vt'(w)
  std::vector<real_t> colors;       // extension: vertex colors

  //
  // TinyObj extension.
  //

  attrib_t() {}

  //
  // For pybind11
  //
  const std::vector<real_t> &GetVertices() const { return vertices; }

  const std::vector<real_t> &GetVertexWeights() const { return vertex_weights; }
};

struct PrimGroup {
  std::vector<face_t> faceGroup;

  void clear() {
    faceGroup.clear();
  }

  bool IsEmpty() const {
    return faceGroup.empty();
  }

  // TODO(syoyo): bspline, surface, ...
};

static bool exportGroupsToShape(shape_t *shape, const PrimGroup &prim_group,
                                const int material_id, const std::string &name,
                                bool triangulate, const std::vector<real_t> &v,
                                std::string *warn) {
  if (prim_group.IsEmpty()) {
    return false;
  }

  shape->name = name;

  // polygon
  // Flatten vertices and indices
  for (size_t i = 0; i < prim_group.faceGroup.size(); i++) {
    const face_t &face = prim_group.faceGroup[i];

    size_t npolys = face.vertex_indices.size();

    if (npolys < 3) {
      // Face must have 3+ vertices.
      if (warn) {
	(*warn) += "Degenerated face found\n.";
      }
      continue;
    }

    if (triangulate) {
      if (npolys == 4) {
	vertex_index_t i0 = face.vertex_indices[0];
	vertex_index_t i1 = face.vertex_indices[1];
	vertex_index_t i2 = face.vertex_indices[2];
	vertex_index_t i3 = face.vertex_indices[3];

	size_t vi0 = size_t(i0.v_idx);
	size_t vi1 = size_t(i1.v_idx);
	size_t vi2 = size_t(i2.v_idx);
	size_t vi3 = size_t(i3.v_idx);

	if (((3 * vi0 + 2) >= v.size()) || ((3 * vi1 + 2) >= v.size()) ||
	    ((3 * vi2 + 2) >= v.size()) || ((3 * vi3 + 2) >= v.size())) {
	  // Invalid triangle.
	  // FIXME(syoyo): Is it ok to simply skip this invalid triangle?
	  if (warn) {
	    (*warn) += "Face with invalid vertex index found.\n";
	  }
	  continue;
	}

	real_t v0x = v[vi0 * 3 + 0];
	real_t v0y = v[vi0 * 3 + 1];
	real_t v0z = v[vi0 * 3 + 2];
	real_t v1x = v[vi1 * 3 + 0];
	real_t v1y = v[vi1 * 3 + 1];
	real_t v1z = v[vi1 * 3 + 2];
	real_t v2x = v[vi2 * 3 + 0];
	real_t v2y = v[vi2 * 3 + 1];
	real_t v2z = v[vi2 * 3 + 2];
	real_t v3x = v[vi3 * 3 + 0];
	real_t v3y = v[vi3 * 3 + 1];
	real_t v3z = v[vi3 * 3 + 2];
	
	// There are two candidates to split the quad into two triangles.
	//
	// Choose the shortest edge.
	// TODO: Is it better to determine the edge to split by calculating
	// the area of each triangle?
	//
	// +---+
	// |\  |
	// | \ |
	// |  \|
	// +---+
	//
	// +---+
	// |  /|
	// | / |
	// |/  |
	// +---+
	
	real_t e02x = v2x - v0x;
	real_t e02y = v2y - v0y;
	real_t e02z = v2z - v0z;
	real_t e13x = v3x - v1x;
	real_t e13y = v3y - v1y;
	real_t e13z = v3z - v1z;
	
	real_t sqr02 = e02x * e02x + e02y * e02y + e02z * e02z;
	real_t sqr13 = e13x * e13x + e13y * e13y + e13z * e13z;
	
	index_t idx0, idx1, idx2, idx3;
	
	idx0.vertex_index = i0.v_idx;
	idx0.normal_index = i0.vn_idx;
	idx0.texcoord_index = i0.vt_idx;
	idx1.vertex_index = i1.v_idx;
	idx1.normal_index = i1.vn_idx;
	idx1.texcoord_index = i1.vt_idx;
	idx2.vertex_index = i2.v_idx;
	idx2.normal_index = i2.vn_idx;
	idx2.texcoord_index = i2.vt_idx;
	idx3.vertex_index = i3.v_idx;
	idx3.normal_index = i3.vn_idx;
	idx3.texcoord_index = i3.vt_idx;
	
	if (sqr02 < sqr13) {
	  // [0, 1, 2], [0, 2, 3]
	  shape->mesh.indices.push_back(idx0);
	  shape->mesh.indices.push_back(idx1);
	  shape->mesh.indices.push_back(idx2);
	  
	  shape->mesh.indices.push_back(idx0);
	  shape->mesh.indices.push_back(idx2);
	  shape->mesh.indices.push_back(idx3);
	} else {
	  // [0, 1, 3], [1, 2, 3]
	  shape->mesh.indices.push_back(idx0);
	  shape->mesh.indices.push_back(idx1);
	  shape->mesh.indices.push_back(idx3);
	  
	  shape->mesh.indices.push_back(idx1);
	  shape->mesh.indices.push_back(idx2);
	  shape->mesh.indices.push_back(idx3);
	}
	
	// Two triangle faces
	shape->mesh.num_face_vertices.push_back(3);
	shape->mesh.num_face_vertices.push_back(3);

	shape->mesh.material_ids.push_back(material_id);
	shape->mesh.material_ids.push_back(material_id);

      } else {
	vertex_index_t i0 = face.vertex_indices[0];
	vertex_index_t i1(-1);
	vertex_index_t i2 = face.vertex_indices[1];

	// find the two axes to work in
	size_t axes[2] = {1, 2};
	for (size_t k = 0; k < npolys; ++k) {
	  i0 = face.vertex_indices[(k + 0) % npolys];
	  i1 = face.vertex_indices[(k + 1) % npolys];
	  i2 = face.vertex_indices[(k + 2) % npolys];
	  size_t vi0 = size_t(i0.v_idx);
	  size_t vi1 = size_t(i1.v_idx);
	  size_t vi2 = size_t(i2.v_idx);

	  if (((3 * vi0 + 2) >= v.size()) || ((3 * vi1 + 2) >= v.size()) ||
	      ((3 * vi2 + 2) >= v.size())) {
	    // Invalid triangle.
	    // FIXME(syoyo): Is it ok to simply skip this invalid triangle?
	    continue;
	  }
	  real_t v0x = v[vi0 * 3 + 0];
	  real_t v0y = v[vi0 * 3 + 1];
	  real_t v0z = v[vi0 * 3 + 2];
	  real_t v1x = v[vi1 * 3 + 0];
	  real_t v1y = v[vi1 * 3 + 1];
	  real_t v1z = v[vi1 * 3 + 2];
	  real_t v2x = v[vi2 * 3 + 0];
	  real_t v2y = v[vi2 * 3 + 1];
	  real_t v2z = v[vi2 * 3 + 2];
	  real_t e0x = v1x - v0x;
	  real_t e0y = v1y - v0y;
	  real_t e0z = v1z - v0z;
	  real_t e1x = v2x - v1x;
	  real_t e1y = v2y - v1y;
	  real_t e1z = v2z - v1z;
	  real_t cx = std::fabs(e0y * e1z - e0z * e1y);
	  real_t cy = std::fabs(e0z * e1x - e0x * e1z);
	  real_t cz = std::fabs(e0x * e1y - e0y * e1x);
	  const real_t epsilon = std::numeric_limits<real_t>::epsilon();
	  
	  if (cx > epsilon || cy > epsilon || cz > epsilon) {
	    // found a corner
	    if (!(cx > cy && cx > cz)) {
		axes[0] = 0;
		if (cz > cx && cz > cy) {
		  axes[1] = 1;
		}
	    }
	    break;
	  }
	}

#ifdef TINYOBJLOADER_USE_MAPBOX_EARCUT
	using Point = std::array<real_t, 2>;

	// first polyline define the main polygon.
	// following polylines define holes(not used in tinyobj).
	std::vector<std::vector<Point> > polygon;
	
	std::vector<Point> polyline;
	
	// Fill polygon data(facevarying vertices).
	for (size_t k = 0; k < npolys; k++) {
	  i0 = face.vertex_indices[k];
	  size_t vi0 = size_t(i0.v_idx);
	  
	  assert(((3 * vi0 + 2) < v.size()));
	  
	  real_t v0x = v[vi0 * 3 + axes[0]];
	  real_t v0y = v[vi0 * 3 + axes[1]];
	  
	  polyline.push_back({v0x, v0y});
	}
	
	polygon.push_back(polyline);
	std::vector<uint32_t> indices = mapbox::earcut<uint32_t>(polygon);
	// => result = 3 * faces, clockwise
	
	assert(indices.size() % 3 == 0);
	
	// Reconstruct vertex_index_t
	for (size_t k = 0; k < indices.size() / 3; k++) {
	  {
	    index_t idx0, idx1, idx2;
	    idx0.vertex_index = face.vertex_indices[indices[3 * k + 0]].v_idx;
	    idx0.normal_index =
	      face.vertex_indices[indices[3 * k + 0]].vn_idx;
	    idx0.texcoord_index =
	      face.vertex_indices[indices[3 * k + 0]].vt_idx;
	    idx1.vertex_index = face.vertex_indices[indices[3 * k + 1]].v_idx;
	    idx1.normal_index =
	      face.vertex_indices[indices[3 * k + 1]].vn_idx;
	    idx1.texcoord_index =
	      face.vertex_indices[indices[3 * k + 1]].vt_idx;
	    idx2.vertex_index = face.vertex_indices[indices[3 * k + 2]].v_idx;
	    idx2.normal_index =
	      face.vertex_indices[indices[3 * k + 2]].vn_idx;
	    idx2.texcoord_index =
	      face.vertex_indices[indices[3 * k + 2]].vt_idx;
	    
	    shape->mesh.indices.push_back(idx0);
	    shape->mesh.indices.push_back(idx1);
	    shape->mesh.indices.push_back(idx2);
	    
	    shape->mesh.num_face_vertices.push_back(3);
	    shape->mesh.material_ids.push_back(material_id);
	  }
	}

#else  // Built-in ear clipping triangulation

	
	face_t remainingFace = face;  // copy
	size_t guess_vert = 0;
	vertex_index_t ind[3];
	real_t vx[3];
	real_t vy[3];
	
	// How many iterations can we do without decreasing the remaining
	// vertices.
	size_t remainingIterations = face.vertex_indices.size();
	size_t previousRemainingVertices =
	  remainingFace.vertex_indices.size();
	
	while (remainingFace.vertex_indices.size() > 3 &&
	       remainingIterations > 0) {

	  npolys = remainingFace.vertex_indices.size();
	  if (guess_vert >= npolys) {
	    guess_vert -= npolys;
	  }
	  
	  if (previousRemainingVertices != npolys) {
	    // The number of remaining vertices decreased. Reset counters.
	    previousRemainingVertices = npolys;
	    remainingIterations = npolys;
	  } else {
	    // We didn't consume a vertex on previous iteration, reduce the
	    // available iterations.
	    remainingIterations--;
	  }

	  for (size_t k = 0; k < 3; k++) {
	    ind[k] = remainingFace.vertex_indices[(guess_vert + k) % npolys];
	    size_t vi = size_t(ind[k].v_idx);
	    if (((vi * 3 + axes[0]) >= v.size()) ||
		((vi * 3 + axes[1]) >= v.size())) {
	      // ???
	      vx[k] = static_cast<real_t>(0.0);
	      vy[k] = static_cast<real_t>(0.0);
	    } else {
	      vx[k] = v[vi * 3 + axes[0]];
	      vy[k] = v[vi * 3 + axes[1]];
	    }
	  }

	  //
	  // area is calculated per face
	  //
	  real_t e0x = vx[1] - vx[0];
	  real_t e0y = vy[1] - vy[0];
	  real_t e1x = vx[2] - vx[1];
	  real_t e1y = vy[2] - vy[1];
	  real_t cross = e0x * e1y - e0y * e1x;
	  
	  real_t area = (vx[0] * vy[1] - vy[0] * vx[1]) * static_cast<real_t>(0.5);
	  // if an internal angle
	  if (cross * area < static_cast<real_t>(0.0)) {
	    guess_vert += 1;
	    continue;
	  }
	  
	  // check all other verts in case they are inside this triangle
	  bool overlap = false;
	  for (size_t otherVert = 3; otherVert < npolys; ++otherVert) {
	    size_t idx = (guess_vert + otherVert) % npolys;
	    
	    if (idx >= remainingFace.vertex_indices.size()) {
	      continue;
	    }
	    
	    size_t ovi = size_t(remainingFace.vertex_indices[idx].v_idx);
	    
	    if (((ovi * 3 + axes[0]) >= v.size()) ||
		((ovi * 3 + axes[1]) >= v.size())) {
	      continue;
	    }
	    real_t tx = v[ovi * 3 + axes[0]];
	    real_t ty = v[ovi * 3 + axes[1]];
	    if (pnpoly(3, vx, vy, tx, ty)) {
	      overlap = true;
	      break;
	    }
	  }

	  if (overlap) {
	    guess_vert += 1;
	    continue;
	  }

	  // this triangle is an ear
	  {
	    index_t idx0, idx1, idx2;
	    idx0.vertex_index = ind[0].v_idx;
	    idx0.normal_index = ind[0].vn_idx;
	    idx0.texcoord_index = ind[0].vt_idx;
	    idx1.vertex_index = ind[1].v_idx;
	    idx1.normal_index = ind[1].vn_idx;
	    idx1.texcoord_index = ind[1].vt_idx;
	    idx2.vertex_index = ind[2].v_idx;
	    idx2.normal_index = ind[2].vn_idx;
	    idx2.texcoord_index = ind[2].vt_idx;
	    
	    shape->mesh.indices.push_back(idx0);
	    shape->mesh.indices.push_back(idx1);
	    shape->mesh.indices.push_back(idx2);
	    
	    shape->mesh.num_face_vertices.push_back(3);
	    shape->mesh.material_ids.push_back(material_id);
	  }

	  // remove v1 from the list
	  size_t removed_vert_index = (guess_vert + 1) % npolys;
	  while (removed_vert_index + 1 < npolys) {
	    remainingFace.vertex_indices[removed_vert_index] =
	      remainingFace.vertex_indices[removed_vert_index + 1];
	    removed_vert_index += 1;
	  }
	  remainingFace.vertex_indices.pop_back();
	}

	if (remainingFace.vertex_indices.size() == 3) {
	  i0 = remainingFace.vertex_indices[0];
	  i1 = remainingFace.vertex_indices[1];
	  i2 = remainingFace.vertex_indices[2];
	  {
	    index_t idx0, idx1, idx2;
	    idx0.vertex_index = i0.v_idx;
	    idx0.normal_index = i0.vn_idx;
	    idx0.texcoord_index = i0.vt_idx;
	    idx1.vertex_index = i1.v_idx;
	    idx1.normal_index = i1.vn_idx;
	    idx1.texcoord_index = i1.vt_idx;
	    idx2.vertex_index = i2.v_idx;
	    idx2.normal_index = i2.vn_idx;
	    idx2.texcoord_index = i2.vt_idx;
	    
	    shape->mesh.indices.push_back(idx0);
	    shape->mesh.indices.push_back(idx1);
	    shape->mesh.indices.push_back(idx2);
	    
	    shape->mesh.num_face_vertices.push_back(3);
	    shape->mesh.material_ids.push_back(material_id);
	  }
	}
#endif
      }  // npolys
    } else {
      for (size_t k = 0; k < npolys; k++) {
	index_t idx;
	idx.vertex_index = face.vertex_indices[k].v_idx;
	idx.normal_index = face.vertex_indices[k].vn_idx;
	idx.texcoord_index = face.vertex_indices[k].vt_idx;
	shape->mesh.indices.push_back(idx);
      }

      shape->mesh.num_face_vertices.push_back(
					      static_cast<unsigned char>(npolys));
      shape->mesh.material_ids.push_back(material_id);  // per face
    }
  }

  return true;
}

bool LoadObj(attrib_t *attrib, std::vector<shape_t> *shapes,
	     std::vector<material_t> *materials, std::string *warn,
	     std::string *err, std::istream *inStream,
	     MaterialReader *readMatFn, bool triangulate,
	     bool default_vcols_fallback) {
  std::stringstream errss;

  std::vector<real_t> v;
  std::vector<real_t> vn;
  std::vector<real_t> vt;
  std::vector<real_t> vc;
  PrimGroup prim_group;
  std::string name;

  // material
  std::map<std::string, int> material_map;
  int material = -1;


 int greatest_v_idx = -1;
 int greatest_vn_idx = -1;
 int greatest_vt_idx = -1;

 shape_t shape;

 bool found_all_colors = true;
 
 size_t line_num = 0;
 std::string linebuf;
 
 while (inStream->peek() != -1) {
   safeGetline(*inStream, linebuf);

   line_num++;

   // Trim newline '\r\n' or '\n'
   if (linebuf.size() > 0) {
     if (linebuf[linebuf.size() - 1] == '\n')
       linebuf.erase(linebuf.size() - 1);
   }
   if (linebuf.size() > 0) {
     if (linebuf[linebuf.size() - 1] == '\r')
       linebuf.erase(linebuf.size() - 1);
   }

   // Skip if empty line.
   if (linebuf.empty()) {
     continue;
   }

   // Skip leading space.
   const char *token = linebuf.c_str();
   token += strspn(token, " \t");

   assert(token);
   if (token[0] == '\0') continue;  // empty line

   if (token[0] == '#') continue;  // comment line

   // vertex
   if (token[0] == 'v' && IS_SPACE((token[1]))) {
     token += 2;
     real_t x, y, z;
     real_t r, g, b;

     found_all_colors &= parseVertexWithColor(&x, &y, &z, &r, &g, &b, &token);

     v.push_back(x);
     v.push_back(y);
     v.push_back(z);

     if (found_all_colors || default_vcols_fallback) {
       vc.push_back(r);
       vc.push_back(g);
       vc.push_back(b);
     }
     continue;
   }
 
   // normal
   if (token[0] == 'v' && token[1] == 'n' && IS_SPACE((token[2]))) {
     token += 3;
     real_t x, y, z;
     parseReal3(&x, &y, &z, &token);
     vn.push_back(x);
     vn.push_back(y);
     vn.push_back(z);
     continue;
   }

   // texcoord
   if (token[0] == 'v' && token[1] == 't' && IS_SPACE((token[2]))) {
     token += 3;
     real_t x, y;
     parseReal2(&x, &y, &token);
     vt.push_back(x);
     vt.push_back(y);
     continue;
   }
   
   // face
   if (token[0] == 'f' && IS_SPACE((token[1]))) {
     token += 2;
     token += strspn(token, " \t");

     face_t face;

     face.vertex_indices.reserve(3);

     while (!IS_NEW_LINE(token[0])) {
       vertex_index_t vi;
       if (!parseTriple(&token, static_cast<int>(v.size() / 3),
			static_cast<int>(vn.size() / 3),
			static_cast<int>(vt.size() / 2), &vi)) {
	 if (err) {
	   std::stringstream ss;
      ss << "Failed parse `f' line(e.g. zero value for face index. line "
	 << line_num << ".)\n";
      (*err) += ss.str();
	 }
	 return false;
       }

       greatest_v_idx = greatest_v_idx > vi.v_idx ? greatest_v_idx : vi.v_idx;
       greatest_vn_idx =
	 greatest_vn_idx > vi.vn_idx ? greatest_vn_idx : vi.vn_idx;
       greatest_vt_idx =
	 greatest_vt_idx > vi.vt_idx ? greatest_vt_idx : vi.vt_idx;

       face.vertex_indices.push_back(vi);
       size_t n = strspn(token, " \t\r");
       token += n;
     }

     prim_group.faceGroup.push_back(face);
     
     continue;
   }

   // use mtl
   if ((0 == strncmp(token, "usemtl", 6))) {
     token += 6;
     std::string namebuf = parseString(&token);

     int newMaterialId = -1;
     std::map<std::string, int>::const_iterator it =
       material_map.find(namebuf);
     if (it != material_map.end()) {
       newMaterialId = it->second;
     } else {
       // { error!! material not found }
       if (warn) {
	 (*warn) += "material [ '" + namebuf + "' ] not found in .mtl\n";
       }
     }

     if (newMaterialId != material) {
       // Create per-face material. Thus we don't add `shape` to `shapes` at
       // this time.
       // just clear `faceGroup` after `exportGroupsToShape()` call.
       exportGroupsToShape(&shape, prim_group, material, name,
			   triangulate, v, warn);
       prim_group.faceGroup.clear();
       material = newMaterialId;
     }

     continue;
   }
 
   // load mtl
   if ((0 == strncmp(token, "mtllib", 6)) && IS_SPACE((token[6]))) {
     if (readMatFn) {
       token += 7;

       std::vector<std::string> filenames;
       SplitString(std::string(token), ' ', '\\', filenames);
       
       if (filenames.empty()) {
	 if (warn) {
	   std::stringstream ss;
      ss << "Looks like empty filename for mtllib. Use default "
	    "material (line "
	 << line_num << ".)\n";

      (*warn) += ss.str();
	 }
       } else {
	 bool found = false;
	 for (size_t s = 0; s < filenames.size(); s++) {
	   std::string warn_mtl;
	   std::string err_mtl;
	   bool ok = (*readMatFn)(filenames[s].c_str(), materials,
				  &material_map, &warn_mtl, &err_mtl);
	   if (warn && (!warn_mtl.empty())) {
	     (*warn) += warn_mtl;
	   }

	   if (err && (!err_mtl.empty())) {
	     (*err) += err_mtl;
	   }

	   if (ok) {
	     found = true;
	     break;
	   }
	 }

	 if (!found) {
	   if (warn) {
	     (*warn) +=
	           "Failed to load material file(s). Use default "
	       "material.\n";
	   }
	 }
       }
     }
     continue;
   }
 
   // group name
   if (token[0] == 'g' && IS_SPACE((token[1]))) {
     // flush previous face group.
     bool ret = exportGroupsToShape(&shape, prim_group, material, name,
				    triangulate, v, warn);
     (void)ret;  // return value not used.

     if (shape.mesh.indices.size() > 0) {
       shapes->push_back(shape);
     }

     shape = shape_t();

     // material = -1;
     prim_group.clear();

     std::vector<std::string> names;

     while (!IS_NEW_LINE(token[0])) {
       std::string str = parseString(&token);
       names.push_back(str);
       token += strspn(token, " \t\r");  // skip tag
     }

     // names[0] must be 'g'

     if (names.size() < 2) {
       // 'g' with empty names
       if (warn) {
	 std::stringstream ss;
	 ss << "Empty group name. line: " << line_num << "\n";
	 (*warn) += ss.str();
	 name = "";
       }
     } else {
       std::stringstream ss;
       ss << names[1];

       // tinyobjloader does not support multiple groups for a primitive.
       // Currently we concatinate multiple group names with a space to get
       // single group name.

       for (size_t i = 2; i < names.size(); i++) {
	 ss << " " << names[i];
       }

       name = ss.str();
     }

     continue;
   }
 
   // object name
   if (token[0] == 'o' && IS_SPACE((token[1]))) {
     // flush previous face group.
     bool ret = exportGroupsToShape(&shape, prim_group, material, name,
				    triangulate, v, warn);
     (void)ret;  // return value not used.

     if (shape.mesh.indices.size() > 0) {
       shapes->push_back(shape);
     }

     // material = -1;
     prim_group.clear();
     shape = shape_t();

     // @todo { multiple object name? }
     token += 2;
     std::stringstream ss;
     ss << token;
     name = ss.str();

     continue;
   }
 
   // Ignore unknown command.
 }
 
 // not all vertices have colors, no default colors desired? -> clear colors
 if (!found_all_colors && !default_vcols_fallback) {
   vc.clear();
 }

 if (greatest_v_idx >= static_cast<int>(v.size() / 3)) {
   if (warn) {
     std::stringstream ss;
     ss << "Vertex indices out of bounds (line " << line_num << ".)\n\n";
     (*warn) += ss.str();
   }
 }
 if (greatest_vn_idx >= static_cast<int>(vn.size() / 3)) {
   if (warn) {
     std::stringstream ss;
     ss << "Vertex normal indices out of bounds (line " << line_num << ".)\n\n";
     (*warn) += ss.str();
   }
 }
 if (greatest_vt_idx >= static_cast<int>(vt.size() / 2)) {
   if (warn) {
     std::stringstream ss;
     ss << "Vertex texcoord indices out of bounds (line " << line_num << ".)\n\n";
     (*warn) += ss.str();
   }
 }

 bool ret = exportGroupsToShape(&shape, prim_group, material, name,
				triangulate, v, warn);
 // exportGroupsToShape return false when `usemtl` is called in the last
 // line.
 
 if (ret || shape.mesh.indices
     .size()) {  // FIXME(syoyo): Support other prims(e.g. lines)
   shapes->push_back(shape);
 }
 prim_group.clear();  // for safety

 if (err) {
   (*err) += errss.str();
 }

 attrib->vertices.swap(v);
 attrib->vertex_weights.swap(v);
 attrib->normals.swap(vn);
 attrib->texcoords.swap(vt);
 attrib->texcoord_ws.swap(vt);
 attrib->colors.swap(vc);
 
 return true;
}

#endif
