#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <math.h>
#include <array>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include "tiny_obj_loader.h" 

#define MIN_DOT 1e-4
#define EPSILON 1e-4

#ifdef USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

typedef Eigen::Vector3f vecF;
typedef Eigen::Vector3i vecI;
typedef Eigen::Matrix4f mat4F;
typedef Eigen::Matrix3f mat3F;
typedef Eigen::Affine3f aff3F;

enum shape_type_t {Cube=0,Sphere=1,Cornell=2,Other=3};

struct mat_t {

  real_t diffuse[3];
  real_t specular[3];
  real_t transmittance[3];
  real_t emission[3];
  real_t shininess;
  real_t ior;       // index of refraction
  real_t dissolve;
  int illum;

  void copy(void *dst, void *src, size_t size) {
    cudaMemcpy(dst,src,size,cudaMemcpyHostToDevice);
  }

  mat_t(material_t material) {
    copy(diffuse,material.diffuse,3*sizeof(real_t));
    copy(specular,material.specular,3*sizeof(real_t));
    copy(transmittance,material.transmittance,3*sizeof(real_t));
    copy(emission,material.emission,3*sizeof(real_t));
    copy(&shininess,&material.shininess,sizeof(real_t));
    copy(&ior,&material.ior,sizeof(real_t));
    copy(&dissolve,&material.dissolve,sizeof(real_t));
    copy(&illum,&material.illum,sizeof(int));
  }
};

class Triangle {
 public:
  int idx;
  mat_t *material;
  vecF vertices[3];
  vecF normal;
  vecF center;
  float area;

  __host__ __device__ Triangle(int i, material_t m, vecF vs[]) {
    idx = i;
    cudaMallocManaged(&material,sizeof(mat_t));
    new(material) mat_t(m);
    center = vecF::Zero();
    for (int j = 0; j < 3; j++){
      vertices[j] = vs[j];
      center += vs[j]/3.;
    }
    vecF a = vertices[1] - vertices[0];
    vecF b = vertices[2] - vertices[1];
    normal = a.cross(b);
    area = normal.norm();
    normal.normalize();
  }

  __host__ __device__ Triangle() {}

  __host__ __device__ ~Triangle() {
    cudaFree(material);
  }
};

class Mesh {
 public:
  vecF m_position;
  vecF *m_vertices;
  vecI *m_faces;
  Triangle *m_triangles;
  Triangle **m_emissives;
  float *m_probabilities;

  int m_nV,m_nF,m_nT,m_nE;

  Mesh(vecF pos, std::vector<material_t> ms, std::vector<vecF> vs, std::vector<vecI> fs) {
    m_position = pos;
    
    cudaMallocManaged(&m_vertices,sizeof(vecF)*vs.size());
    cudaMemcpy(m_vertices,vs.data(),sizeof(vecF)*vs.size(),cudaMemcpyHostToDevice);
    m_nV = vs.size();

    cudaMallocManaged(&m_faces,sizeof(vecI)*fs.size());
    cudaMemcpy(m_faces,fs.data(),sizeof(vecI)*fs.size(),cudaMemcpyHostToDevice);
    m_nF = fs.size();

    cudaMallocManaged(&m_triangles,sizeof(Triangle)*fs.size());
    std::vector<Triangle *> es;
    for (int i = 0; i < fs.size(); i++) {
      vecI face = m_faces[i];
      vecF tri_v[] = {m_vertices[face[0]],
                      m_vertices[face[1]],
                      m_vertices[face[2]]};
      new(&(m_triangles[i])) Triangle(i,ms[i],tri_v);
      real_t *e = ms[i].emission;
      if (e[0] > 0. || e[1] > 0. || e[2] > 0.) {
	es.push_back(&(m_triangles[i]));
      }
    }
    m_nT = fs.size();

    cudaMallocManaged(&m_emissives,sizeof(Triangle *)*es.size());
    cudaMemcpy(m_emissives,es.data(),sizeof(Triangle *)*es.size(),cudaMemcpyHostToDevice);
    m_nE = es.size();
  }
  ~Mesh() {
    cudaFree(m_vertices);
    cudaFree(m_faces);
    cudaFree(m_triangles);
    cudaFree(m_emissives);
  }

};

class Ray {
 public:
  vecF p;
  vecF d;

  __host__ __device__ Ray(vecF point, vecF direction) {
    p = point;
    d = direction;
  }

  __host__ __device__ Ray(const Ray &ray) {
    p = ray.p;
    d = ray.d;
  }

  __host__ __device__ void transform(mat4F m) {
    // 4D transformation
    Eigen::Vector4f p_temp;
    p_temp << p,1.;
    Eigen::Vector4f d_temp;
    d_temp << d,0.;
    
    p_temp = m * p_temp;
    d_temp = m * d_temp;
    p = p_temp.head<3>();
    d = d_temp.head<3>();
    d.normalize();
  }
};

__host__ __device__ struct intersection_t {
  float t;
  vecF hit;
  Triangle *tri;

  __host__ __device__ intersection_t() {
    t = INFINITY;
    hit = vecF::Zero();
    tri = NULL;
  }

  __host__ __device__ operator bool() {return t != INFINITY;}
};

__host__ __device__ struct bbox_t {
  vecF min = vecF::Zero();
  vecF max = vecF::Zero();
  vecF extent = vecF::Zero();

  void setP(vecF p) {
    min = p;
    max = p;
    extent = vecF::Zero();
  }

  void expandToInclude(vecF o) {
    Eigen::MatrixXf mat(2,3);
    // Max
    mat << min,
      o;
    min = mat.rowwise().minCoeff();
    // Min
    mat << max,
      o;
    max = mat.rowwise().maxCoeff();
    extent = max - min;
  }

  void expandToInclude(bbox_t o) {
    Eigen::MatrixXf mat(2,3);
    // Max
    mat<< min,
      o.min;
    min = mat.rowwise().minCoeff();
    // Min
    mat<< max,
      o.max;
    max = mat.rowwise().maxCoeff();
    extent = max - min;
  }

  int maxDimension() {
    int argmax = 0;
    float max = 0.;
    for (int i = 0; i < 3; i++) {
      if (extent[i] > max) {
	max = extent[i];
	argmax = i;
      }
    }
    return argmax;
  }

  float surfaceArea() {return vecF(extent[1],extent[2],extent[0]).dot(extent) * 2.;}

  __host__ __device__ bool intersect(Ray ray, float &tmin, float &tmax) {   
    aff3F div_arg = aff3F::Identity();
    div_arg.scale(ray.d.cwiseInverse());
    vecF l1 = div_arg * (min - ray.p);
    vecF l2 = div_arg * (max - ray.p);

    for (int i = 0; i < 3; i++) {
      if (std::isnan(l1[i])) {
	l1[i] = (min[i] - ray.p[i]) * INFINITY;
      }
      if (std::isnan(l2[i])) {
	l2[i] = (max[i] - ray.p[i]) * INFINITY;
      }
    }

    Eigen::MatrixXf m(2,3);
    m << l1,
      l2;
    vecF lmax = m.rowwise().maxCoeff();
    vecF lmin = m.rowwise().minCoeff();

    for (int i = 0; i < 3; i++) {
      if (lmin[i] > 0 || lmax[i] <= lmin[i]) {
	return false;
      }
    }
    tmin = lmin.maxCoeff();
    tmax = lmax.minCoeff();
    return true;
  }
};

class Object {
 public:
  __host__ Object(shape_type_t shp,
	 vecF pos=vecF(0.,0.,2.5),
	 vecF ori=vecF(0.,0.,0.),
	 vecF scl=vecF(1.,1.,1.),
	 std::string obj_file="",
	 std::string mtl_file="") {

    // Transform                                                                                   
    aff3F T = aff3F::Identity();
    T.scale(scl);
    float angle = ori.norm();
    ori.normalize();
    T.rotate(Eigen::AngleAxisf(angle,ori));
    T.translate(pos);

    // Load Mesh                                            
    switch(shp) {
    case Cube:
      obj_file = "/users/bblinn/pt_inv/shapes/cube.obj";
      break;
    case Sphere:
      obj_file = "/users/bblinn/pt_inv/shapes/sphere.obj";
      break;
    case Cornell:
      obj_file = "/users/bblinn/pt_inv/CornellBox/CornellBox-Empty-CO.obj";
      mtl_file = "/users/bblinn/pt_inv/CornellBox/CornellBox-Empty-CO.mtl";
    default:
      break;
    }

    std::vector<vecF> vertices;
    std::vector<vecI> faces;
    std::vector<material_t> materials;

    if (!ParseFromString(obj_file,mtl_file,&vertices,&faces,&materials,T)) {exit(1);}
    
    // Set Bounding Box
    for (int i = 0; i < vertices.size(); i++) {
      m_bbox.expandToInclude(vertices[i]);
    }

    // Create Mesh
    cudaMallocManaged(&m_mesh,sizeof(Mesh));
    m_mesh = new(m_mesh) Mesh(pos,materials,vertices,faces);
  };

  __host__ __device__ Object() {}

  __host__ __device__ bbox_t getBBox() {
    bbox_t box = m_bbox;
    return box;
  }

  __host__ __device__ vecF getPosition() {
    return vecF(m_mesh->m_position);
  }

  __host__ __device__ void getIntersection(Ray ray, intersection_t &intersection) {

    float t, denom, sd;
    vecF normal, center, point;
    Triangle *tri;
    bool has_int;
    
    for (int i = 0; i < m_mesh->m_nT; i++) {
      tri = &(m_mesh->m_triangles[i]);

      normal = tri->normal;
      center = tri->center;

      denom = normal.dot(ray.d);
      if (abs(denom) < MIN_DOT) {continue;}
      
      t = (ray.p - center).dot(normal) / -denom;
      if (t < EPSILON || t >= intersection.t) {continue;}
      
      point = ray.p + ray.d * t;
      has_int = true;
      for (int j = 0; j < 3; j++) {
	sd = signedDistance(point,tri->vertices[j],tri->vertices[(j+1)%3],normal);
	if (sd > 0) {has_int = false;}
      }
      
      if (!has_int) {continue;}

      intersection.t = t;
      intersection.hit = point;
      intersection.tri = tri;
    }
  }

  __host__ __device__ Triangle** getEmissives() {return m_mesh->m_emissives;}

  __host__ __device__ int nEmissives() {return m_mesh->m_nE;}

 private:
  Mesh *m_mesh;
  bbox_t m_bbox;

  bool ParseFromString(const std::string &obj_file,
		       const std::string &mtl_file,
		       std::vector<vecF> *vertices,
		       std::vector<vecI> *faces,
		       std::vector<material_t> *face_materials,
		       aff3F T) {

    std::filebuf obj_fb;
    if(!obj_fb.open(obj_file,std::ios::in)) {
      printf("Error: Object File was not able to be opened\n");
      exit(1);
    }
    std::istream obj_ifs(&obj_fb);
    
    std::filebuf mtl_fb;
    if(!mtl_fb.open(mtl_file,std::ios::in) && mtl_file != "") {
      printf("Error: Material File was not able to be opened\n");
      exit(1);
    }
    std::istream mtl_ifs(&mtl_fb);

    MaterialStreamReader mtl_ss(mtl_ifs);

    attrib_t attrib;
    std::vector<shape_t> shapes;
    std::vector<material_t> materials;
    std::string warning;
    std::string error;

    bool valid_ = LoadObj(&attrib, &shapes, &materials, &warning, &error, &obj_ifs, &mtl_ss, true, true);
    
    std::vector<real_t> v_xyz = attrib.vertices;
    for (int i = 0; i < v_xyz.size() / 3; i++) {
      vertices->push_back(T * vecF(v_xyz[3*i],v_xyz[3*i+1],v_xyz[3*i+2]));
    }

    std::vector<index_t> f_idx;
    std::vector<int> mat_ids;

    // Create random mat                                                                            
    material_t rand_mat;
    InitMaterial(&rand_mat);
    for (int i = 0; i < 3; i++) {rand_mat.diffuse[i] = (real_t) rand() / RAND_MAX; 
      rand_mat.specular[i] = (real_t) rand() / RAND_MAX;}
    rand_mat.shininess = (real_t) rand() / RAND_MAX;
    rand_mat.ior = (real_t) rand() / RAND_MAX;
    
    for (int i = 0; i < shapes.size(); i++) {
      f_idx = shapes[i].mesh.indices;
      mat_ids = shapes[i].mesh.material_ids;
      for (int j = 0; j < mat_ids.size(); j++) {
	faces->push_back(vecI(f_idx[3*j].vertex_index,
			      f_idx[3*j+1].vertex_index,
			      f_idx[3*j+2].vertex_index));
	int mat_id = mat_ids[j];
	material_t mat;
	if (mat_id != -1) {
	  mat = materials[mat_id];
	} else {
	  mat = rand_mat;
	}
	face_materials->push_back(mat);
      }
    }
    
    return valid_;
  }

  __host__ __device__ float signedDistance(vecF point, vecF s0, vecF s1, vecF normal) {
    
    vecF out = (s1 - s0).cross(normal);
    out.normalize();
    float d = -out.dot(s1 + s0) / 2.;

    return point.dot(out) + d;
  }
};
