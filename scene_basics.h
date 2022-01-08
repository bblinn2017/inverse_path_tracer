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
#define EPSILON 1e-2

#ifdef USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

typedef Eigen::Vector3f vecF;
typedef Eigen::Vector3i vecI;
typedef Eigen::VectorXf vecXF;
typedef Eigen::Matrix4f mat4F;
typedef Eigen::Matrix3f mat3F;
typedef Eigen::MatrixX3f matXF;
typedef Eigen::Affine3f aff3F;

__host__ __device__ vecF CMIN(vecF x, vecF y) {return vecF(min(x[0],y[0]),min(x[1],y[1]),min(x[2],y[2]));}
__host__ __device__ vecF CMAX(vecF x, vecF y) {return vecF(max(x[0],y[0]),max(x[1],y[1]),max(x[2],y[2]));}
__host__ __device__ void PRINT(vecF v) {printf("%f %f %f\n",v[0],v[1],v[2]);}
__host__ __device__ void PRINT(Eigen::MatrixXf m) {for (int i = 0; i < m.rows(); i++) {for(int j = 0; j < m.cols(); j++) {printf("%f ",m(i,j));} printf("\n");}}

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

  mat_t() {}
};

class Triangle {
 public:
  int idx, idxE;
  mat_t material;
  vecF vertices[3];
  mat3F normals;
  vecF normal;
  vecF center;
  float area;

  __host__ __device__ Triangle(int i, material_t m, vecF vs[], vecF *ns) 
    : material(m)
{
    idx = i;
    idxE = -1;
    
    center = vecF::Zero();
    for (int j = 0; j < 3; j++){
      vertices[j] = vs[j];
      center += vs[j]/3.;
    }
    
    vecF a = vertices[1] - vertices[0];
    vecF b = vertices[2] - vertices[1];

    normal = a.cross(b);
    area = normal.norm() / 2.;
    normal.normalize();
    
    for (int i = 0; i < 3; i++) {
      normals.col(i) = (ns) ? ns[i] : normal;
    }
  }

  __host__ __device__ Triangle() {}
  
  __host__ __device__ vecF getNormal(vecF point) {
    vecF weights; float weight;
    for (int i = 0; i < 3; i++) {
      weight = 0.5 * abs((vertices[(i+1)%3] - point).cross(vertices[(i+2)%3] - point).norm()) / area;
      weights[i] = weight;
    }
    vecF normal = normals * weights;
    normal.normalize();
    return normal;
  }
};

struct ObjParams_t{

  vecF pos;
  vecF ori;
  vecF scl;
  char *obj_file;
  char *mtl_file;

  ObjParams_t(float p[3], float o[3], float sc[3], char *ob, char *m) {
    pos = vecF(p[0],p[1],p[2]);
    ori = vecF(o[0],o[1],o[2]);
    scl = vecF(sc[0],sc[1],sc[2]);
    obj_file = ob;
    mtl_file = m;
  }

  ObjParams_t(vecF p, vecF o, vecF sc, char *ob, char *m) {
    pos = p;
    ori = o;
    scl = sc;
    obj_file = ob;
    mtl_file = m;
  }

  ObjParams_t() {}
};

class Mesh {
 public:
  vecF m_position;
  Triangle *m_triangles;
  Triangle **m_emissives;
  
  int m_nV,m_nF,m_nT,m_nE;

  Mesh(ObjParams_t obj) {
    // Transform
    aff3F T = aff3F::Identity();
    // Translate
    T.translate(obj.pos);
    // Rotate
    float angle = obj.ori.norm();
    obj.ori.normalize();
    T.rotate(Eigen::AngleAxisf(angle,obj.ori));
    // Scale
    T.scale(obj.scl);

    std::vector<vecF> vs;
    std::vector<vecF> ns;
    std::vector<vecI> fs;
    std::vector<material_t> ms;

    if (!ParseFromString(obj.obj_file,obj.mtl_file,&vs,&ns,&fs,&ms,T)) {exit(1);}

    m_position = obj.pos;

    cudaMallocManaged(&m_triangles,sizeof(Triangle)*fs.size());
    std::vector<Triangle *> es;
    for (int i = 0; i < fs.size(); i++) {
      vecI face = fs[i];
      vecF tri_v[] = {vs[face[0]],
                      vs[face[1]],
                      vs[face[2]]};
      vecF *tri_n = NULL;
      if (ns.size() == vs.size()) {
	vecF normals[] = {ns[face[0]],
			  ns[face[1]],
			  ns[face[2]]};
	tri_n = normals;
      }
      new(m_triangles+i) Triangle(i,ms[i],tri_v,tri_n);
      real_t *e = ms[i].emission;
      if (e[0] > 0. || e[1] > 0. || e[2] > 0.) {
	m_triangles[i].idxE = es.size();
	es.push_back(&(m_triangles[i]));
      }
    }
    m_nT = fs.size();
    
    cudaMallocManaged(&m_emissives,sizeof(Triangle *)*es.size());
    cudaMemcpy(m_emissives,es.data(),sizeof(Triangle *)*es.size(),cudaMemcpyHostToDevice);
    m_nE = es.size();
  }

  Mesh() {}
  
  ~Mesh() {
    for (int i = 0; i < m_nT; i++) {
      (m_triangles+i)->~Triangle();
    }
    cudaFree(m_triangles);
    cudaFree(m_emissives);
  }

 private:
  bool ParseFromString(const std::string &obj_file,
                       const std::string &mtl_file,
                       std::vector<vecF> *vertices,
                       std::vector<vecF> *vert_normals,
                       std::vector<vecI> *faces,
                       std::vector<material_t> *face_materials,
                       aff3F T) {

    std::filebuf obj_fb;
    if(!obj_fb.open(obj_file,std::ios_base::in)) {
      printf("Error: Object File was not able to be opened\n");
      std::cerr << "Error: " << strerror(errno) << std::endl;
      exit(1);
    }
    std::istream obj_ifs(&obj_fb);
    
    bool inString = mtl_file[0] == '*';
    std::ifstream mtl_ifs(inString ? NULL : mtl_file.c_str());    
    MaterialStreamReader mtl_ss(mtl_ifs);
    
    attrib_t attrib;
    std::vector<shape_t> shapes;
    std::vector<material_t> materials;
    std::string warning;
    std::string error;

    bool valid_ = LoadObj(&attrib, &shapes, &materials, &warning, &error, &obj_ifs, &mtl_ss, true, true);

    std::vector<real_t> v_xyz = attrib.vertices;
    std::vector<real_t> n_xyz = attrib.normals;
    for (int i = 0; i < v_xyz.size() / 3; i++) {
      vertices->push_back(T * vecF(v_xyz[3*i],v_xyz[3*i+1],v_xyz[3*i+2]));
    }
    mat3F T_2 = T.linear().transpose().inverse();
    for (int i = 0; i < n_xyz.size() / 3; i++) {
      vert_normals->push_back(T_2 * vecF(n_xyz[3*i],n_xyz[3*i+1],n_xyz[3*i+2]));
    }

    std::vector<index_t> f_idx;
    std::vector<int> mat_ids;

    // Include mat
    material_t rand_mat;
    InitMaterial(&rand_mat);
    if (inString) {
      std::stringstream ss(mtl_file.substr(1,mtl_file.size()-2));
      std::string line;
      while(std::getline(ss,line,'\n')) {
	const char *token = line.c_str();
	if (token[0] == 'K' && IS_SPACE((token[2]))) {
	  char k = token[1];
	  token += 2;
	  real_t r, g, b;
	  parseReal3(&r, &g, &b, &token);
	  if (k == 'd') {
	    rand_mat.diffuse[0] = r;
	    rand_mat.diffuse[1] = g;
	    rand_mat.diffuse[2] = b;
	  }
	}
      }
    }

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

  __host__ __device__ operator bool() {return tri != NULL;}
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
    min = CMIN(min,o);
    max = CMAX(max,o);
    extent = max - min;
  }

  void expandToInclude(bbox_t o) {
    min = CMIN(min,o.min);
    max = CMAX(max,o.max);
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

    vecF lmax = CMAX(l1,l2);
    vecF lmin = CMIN(l1,l2);

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
  
 Object(ObjParams_t obj) : m_mesh(obj) {
    // Set Bounding Box
    m_bbox.setP(obj.pos);
    for (int i = 0; i < m_mesh.m_nT; i++) {
      for (int j = 0; j < 3; j++) {
	m_bbox.expandToInclude(m_mesh.m_triangles[i].vertices[j]);
      }
    }
  };

  __host__ __device__ Object() {}

  __host__ __device__ bbox_t getBBox() {
    bbox_t box = m_bbox;
    return box;
  }

  __host__ __device__ vecF getPosition() {
    return vecF(m_mesh.m_position);
  }

  __host__ __device__ void getIntersection(Ray ray, intersection_t &intersection) {
    
    float t, denom, sd;
    vecF normal, center, point;
    Triangle *tri;
    bool has_int;
    int nT = m_mesh.m_nT;
    
    for (int i = 0; i < nT; i++) {
      tri = &(m_mesh.m_triangles[i]);

      center = tri->center;
      normal = tri->normal;

      denom = normal.dot(ray.d);
      if (abs(denom) < MIN_DOT) {continue;}
      
      t = (ray.p - center).dot(normal) / -denom;
      if (t < EPSILON || t >= intersection.t) {continue;}

      point = ray.p + ray.d * t;
      has_int = true;
      for (int j = 0; j < 3; j++) {
	sd = signedDistance(point,tri->vertices[j],tri->vertices[(j+1)%3],normal);
	if (sd > 0) {has_int = false; break;}
      }
      
      if (!has_int) {continue;}

      intersection.t = t;
      intersection.hit = point;
      intersection.tri = tri;
    }
  }

  __host__ __device__ Triangle** getEmissives() {return m_mesh.m_emissives;}

  __host__ __device__ int nEmissives() {return m_mesh.m_nE;}

  __host__ __device__ int nTriangles() {return m_mesh.m_nT;}

  void setOffsets(int currT, int currE) {
    for (int i = 0; i < m_mesh.m_nT; i++) {
      m_mesh.m_triangles[i].idx += currT;
    }
    for (int i = 0; i < m_mesh.m_nE; i++) {
      m_mesh.m_emissives[i]->idxE += currE;
    }
  }

  std::vector<float> getMaterials() {
    std::vector<float> materials;
    for (int i = 0; i < m_mesh.m_nT; i++) {
      mat_t mat = (m_mesh.m_triangles + i)->material;
      for (int j = 0; j < 3; j++) {materials.push_back(mat.diffuse[j]);}
    }
    return materials;
  }

  void setMaterials(std::vector<float> materials) {
    int idx = 0;
    for (int i = 0; i < m_mesh.m_nT; i++) {
      Triangle *t = m_mesh.m_triangles + i;
      for (int j = 0; j < 3; j++) {t->material.diffuse[j] = materials[idx]; idx++;}
    }
  }

 private:
  Mesh m_mesh;
  bbox_t m_bbox;

  __host__ __device__ float signedDistance(vecF point, vecF s0, vecF s1, vecF normal) {
    
    vecF out = (s1 - s0).cross(normal);
    out.normalize();
    float d = -out.dot(s1 + s0) / 2.;
    return point.dot(out) + d;
  }

  __host__ __device__ vecXF signedDistance(matXF point, matXF s0, matXF s1, matXF normal) {
    
    matXF diff  = s1 - s0;
    matXF out(s0.rows(),3);
    for (int i = 0; i < s0.rows(); i++) {
      out.row(i) = diff.row(i).cross(normal.row(i));
      out.row(i) /= out.row(i).norm();
    }
    matXF mean = (s1 + s0) / 2.;
    vecXF d = -out.cwiseProduct(mean).rowwise().sum();
    return point.cwiseProduct(out).rowwise().sum() + d;
  }
};
