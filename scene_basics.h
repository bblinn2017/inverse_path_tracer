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

#define MIN_DOT 1e-4

#ifdef USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

typedef Eigen::Vector3f vecF;
typedef Eigen::Vector3i vecI;
typedef Eigen::Matrix4f mat4F;
typedef Eigen::Matrix3f mat3F;

enum shape_t {Cube=0,Sphere=1,Other=2};
enum matParam_t {Ambient=0,Diffuse=1,Specular=2,Shininess=3,IOR=4,Emissive=5};

struct mat_t {
  real_t ambient[3];
  real_t diffuse[3];
  real_t specular[3];
  real_t shininess;
  real_t ior;
  real_t emissive[3];
  
  __host__ __device__ void SetParameter(matParam_t mp, real_t *values) {

    switch (mp) {
    case Ambient:
      for (int i = 0; i < 3; i++) {
        ambient[i] = values[i];
      }
      break;
    case Diffuse:
      for (int i = 0; i< 3; i++) {
	diffuse[i] = values[i];
      }
      break;
    case Specular:
      for (int i = 0; i< 3; i++) {
	specular[i] = values[i];
      }
      break;
    case Shininess:
      shininess = values[0];
      break;
    case IOR:
      ior = values[0];
      break;
    case Emissive:
      for (int i = 0; i < 3; i++) {
	emissive[i] = values[i];
      }
      break;
    default:
      printf("Error: Material does not contain this parameter\n");
      exit(1);
      break;
    }
  }
};

class Triangle {
 public:
  int idx;
  mat_t material;
  vecF vertices[3];
  vecF normal;
  vecF center;

  __host__ __device__ Triangle(int i, mat_t m, vecF vs[]) {
    idx = i;
    material = m;
    center = vecF::Zero();
    for (int j = 0; j < 3; j++){
      vertices[j] = vs[j];
      center += vs[j]/3.;
    }
    vecF a = vertices[1] - vertices[0];
    vecF b = vertices[2] - vertices[1];
    normal = a.cross(b);
    normal.normalize();
  }

  __host__ __device__ Triangle() {}
  
  __host__ __device__ float area() {
    return (vertices[0]-vertices[1]).cross(vertices[1]-vertices[2]).norm() / 2.;
  }
};

class Mesh {
 public:
  vecF m_position;
  mat_t m_material;
  vecF *m_vertices;
  vecI *m_faces;
  Triangle *m_triangles;

  int m_nV,m_nF,m_nT;

  Mesh(vecF pos, mat_t mat, std::vector<vecF> vs, std::vector<vecI> fs) {
    m_position = pos;
    m_material = mat;

    cudaMallocManaged(&m_vertices,sizeof(vecF)*vs.size());
    cudaMemcpy(m_vertices,vs.data(),sizeof(vecF)*vs.size(),cudaMemcpyHostToDevice);
    m_nV = vs.size();

    cudaMallocManaged(&m_faces,sizeof(vecI)*fs.size());
    cudaMemcpy(m_faces,fs.data(),sizeof(vecI)*fs.size(),cudaMemcpyHostToDevice);
    m_nF = fs.size();

    cudaMallocManaged(&m_triangles,sizeof(Triangle)*fs.size());
    for (int i = 0; i < fs.size(); i++) {
      vecI face = m_faces[i];
      vecF tri_v[] = {m_vertices[face[0]],
                      m_vertices[face[1]],
                      m_vertices[face[2]]};
      m_triangles[i] = Triangle(i,m_material,tri_v);
    }
    m_nT = fs.size(); 
  }

  ~Mesh() {
    cudaFree(m_vertices);
    cudaFree(m_faces);
    cudaFree(m_triangles);
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
  bool hit;
  Triangle *tri;

  intersection_t() {
    t = INFINITY;
    hit = false;
    tri = NULL;
  }
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
    mat3F div_arg;
    for (int i = 0; i < 3; i++) {div_arg(i,i) = 1./ray.d[i];}
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
  __host__ Object(shape_t shp,
	 vecF pos=vecF(0.,0.,2.5),
	 vecF ori=vecF(0.,0.,0.),
	 vecF scl=vecF(1.,1.,1.),
	 std::string obj_file="",
	 std::string mtl_file="") {

    // Transform                                                                                   
    Eigen::Affine3f T = Eigen::Affine3f::Identity();
    T.scale(scl);
    float angle = ori.norm();
    ori.normalize();
    T.rotate(Eigen::AngleAxisf(angle,ori));
    T.translate(pos);

    // Set Vertices, Faces                                            
    std::string filename_c;
    switch(shp) {
    case Cube:
      filename_c = "/users/bblinn/pt_inv/shapes/cube.obj";
      break;
    case Sphere:
      filename_c = "/users/bblinn/pt_inv/shapes/sphere.obj";
      break;
    case Other:
      filename_c = obj_file;
      break;
    default:
      printf("Error: This shape type is not allowed\n");
      exit(1);
      break;
    }
    char *cstr = new char[filename_c.length() + 1];
    strcpy(cstr, filename_c.c_str());
    std::ifstream file(cstr);
    delete [] cstr;
    if (!file.is_open()) {
      printf("Error: Object file was not able to be opened\n");
      exit(1);
    }

    std::string currentLine;
    std::vector<vecF> vertices;
    std::vector<vecI> faces;
    while (getline(file,currentLine)) {
      char *cstr = new char[currentLine.length() + 1];
      strcpy(cstr, currentLine.c_str());
      if (strncmp(cstr,"v ",2) == 0) {
        vertices.push_back(T * processVertex(currentLine));
      }
      else if (strncmp(cstr,"f ",2) == 0) {
        faces.push_back(processFace(currentLine));
      }
      delete [] cstr;
    }
    file.close();
    // Set Bounding Box
    for (int i = 0; i < vertices.size(); i++) {
      m_bbox.expandToInclude(vertices[i]);
    }

    // Set Material
    mat_t mat;
    if (!mtl_file.empty()) {
      filename_c = mtl_file.c_str();
      file.open(filename_c);

      if (!file.is_open()) {
        printf("Error: Material file was not able to be opened");
        exit(1);
      }
 
      while (getline(file,currentLine)) {
	std::vector<real_t> params = processMatParams(currentLine);
	char *cstr = new char[currentLine.length() + 1];
	strcpy(cstr, currentLine.c_str());
        matParam_t mp;
        if (strncmp(cstr,"Ka",2) == 0) {
          mp = matParam_t::Ambient;
        } else if (strncmp(cstr,"Kd",2) == 0) {
          mp = matParam_t::Diffuse;
        } else if (strncmp(cstr,"Ks",2) == 0) {
          mp = matParam_t::Specular;
        } else if (strncmp(cstr,"Ns",2) == 0) {
          mp = matParam_t::Shininess;
        } else if (strncmp(cstr,"Ni",2) == 0) {
          mp = matParam_t::IOR;
        } else if (strncmp(cstr,"Ke",2) == 0) {
          mp = matParam_t::Emissive;
        }
	
	real_t params_p[3];
	for (int i = 0; i < params.size(); i++) {
	  params_p[i] = params[i];
	}
        mat.SetParameter(mp,params_p);
      }
      file.close();
    } else { // Generate Random Material                                                             
      real_t params[3];
      matParam_t mps[] = {matParam_t::Ambient,matParam_t::Diffuse,matParam_t::Specular};
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          params[j] = (real_t) rand();
        }
        mat.SetParameter(mps[i],params);
      }

      real_t shininess = (real_t) rand();
      real_t ior = (real_t) rand();

      mat.SetParameter(matParam_t::Shininess,&shininess);
      mat.SetParameter(matParam_t::IOR,&ior);
    }  
    
    // Create Mesh
    cudaMallocManaged(&m_mesh,sizeof(Mesh));
    *m_mesh = Mesh(pos,mat,vertices,faces);
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
    
    for (int i = 0; i < m_mesh->m_nV; i++) {
      vecF v = m_mesh->m_vertices[i];
      printf("%f %f %f\n",v[0],v[1],v[2]);
    }
    for (int i = 0; i < m_mesh->m_nT; i++) {
      tri = &(m_mesh->m_triangles[i]);

      normal = tri->normal;
      center = tri->center;

      denom = abs(normal.dot(ray.d));
      if (denom < MIN_DOT) {continue;}
      
      t = (ray.p - center).dot(normal) / denom;
      if (t < 0 || t > intersection.t) {continue;}

      point = ray.p + ray.d * t;
      has_int = true;
      for (int j = 0; j < 3; j++) {
	sd = signedDistance(point,tri->vertices[j],tri->vertices[(j+1)%3],normal);
	if (sd > 0) {has_int = false;}
      }
      if (!has_int) {continue;}

      intersection.t = t;
      intersection.hit = has_int;
      intersection.tri = tri;
    }
  }
  
  Mesh *m_mesh;
 private:
  //Mesh *m_mesh;
  bbox_t m_bbox;

  __host__ __device__ float signedDistance(vecF point, vecF s0, vecF s1, vecF normal) {
    
    vecF out = (s1 - s0).cross(normal);
    out.normalize();
    float d = -out.dot(s1 + s0) / 2.;

    return point.dot(out) + d;
  }

  std::vector<std::string> readValues(std::string line) {
    std::vector<std::string> c_strings = std::vector<std::string>();

    std::string string;
    for (int i = 0; i < line.size(); i++) {
      if (std::isdigit(line[i]) || line[i] == '.' || line[i] == '-') {
        string.push_back(line[i]);
      } else if(string.size()) {
        c_strings.push_back(string);
        string.clear();
      }
    }
    if (string.size()) {
      c_strings.push_back(string);
    }
    return c_strings;
  }

  std::vector<real_t> processMatParams(std::string line) {
    std::vector<std::string> strs = readValues(line);

    std::vector<real_t> values(3);
    for (int i = 0; i < strs.size(); i++) {
      char *cstr = new char[strs[i].length() + 1];
      strcpy(cstr, strs[i].c_str());
      values[i] = (real_t) atof(cstr);
      delete [] cstr;
    }
    return values;
  }

  vecF processVertex(std::string line) {
    std::vector<std::string> strs = readValues(line);
    float xyz[3];

    vecF values;
    for (int i = 0; i < strs.size(); i++) {
      char *cstr = new char[strs[i].length() + 1];
      strcpy(cstr, strs[i].c_str());
      values[i] = (float) atof(cstr);
      delete [] cstr;
    }
    return values; 
  }

  vecI processFace(std::string line) {
    std::vector<std::string> strs = readValues(line);

    vecI values;
    for (int i = 0; i < 3; i ++) {
      char *cstr = new char[strs[i].length() + 1];
      strcpy(cstr, strs[i].c_str());
      values[i] = ((int) atoi(cstr)) - 1;
      delete [] cstr;
    }
    return values;
  }
};
