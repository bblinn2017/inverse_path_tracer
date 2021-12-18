#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <math.h>
#include <array>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <Eigen/Geometry>

#ifdef USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif 

typedef Eigen::Vector3f vecF;
typedef Eigen::Vector3i vecI;

enum shape_t {Cube=0,Sphere=1,Other=2};
enum matParam_t {Ambient=0,Diffuse=1,Specular=2,Shininess=3,IOR=4,Emissive=5};

struct mat_t {
  real_t ambient[3];
  real_t diffuse[3];
  real_t specular[3];
  real_t shininess;
  real_t ior;
  real_t emissive[3];
  
  void SetParameter(matParam_t mp, real_t *values) {

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

struct triangle_t {
  
  int idx;
  mat_t material;
  vecF vertices[3];
  vecF normal;

  void initialize(int i, mat_t m, vecF vs[]) {
    idx = i;
    material = m;
    for (int j = 0; j < 3; j++){
      vertices[j] = vs[j];
    }
    vecF a = vertices[1] - vertices[0];
    vecF b = vertices[2] - vertices[1];
    normal = a.cross(b);
  }
};

struct mesh_t {
  float position[3];
  mat_t material;
  thrust::host_vector<vecF> vertices;
  thrust::host_vector<vecI> faces;		  
};

struct bbox_t {
  vecF min;
  vecF max;
};

class Object {
 public:
  Object(float *pos,
	 float *ori,
	 float *scl,
	 shape_t shp,
	 std::string obj_file,
	 std::string mtl_file) {
    
    // Transform                                                                                     
    Eigen::Affine3f t = Eigen::Affine3f::Identity();
    t.scale(vecF(scl[0],scl[1],scl[2]));
    vecF rot = vecF(ori[0],ori[1],ori[2]);
    float angle = rot.norm();
    rot.normalize();
    t.rotate(Eigen::AngleAxisf(angle,rot));
    t.translate(vecF(pos[0],pos[1],pos[2]));
    Eigen::Matrix4f T = t.matrix();

    // Set Vertices, Faces, Triangles                                            
    std::string filename_c;
    switch(shp) {
    case Cube:
      filename_c = "shapes/cube.obj";
      break;
    case Sphere:
      filename_c = "shapes/sphere.obj";
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
    while (getline(file,currentLine)) {
      char *cstr = new char[currentLine.length() + 1];
      strcpy(cstr, currentLine.c_str());
      if (strncmp(cstr,"v ",2) == 0) {
        m_mesh.vertices.push_back(processVertex(currentLine,T));
      }
      else if (strncmp(cstr,"f ",2) == 0) {
        m_mesh.faces.push_back(processFace(currentLine));
      }
      delete [] cstr;
    }
    file.close();
    // Set Triangles and Bounding Box
    for (int i = 0; i < m_mesh.faces.size(); i++) {
      vecI face = m_mesh.faces[i];
      vecF tri_v[] = {m_mesh.vertices[face[0]],
		      m_mesh.vertices[face[1]],
		      m_mesh.vertices[face[2]]};
      // TODO TRIANGLE
    }
    Eigen::Vector4f bmin; Eigen::Vector4f bmax;
    if (shp == Cube || shp == Sphere) {
      bmin = Eigen::Vector4f(-.5,-.5,-.5,1.);
      bmax = Eigen::Vector4f(.5,.5,.5,1.);
    } else {
      bmin = Eigen::Vector4f(0.,0.,0.,1.);
      bmax = Eigen::Vector4f(0.,0.,0.,1.);
      for (int i = 0; i < m_mesh.vertices.size(); i++) {
	for (int j = 0; j < 3; j++) {
	  bmin[j] = min(bmin[j],m_mesh.vertices[i][j]);
	  bmax[j] = min(bmax[j],m_mesh.vertices[i][j]);
	}
      }
    }
    bmin = T * bmin;
    bmax = T * bmax;
    m_bbox.min = bmin.head<3>();
    m_bbox.max = bmax.head<3>();
       
    // Set Position 
    for (int i = 0; i < 3; i++) {
      m_mesh.position[i] = pos[i];
    }

    // Set Material                                                                                  
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
        m_mesh.material.SetParameter(mp,params_p);
      }
      file.close();
    } else { // Generate Random Material                                                             
      real_t params[3];
      matParam_t mps[] = {matParam_t::Ambient,matParam_t::Diffuse,matParam_t::Specular};
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          params[j] = (real_t) rand();
        }
        m_mesh.material.SetParameter(mps[i],params);
      }

      real_t shininess = (real_t) rand();
      real_t ior = (real_t) rand();

      m_mesh.material.SetParameter(matParam_t::Shininess,&shininess);
      m_mesh.material.SetParameter(matParam_t::IOR,&ior);
    }
  };
 private:
  mesh_t m_mesh;
  bbox_t m_bbox;

  std::vector<std::string> readValues(std::string line) {
    std::vector<std::string> c_strings = std::vector<std::string>();

    std::string string;
    bool space;
    for (int i = 0; i < line.size(); i++) {
      if (std::isdigit(line[i]) || line[i] == '.' || line[i] == '-') {
        string.push_back(line[i]);
        space = false;
      } else if(!space) {
        c_strings.push_back(string);
        string.clear();
        space = true;
      }
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

  vecF processVertex(std::string line, Eigen::Matrix4f T) {
    std::vector<std::string> strs = readValues(line);
    float xyz[3];

    for (int i = 0; i < strs.size(); i++) {
      char *cstr = new char[strs[i].length() + 1];
      strcpy(cstr, strs[i].c_str());
      xyz[i] = (float) atof(cstr);
      delete [] cstr;
    }
    
    Eigen::Vector4f vertex = Eigen::Vector4f(xyz[0],xyz[1],xyz[2],1.);
                                         
    vertex = T * vertex;
    return vertex.head<3>();
  }

  vecI processFace(std::string line) {
    std::vector<std::string> strs = readValues(line);

    vecI values;
    for (int i = 0; i < 3; i ++) {
      char *cstr = new char[strs[i].length() + 1];
      strcpy(cstr, strs[i].c_str());
      values[i] = (int) atoi(cstr);
      delete [] cstr;
    }
    return values;
  }
};
