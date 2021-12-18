#include <stdlib.h>
#include <math.h>
#include <array>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <Eigen/Geometry>

#ifdef USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

typedef thrust::device_vector<T> vector<T> 

enum shape_t {Cube=0,Sphere=1,Other=2};
enum matParam_t {Ambient=0,Diffuse=1,Specular=2,Shininess=3,IOR=4,Emissive=5};

struct mat_t {
  real_t ambient[3];
  real_t diffuse[3];
  real_t specular[3];
  real_t shininess;
  real_t ior;
  real_t emissive[3];
  
  void SetParameter(matParam_t mp, double *values) {

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
      shininess = *values;
      break;
    case IOR:
      ior = *values;
      break;
    case Emissive:
      for (int i = 0; i < 3; i++) {
	emissive[i] = values[i];
      }
    default:
      printf("Error: Material does not contain this parameter\n");
      exit(1);
    }
};

struct triangle_t {
  
  int idx;
  mat_t material;
  real_t vertices[3][3];
  real_t normal[3];

  void initialize(int i, mat_t m, array<array<double>> vs, array<double> n) {
    idx = i;
    material = m;
    for (int j = 0; j < 3; j++){
      for (int k = 0; k < 3; k++) {
	vertices[j][k] = real_t(vs[j][k]);
      }
      normal[j] = real_t(n[j]);
    }
  }

  
};

struct mesh_t {
  real_t position[3];
  mat_t material;
  vector<vector<real_t>> vertices;
  vector<vector<int>> faces;		  
};

class Object {
 public:
  Object(array<double> pos,
	 array<double> ori,
	 array<double> scl,
	 shape_t shp,
	 std::string obj_file,
	 std::string mtl_file) {
    
    build_mesh(pos,ori,scl,mat,shp,obj_file,mtl_file);
  };
 private:
  mesh_t mesh;

  std::vector<char *> readValues(std::string line) {
    std::vector<char *> c_strings = std::vector<char *>();

    std::string string;
    for (int i = 0; i < line.size(); i++) {
      if (std::isdigit(line[i]) || line[i] == '.' || line[i] == '-') {
        string.push_back(line[i]);
        space = False;
      } else if(!space) {
        c_strings.push_back(string.c_str());
        string.clear();
        space = True;
      }
    }

    return c_strings;
  }

  array<double> processMatParams(std::string line) {
    std::vector<char *> strs = readValues(line);

    array<double> values(3);
    for (int i = 0; i < strs.size(); i++) {
      values[i] = double(atod(strs[i].c_str()));
    }
    return values;
  }

  vector<real_t> processVertex(std::string line, Eigen::Matrix4d T) {
    std::vector<char *> strs = readValues(line);

    Eigen::Vector4d vertex = Eigen::Vector4d(
                                             double(atod(strs[0].c_str())),
                                             double(atod(strs[1].c_str())),
                                             double(atod(strs[2].c_str())),
                                             1.
                                             );
    vertex = T * vertex;
    vector<real_t> values(3);
    for (int i = 0; i < 3; i++) {
      values[i] = real_t(vertex[i]);
    }
    return values;
  }

  vector<int> processFace(std::string line) {
    std::vector<char *> strs = readValues(line);

    vector<int> values(3);
    for (int i = 0; i < 3; i ++) {
      values[i] = int(atoi(strs[i].c_str()));
    }
    return values;
  }

  void build_mesh(array<double> pos,
                  array<double> ori,
                  array<double> scl,
                  shape_t shp,
                  std::string obj_file,
		  std::string mat_file) {
    
    // Transform                                                                                     
    Eigen::Transform<double,3,Eigen::TransformTraits::Affine> t;
    t.setIdentity();
    t.scale(Eigen::Vector3d(scl[0],scl[1],scl[2]));
    Eigen::Vector3d rot = Eigen::Vector3d(ori[0],ori[1],ori[2]);
    double angle = rot.norm();
    rot.normalize();
    t.rotate(Eigen::AngleAxisd(angle,rot));
    t.translate(pos[0],pos[1],pos[2]);
    Eigen::Matrix4d T = t.matrix();

    // Set Vertices, Faces, Triangles, and Bounding Box
    char * filename_c;
    switch(shp) {
    case Cube:
      filename_c = "shapes/cube.obj";
      break;
    case Sphere:
      filename_c = "shapes/sphere.obj";
      break;
    case Other:
      filename_c = const_cast<char*>(filename.c_str());
    default:
      printf("Error: This shape type is not allowed\n");
      exit(1);
    }
    std::ifstream file(filename_c);
    if (!file.is_open()) {
      printf("Error: Object file was not able to be opened");
      exit(1);
    }
    
    std::string currentLine;
    while (getline(file,currentLine)) {
      if (currentLine[0] == 'v' && currentLine[1] == ' ') {
	m_mesh.vertices.push_back(processVertex(currentLine,T));
      } 
      else if (currentLine[0] == 'f' && currentLine[1] == ' ') {
	m_mesh.faces.push_back(processFace(currentLine));
      }
    }
    file.close();
    
    // TODO: Set Triangles and Bounding Box
    
    // Set Position
    for (int i = 0; i < 3; i++) {
      m_mesh.position[i] = pos[i];
    }

    // Set Material
    if (*mat_file != NULL) {
      filename_c = mat_file.c_str();
      file.open(filename_c);

      if (!file.is_open()) {
	printf("Error: Material file was not able to be opened");
	exit(1);
      }
      
      while (getline(file,currentLine)) {
	array<double> params = processMatParams(currentLine);
	matParam_t mp;
	if (strncmp(currentLine,"Ka") == 0) {
	  mp = matParam_t::Ambient;
	} else if (strncmp(currentLine,"Kd") == 0) {
	  mp = matParam_t::Diffuse;
	} else if (strncmp(currentLine,"Ks") == 0) {
	  mp = matParam_t::Specular;
	} else if (strncmp(currentLine,"Ns") == 0) {
	  mp = matParam_t::Shininess;
	} else if (strncmp(currentLine,"Ni") == 0) {
	  mp = matParam_t::IOR;
	} else if (strncmp(currentLine,"Ke") == 0) {
	  mp = matParam_t::Emissive;
	}
	
	m_mesh.mat.SetParameter(mp,params);	
      }
      file.close();
    } else { // Generate Random Material
      array<double> params = array<double>(3);
      matParam_t mps[] = {matParam_t::Ambient,matParam_t::Diffuse,matParam_t::Specular};
      for (int i = 0; i < 3; i++) {
	for (int j = 0; j < 3; j++) {
	  params[j] = double(rand());
	}
	m_mesh.mat.SetParameter(mps[i],params);
      }

      double shininess = double(rand());
      double ior = double(rand());
      
      m_mesh.mat.SetParameter(matParam_t::Shininess,&shininess);
      m_mesh.mat.SetParameter(matParam_t::IOR,&ior);
    }
  };
}
