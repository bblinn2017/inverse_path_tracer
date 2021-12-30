#include "bvh.h"

#define EYE vecF(0.,0.,0.)
#define LOOK vecF(0.,0.,1.)
#define UP vecF(0.,1.,0.)
#define HA 90.f
#define AR 1.f
#define IM_WIDTH 500
#define IM_HEIGHT 500
#define SAMPLE_NUM 100
#define p_RR .9f
#define BLOCKSIZE 256
#define NBLOCKS IM_WIDTH * IM_HEIGHT * SAMPLE_NUM / BLOCKSIZE + 1

struct CameraParams_t {
  float eye[3];
  float look[3];
  float up[3];
  float heightAngle;
  float aspectRatio;

  CameraParams_t(bool isDefault=false) {
    if (isDefault) {
      for (int i = 0; i < 3; i++) {
	eye[i] = EYE[i];
	look[i] = LOOK[i];
	up[i] = UP[i];
      }
      heightAngle = HA;
      aspectRatio = AR;
    }
  }
};

class Camera {
 public:
  Camera(CameraParams_t camera) {
    m_position = vecF(camera.eye[0],camera.eye[1],camera.eye[2]);
    m_direction = vecF(camera.look[0],camera.look[1],camera.look[2]);
    m_direction.normalize();
    m_up = vecF(camera.up[0],camera.up[1],camera.up[2]);
    m_up.normalize();
    m_heightAngle = M_PI * camera.heightAngle / 360.f;
    m_aspectRatio = camera.aspectRatio;
  }

  Camera() {}

  __host__ __device__ mat4F getViewMatrix() {

    vecF f = m_direction;
    f.normalize();
    vecF s = f.cross(m_up);
    s.normalize();
    vecF u = s.cross(f);
    u.normalize();

    mat4F vM;
    vM << s[0],s[1],s[2],-s.dot(m_position),
      u[0],u[1],u[2],-u.dot(m_position),
      f[0],f[1],f[2],-f.dot(m_position),
      0,0,0,1;

    return vM;
  }

  __host__ __device__ mat4F getScaleMatrix() {
    
    mat4F sM = mat4F::Identity();
    sM(0,0) = tan(m_heightAngle);
    sM(1,1) = tan(m_heightAngle*m_aspectRatio);
    return sM;
  }

  __host__ __device__ mat4F getInverseViewMatrix() {
    return getScaleMatrix() * getViewMatrix().transpose();
  }
 private:
  vecF m_position;
  vecF m_direction;
  vecF m_up;
  float m_heightAngle;
  float m_aspectRatio;
};

class Scene {
 public:
  
 Scene(CameraParams_t camera, std::vector<ObjParams_t> objects) : m_camera(camera) {
    
    m_nO = objects.size();
    cudaMallocManaged(&m_objects,sizeof(Object)*m_nO);
    for (int i = 0; i < m_nO; i++) {
      new(&(m_objects[i])) Object(objects[i]);
    }
    
    m_nT = 0;
    m_nE = 0;

    std::vector<Triangle *> emissives;
    for (int i = 0; i < m_nO; i++) {
      Object *o = m_objects+i;

      o->setOffsets(m_nT,m_nE);
      
      Triangle **o_emissives = o->getEmissives();
      for (int j = 0; j < o->nEmissives(); j++) {
	emissives.push_back(o_emissives[j]);
      }
      m_nT += o->nTriangles();
      m_nE += o->nEmissives();
    }
    cudaMalloc(&m_emissives,sizeof(Triangle *)*emissives.size());
    cudaMemcpy(m_emissives,emissives.data(),sizeof(Triangle *)*emissives.size(),cudaMemcpyHostToDevice);
    
    m_bvh.build(m_objects,m_nO);
  }
  
  ~Scene() {
    for (int i = 0; i < m_nO; i++) {
      (m_objects+i)->~Object();
    }
    cudaFree(m_objects);
    cudaFree(m_emissives);
  }

  __host__ __device__ void getIntersection(Ray ray, intersection_t &intersect, bool occlusion=false) {
    return m_bvh.getIntersection(ray,intersect,occlusion);
  }

  __host__ __device__ mat4F getInverseViewMatrix() {
    return m_camera.getInverseViewMatrix();
  }

  __host__ __device__ void emissives(Triangle ** &emissives) {
    emissives = m_emissives;
  }

  __host__ __device__ int nTriangles() {return m_nT;}

  __host__ __device__ int nObjects() {return m_nO;}

  __host__ __device__ int nEmissives() {return m_nE;}
 
 private:
  Triangle **m_emissives;
  Camera m_camera;
  Object *m_objects;
  BVH m_bvh;
  
  int m_nO;
  int m_nE;
  int m_nT;
};
