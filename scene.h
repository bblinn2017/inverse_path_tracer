#include "bvh.h"

class Camera {
 public:
  __host__ __device__ Camera(vecF p,vecF d,vecF u,float ha,float ar) {
    m_position = p;
    m_direction = d;
    m_up = u;
    m_heightAngle = M_PI * ha / 360.f;
    m_aspectRatio = ar;
  }

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

  __host__ __device__ Scene(Camera * camera, Object *objects, int n) {
    m_camera = camera;
    m_objects = objects;
    m_n = n;
    
    cudaMallocManaged(&m_bvh,sizeof(BVH));
    *m_bvh = BVH(objects,n);
  }
  
  __host__ __device__ ~Scene() {
  }

  __host__ __device__ void getIntersection(Ray ray, intersection_t &intersect, bool occlusion=false) {
    return m_bvh->getIntersection(ray,intersect,occlusion);
  }

  __host__ __device__ mat4F getInverseViewMatrix() {
    return m_camera->getInverseViewMatrix();
  }
  
 private:
  Triangle *m_emissives;
  Camera *m_camera;
  Object *m_objects;
  int m_n;
  BVH *m_bvh; 
};
