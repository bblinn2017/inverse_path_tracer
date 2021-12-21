#include "bvh.h"

class Camera {
 public:

  __host__ __device__ Camera(vecF p, vecF d, vecF u, float ha, float ar) {
    m_position = p;
    m_direction = d / d.norm();
    m_up = u / u.norm();
    m_heightAngle = M_PI * ha / 360.f;
    m_aspectRatio = ar;
  }

  __host__ __device__  Camera() {}

  __host__ __device__ mat4F getViewMatrix() {

    vecF f = m_direction;
    vecF s = f.cross(m_up);
    vecF u = s.cross(f);

    mat4F vM;
    vM << s,-s.dot(m_position),
      u,-u.dot(m_position),
      f,-f.dot(m_position),
      0,0,0,1;

    return vM;
  }

  __host__ __device__ mat4F getScaleMatrix() {
    
    Eigen::Affine3f sM = Eigen::Affine3f::Identity();
    sM.scale(vecF(tan(m_heightAngle),tan(m_heightAngle*m_aspectRatio),1.));
    return sM.matrix();
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

  __host__ __device__ Scene(Camera *camera, Object *objects, int n) {
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

  __host__ __device__ mat4F getInverseViewMatrix() {return m_camera->getInverseViewMatrix();}
  
 private:
  Camera *m_camera;
  Object *m_objects;
  int m_n;
  BVH *m_bvh; 
};
