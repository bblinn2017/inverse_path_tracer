#include "scene.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

enum PTYPE {DIFFUSE=0,SPECULAR=1};

class Image {
 public:
  Image(const char *filename) {
    cudaMallocManaged(&m_values,sizeof(u_int8_t)*IM_WIDTH*IM_HEIGHT*3);

    int width, height, channels;
    width = 0; height = 0; channels = 0;
    stbi_uc *values = stbi_load(filename, &width, &height, &channels, 3);
    cudaMemcpy(m_values,values,sizeof(u_int8_t)*IM_WIDTH*IM_HEIGHT*3,cudaMemcpyHostToDevice);
    //assert(width == IM_WIDTH && height == IM_HEIGHT && channels == 3);
  }

  __host__ __device__ vecF getValue(int r, int c) {
    int idx = (r * IM_WIDTH + c) * 3;
    return vecF(m_values[idx],m_values[idx+1],m_values[idx+2]) / 255.;
  }
 private:
  u_int8_t *m_values;
};

struct LightEdge {
  
  int n;
  float w_sum;
  float ff_sum;
  float pixel_sum[3];
  float light_sum[3];
  float factors_sum[2];
  
  __device__ void update(float w, float ff, vecF pixel, vecF light, float *factors) {
    atomicAdd_system(&n,1);
    atomicAdd_system(&w_sum,w);
    atomicAdd_system(&ff_sum,w*ff);
    for (int i = 0; i < 3; i++) {
      atomicAdd_system(&(pixel_sum[i]),w*pixel[i]);
      atomicAdd_system(&(light_sum[i]),w*light[i]);
    }
    for (int i = 0; i < 2; i++) {atomicAdd_system(&(factors_sum[i]),w*factors[i]);}
  }

  __host__ __device__ void normalize() {
    float weight = (w_sum) ? w_sum : 1.;
    ff_sum /= weight;
    for (int i = 0; i < 3; i++) {
      pixel_sum[i] /= weight;
      light_sum[i] /= weight;
    }
    for (int i = 0; i < 2; i++) {factors_sum[i] /= weight;}
  }
};

struct TriData {
  
  std::vector<weights> e_weights;

  TriData(std::vector<weights> ws) {
    e_weights = ws;
  }
  
}
