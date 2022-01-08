#include "scene.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

#define P_SPEC 0.

enum PTYPE {DIFFUSE=0,SPECULAR=1};

struct Edge {
  
  int n;
  float w_sum;
  float pixel_sum[2][3];
  float light_sum[2][3];
  float factors_sum[2];

  Edge() {
    n = 0;
    w_sum = 0.;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {pixel_sum[i][j] = 0.; light_sum[i][j] = 0.;}
      factors_sum[i] = 0.;
    }
  }

  __device__ void update(float w, vecF pixel, vecF light, float *factors) {
    atomicAdd_system(&n,1);
    atomicAdd_system(&w_sum,w);
    for (int j = 0; j < 2; j++) {
      for (int i = 0; i < 3; i++) {
	atomicAdd_system(&(pixel_sum[j][i]),w*factors[j]*pixel[i]);
	atomicAdd_system(&(light_sum[j][i]),w*factors[j]*light[i]);
      }
    }
    for (int i = 0; i < 2; i++) atomicAdd_system(&(factors_sum[i]),w*factors[i]);
  }

  void normalize() {
    w_sum = log(w_sum+1);
    for (int j = 0; j < 2; j++) {
      float weight = (factors_sum[j]) ? factors_sum[j] : 1.;
      for (int i = 0; i < 3; i++) {
	pixel_sum[j][i] /= weight;
	light_sum[j][i] /= weight;
      }
    }
  }
};

class DataWrapper {
 public:
 DataWrapper(const char *filename, int nT) : m_nT(nT) 
  {
    int width, height, channels;
    width = 0; height = 0; channels = 0;
    stbi_uc *vals = stbi_load(filename, &width, &height, &channels, 3);
    cudaMemcpy(m_values,vals,sizeof(u_int8_t)*IM_WIDTH*IM_HEIGHT*3,cudaMemcpyHostToDevice);

    int nE = m_nT * m_nT;
    cudaMallocManaged(&m_edges,sizeof(Edge)*nE);
    new(m_edges) Edge[nE];
		 
    cudaMallocManaged(&m_eyeEdges,sizeof(Edge)*m_nT);
    new(m_eyeEdges) Edge[m_nT];
  }

  ~DataWrapper() {
    for (int i = 0; i < m_nT * m_nT; i++) (m_edges + i)->~Edge();
    for (int i = 0; i < m_nT; i++) (m_eyeEdges + i)->~Edge();
    cudaFree(m_edges);
    cudaFree(m_eyeEdges);
  }

  __host__ __device__ vecF getPixel(int r, int c) {
    int idx = (r * IM_WIDTH + c) * 3;
    return vecF(m_values[idx],m_values[idx+1],m_values[idx+2]) / 255.;
  }

  __host__ __device__ Edge * get(int dst, int src) {
    return (dst != m_nT) ? m_edges + dst * m_nT + src : m_eyeEdges + src;
  }

  __device__ void update(int dst, int src, float w, vecF pixel, vecF light, float *factors) {
    get(dst,src)->update(w,pixel,light,factors);
  }

  std::vector<float> compress() {
    std::vector<float> weights;
    std::vector<float> pixels;
    std::vector<float> lights;
    
    for (int dst = 0; dst <= m_nT; dst++) {
      float w_total = 0.;
      for (int src = 0; src < m_nT; src++) {
        Edge *e = get(dst,src);
	e->normalize();
	float val = e->w_sum;
	w_total += val;
	for (int i = 0; i < 3; i++) {
	  pixels.push_back(e->pixel_sum[DIFFUSE][i]);
	  lights.push_back(e->light_sum[DIFFUSE][i]);
	}
      }
      for (int src = 0; src < m_nT; src++) {
	float val = w_total ? get(dst,src)->w_sum / w_total : 0.;
	weights.push_back(val);
      }
    }
    
    std::vector<float> data;
    data.insert(data.end(), weights.begin(), weights.end());
    data.insert(data.end(), pixels.begin(), pixels.end());
    data.insert(data.end(), lights.begin(), lights.end());
    return data;
  }

 private:
  int m_nT;
  u_int8_t m_values[IM_HEIGHT*IM_WIDTH*3];
  
  Edge *m_edges;
  Edge *m_eyeEdges;
};
