#include "scene.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

#define P_SPEC 0.5
#define P_MIN 0.01

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
  float pixel_sum[2][3];
  float light_sum[2][3];
  float factors_sum[2];

  __device__ void update(float w, vecF pixel, vecF light, float *factors) {
    atomicAdd_system(&n,1);
    atomicAdd_system(&w_sum,w);
    for (int j = 0; j < 2; j++) {
      for (int i = 0; i < 3; i++) {
	atomicAdd_system(&(pixel_sum[j][i]),w*factors[j]*pixel[i]);
	atomicAdd_system(&(light_sum[j][i]),w*factors[j]*light[i]);
      }
    }
    for (int i = 0; i < 2; i++) {atomicAdd_system(&(factors_sum[i]),w*factors[i]);}
  }

  __host__ __device__ void normalize() {
    float weight = (w_sum) ? w_sum : 1.;
    for (int j = 0; j < 2; j++) {
      for (int i = 0; i < 3; i++) {
	pixel_sum[j][i] /= factors_sum[j];
	light_sum[j][i] /= factors_sum[j];
      }
    }
  }
};

struct ProcessedEdges {

  int dst;
  std::vector<float> p_src;
  std::vector<vecF> d_light;
  std::vector<vecF> d_pixel;
  std::vector<vecF> s_light;
  std::vector<vecF> s_pixel;
  
  ProcessedEdges() {}

  ProcessedEdges(int d, std::vector<LightEdge> les) {
    float w_total = 0.;
    for (int i = 0; i < les.size(); i++) {
      if (!(i == dst)) {w_total+= les[i].w_sum;}
    }
    w_total = (w_total) ? w_total : 1.;
    for (int i = 0; i < les.size(); i++) {
      LightEdge le = les[i];
      le.normalize();
      float p_s = float(le.w_sum) / w_total;
      
      p_src.push_back(p_s);
      d_light.push_back(vecF(le.light_sum[DIFFUSE][0],le.light_sum[DIFFUSE][1],le.light_sum[DIFFUSE][2]));
      d_pixel.push_back(vecF(le.pixel_sum[DIFFUSE][0],le.pixel_sum[DIFFUSE][1],le.pixel_sum[DIFFUSE][2]));
      s_light.push_back(vecF(le.light_sum[SPECULAR][0],le.light_sum[SPECULAR][1],le.light_sum[SPECULAR][2]));
      s_pixel.push_back(vecF(le.pixel_sum[SPECULAR][0],le.pixel_sum[SPECULAR][1],le.pixel_sum[SPECULAR][2]));
    }
    dst = d;
  }

  friend std::ostream& operator<<(std::ostream & str, const ProcessedEdges& pe) {
    for (int i = 0; i < pe.p_src.size(); i++) {
      if (pe.p_src[i] < P_MIN || i == pe.dst) {continue;}
      
      str << pe.dst << ",";
      str << i << ",";
      str << pe.p_src[i] << ",";
      str << pe.d_light[i][0] << "," << pe.d_light[i][1] << "," << pe.d_light[i][2] << ",";
      str << pe.d_pixel[i][0] << "," << pe.d_pixel[i][1] << "," << pe.d_pixel[i][2] << ",";
      str << pe.s_light[i][0] << "," << pe.s_light[i][1] << "," << pe.s_light[i][2] << ",";
      str << pe.s_pixel[i][0] << "," << pe.s_pixel[i][1] << "," << pe.s_pixel[i][2] << std::endl;
    }
    return str;
  }
};
