#include "scene.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

#include <curand.h>
#include <curand_kernel.h>

__device__ mat3F BSDF(intersection_t intersect, vecF w, vecF w_i, bool isDirect) {
    
    mat_t *mat = intersect.tri->material;
    
    vecF diffuse = vecF(mat->diffuse[0],mat->diffuse[1],mat->diffuse[2]);
    if (!isDirect) {
       diffuse /= M_PI;
    }

    float n = mat->shininess;
    vecF norm = intersect.tri->getNormal(intersect.hit);
    vecF refl = -w_i + 2 * norm.dot(w_i) * norm;
    float coeff = (n+2)/2./M_PI * max(pow(refl.dot(w),n),0.f);
    vecF specular = vecF(mat->specular[0],mat->specular[1],mat->specular[2]) * coeff;
        
    mat3F T = mat3F::Zero();
    for (int i = 0; i < 3; i++) {T(i,i) = diffuse[i] + specular[i];}
    return T;
}

__device__ vecF directLighting(Scene *scene, vecF d, intersection_t intersect, curandState_t &state) {

    Triangle *t_curr = intersect.tri;
    int n_e = scene->nEmissives();
    if (!n_e) {return vecF::Zero();}
    Triangle **emissives;
    scene->emissives(emissives);

    float area_sum = 0.;
    for (int i = 0; i < n_e; i++) {
    	area_sum += emissives[i]->area;
    }
    float p = curand_uniform(&state);
    float p_curr = 0.;
    int idx = 0;
    for (int i = 0; i < n_e; i++) {
    	  p_curr += emissives[idx]->area / area_sum;
	  if (p_curr >= p) {break;}
	  idx++;
    }
    float p_t = emissives[idx]->area / area_sum;
    
    Triangle *t_emm = emissives[idx];
    
    float r1 = curand_uniform(&state);
    float r2 = curand_uniform(&state);
    
    vecF v1,v2,v3;
    v1 = t_emm->vertices[0]; v2 = t_emm->vertices[1]; v3 = t_emm->vertices[2];
    vecF emm_point = (1 - pow(r1,0.5)) * v1 +
    	 (pow(r1,0.5) * (1 - r2)) * v2 + 
	 (r2 * pow(r1,0.5)) * v3;

    vecF curr_int = intersect.hit;
    vecF toLight = emm_point - curr_int;
    toLight.normalize();

    Ray light_ray = Ray(curr_int,toLight);
    
    float cos_theta = t_curr->getNormal(intersect.hit).dot(toLight);
    if (cos_theta < 0.) {return vecF::Zero();}

    intersection_t i;
    scene->getIntersection(light_ray,i);
    if (!i) {return vecF::Zero();}
    
    float cos_theta_prime = -t_emm->getNormal(i.hit).dot(toLight);
    if (cos_theta_prime < 0.) {return vecF::Zero();}    

    Triangle *t_int = i.tri;
    if (t_int != t_emm) {return vecF::Zero();}

    real_t *e = t_emm->material->emission;
    vecF L_o = vecF(e[0],e[1],e[2]);
    
    L_o *= cos_theta * cos_theta_prime / pow(i.t, 2.) / p_t;
    L_o = BSDF(intersect, d, toLight, true) * L_o;
    return L_o;
}

__device__ float sampleNextDir(vecF normDir, bool isSpecular, float shininess, vecF &nextDir, curandState_t &state) {
    
    float phi = 2 * M_PI * curand_uniform(&state);
    float theta = acos(pow(curand_uniform(&state),isSpecular ? 1./(shininess+1.) : .5));
    
    vecF hemiDir = vecF(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));
    
    mat3F R;
    if (normDir[2] == -1) {
      R = -mat3F::Identity();
    } else {
      R = Eigen::Quaternionf::FromTwoVectors(vecF(0,0,1),normDir).matrix();
    }
    nextDir = R * hemiDir;
    nextDir.normalize();

    vecF output = R * vecF(0,0,1);
    return isSpecular ? pow((shininess+1) * cos(theta), shininess) : 1 / M_PI;
}

__device__ bool radiance(Scene *scene, Ray &ray, vecF &L_e, vecF &L_d, mat3F &multiplier, int recursion, curandState_t &state) {
    	   
    intersection_t intersect;
    scene->getIntersection(ray,intersect);    
    if (!intersect) {return false;}
    
    Triangle *tri = intersect.tri;
    mat_t *mat = tri->material;
    
    // Emission
    if (recursion == 0) {
       for (int i = 0; i < 3; i++) {L_e[i] = mat->emission[i];}
    }

    // Direct
    L_d = directLighting(scene, ray.d, intersect, state);
    
    // Indirect
    float pRecur = curand_uniform(&state);
    if (pRecur >= p_RR) {return false;}
    bool isSpecular = (mat->specular[0] || mat->specular[1] || mat->specular[2]) && mat->shininess;
    vecF nextDir;
    float p_sample = sampleNextDir(tri->normal,isSpecular,mat->shininess,nextDir,state);
    Ray nextRay = Ray(intersect.hit,nextDir);
    mat3F bsdf = BSDF(intersect,ray.d,nextDir,false);
    float coeff = nextDir.dot(tri->getNormal(intersect.hit)) / p_sample / p_RR;    

    // Setup next recursion - the multiplier and the ray
    multiplier = multiplier * bsdf * coeff;
    ray = nextRay;
    
    return true;
}

__global__ void renderSample(Scene *scene, vecF values[], unsigned long long seed) {

    int curr = blockIdx.x * blockDim.x + threadIdx.x;
    if (curr >= IM_WIDTH * IM_HEIGHT * SAMPLE_NUM) {return;}

    seed += curr;
    curandState_t state;
    curand_init(seed,0,0,&state);
    
    int r = curr / SAMPLE_NUM / IM_WIDTH;
    int c = (curr / SAMPLE_NUM) % IM_WIDTH;

    float x = 2.f * ((float) c + curand_uniform(&state)) / (float) IM_WIDTH - 1.f;
    float y = 1.f - 2.f * ((float) r + curand_uniform(&state)) / (float) IM_HEIGHT;

    vecF p = vecF::Zero();
    vecF d = vecF(x, y, 1);
    d.normalize();
    Ray ray = Ray(p,d);
    
    ray.transform(scene->getInverseViewMatrix());
    
    vecF L = vecF::Zero();

    vecF L_e = vecF::Zero();
    vecF L_d = vecF::Zero();
    mat3F multiplier = mat3F::Identity();

    int recursion = 0;
    vecF L_o; mat3F prevMultiplier = multiplier;
    bool recur = true;    
    while (recur) {
    	  recur = radiance(scene, ray, L_e, L_d, multiplier, recursion, state);
	  L_o = L_e + L_d;
	  L += prevMultiplier * L_o;
	  prevMultiplier = multiplier;
	  recursion++;
    }

    values[curr] = L;
}

void renderScene(Scene *scene, vecF values[]) {
    // Render Scene
    unsigned long long seed = (unsigned long long) time(NULL); 
    renderSample<<<NBLOCKS,BLOCKSIZE>>>(scene,values,seed);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    printf("%d\n",err);
}

int main(int argc, char argv[]) {

    Object *objects;
    int n = 2;
    cudaMallocManaged(&objects,sizeof(Object)*n);
    new(&(objects[0])) Object(shape_type_t::Cornell,
				vecF(0,0,4),
                  		vecF(0,0,0),
                  		vecF(2,2,2));
    new(&(objects[1])) Object(shape_type_t::Cube,
				vecF(0,-1.5,4),
				vecF(0,0,0),
				vecF(1,1,1)
    );
    
    Camera *camera;
    cudaMallocManaged(&camera,sizeof(Camera));
    new(camera) Camera(EYE,LOOK,UP,HA,AR);
    
    Scene *scene;
    cudaMallocManaged(&scene,sizeof(Scene));
    new(scene) Scene(camera,objects,n);
    
    vecF *gpuValues;
    cudaMallocManaged(&gpuValues,sizeof(vecF)*IM_WIDTH*IM_HEIGHT*SAMPLE_NUM);
    renderScene(scene,gpuValues);
    vecF values[IM_WIDTH*IM_HEIGHT*SAMPLE_NUM];
    cudaMemcpy(values,gpuValues,sizeof(vecF)*IM_WIDTH*IM_HEIGHT*SAMPLE_NUM,cudaMemcpyDeviceToHost);

    unsigned char img[IM_WIDTH*IM_HEIGHT][3];
    for (int i = 0; i < IM_WIDTH*IM_HEIGHT; i++) {
    	vecF curr = vecF::Zero();
    	for (int j = 0; j < SAMPLE_NUM; j++) {
	    curr += values[i*SAMPLE_NUM + j] / (float) SAMPLE_NUM;
	}
	for (int j = 0; j < 3; j++) {
	    uint8_t val = (uint8_t)(255.f * curr[j] / (1 + curr[j]));
	    img[i][j] = val;
	}
    }
    stbi_write_png("temp.png", IM_WIDTH, IM_HEIGHT, 3, img, IM_WIDTH*3);

    cudaFree(gpuValues);
    cudaFree(objects);
    cudaFree(camera);
    cudaFree(scene);
}