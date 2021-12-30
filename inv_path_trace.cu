#include "inv_scene.h"

#include <curand.h>
#include <curand_kernel.h>

__device__ void BSDF(intersection_t intersect, vecF w, vecF w_i, bool isSpecular, float n, bool isDirect, float *factors) {
    
    factors[PTYPE::DIFFUSE] = isDirect ? 1./M_PI : 1.;
    
    vecF norm = intersect.tri->getNormal(intersect.hit);
    vecF refl = -w_i + 2 * norm.dot(w_i) * norm;
    float specCoeff = (n + 2.)/2./M_PI * pow(max(refl.dot(w),0.f),n);
    factors[PTYPE::SPECULAR] = (isSpecular) ? specCoeff / P_SPEC : 0.;
}

__device__ void directLighting(Scene *scene, Image *image, vecF d, intersection_t intersect, LightEdge *lightEdges, bool isSpecular, float shininess, float prev_weight, curandState_t &state) {

    Triangle *t_curr = intersect.tri;
    Triangle **emissives; int n_e = scene->nEmissives();
    scene->emissives(emissives);
    if (!n_e) {return;}

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
    if (cos_theta < 0.) {return;}

    intersection_t i;
    scene->getIntersection(light_ray,i);
    if (!i) {return;}
    
    float cos_theta_prime = -t_emm->getNormal(i.hit).dot(toLight);
    if (cos_theta_prime < 0.) {return;}    

    Triangle *t_int = i.tri;
    if (t_int != t_emm) {return;}

    real_t *e = t_emm->material.emission;
    vecF L_o = vecF(e[0],e[1],e[2]);

    int curr = blockIdx.x * blockDim.x + threadIdx.x;
    int r = curr / SAMPLE_NUM / IM_WIDTH;
    int c = (curr / SAMPLE_NUM) % IM_WIDTH;

    // Update LightEdge
    float weight = prev_weight * cos_theta * cos_theta_prime / pow(i.t, 2.) / p_t;
    vecF light = L_o;
    vecF pixel = image->getValue(r,c);
    float factors[2]; 
    BSDF(intersect, d, toLight, isSpecular, shininess, true, factors);
    
    int src = emissives[t_emm->idxE]->idx;
    int dst = t_curr->idx;
    int le_i = dst * scene->nTriangles() + src;
    lightEdges[le_i].update(weight,pixel,light,factors);
    
    return;
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
    return isSpecular ? pow((shininess+1) * cos(theta), shininess) : 1. / M_PI;
}

__device__ int radiance(Scene *scene, int dst, Image *image, Ray &ray, LightEdge *lightEdges, int recursion, float &prev_weight, float *prev_factors, curandState_t &state) {
   	 
    intersection_t intersect;
    scene->getIntersection(ray,intersect);
    if (!intersect) {return -1;}

    Triangle *tri = intersect.tri;
    mat_t mat = tri->material;
    bool isSpecular = false;//curand_uniform(&state) < P_SPEC;
    float shininess = 0.;//curand_uniform(&state);

    int src = tri->idx;
    int nT = scene->nTriangles();
    if (dst != nT) {    
       int le_i = dst * nT + src;
       
       int curr = blockIdx.x * blockDim.x + threadIdx.x;
       int r = curr / SAMPLE_NUM / IM_WIDTH;
       int c = (curr / SAMPLE_NUM) % IM_WIDTH;

       vecF pixel = image->getValue(r,c);
       lightEdges[le_i].update(prev_weight,pixel,vecF::Zero(),prev_factors);
    }    

    // Emission
    // Not factoring in recursion 0 emitted light yet    
        
    // Direct
    directLighting(scene, image, ray.d, intersect, lightEdges, isSpecular, shininess, prev_weight, state);
    
    // Indirect
    float pRecur = curand_uniform(&state);
    if (pRecur >= p_RR) {return -1;}
    
    // Setup next recursion - the ray, the weight, and the factors
    vecF nextDir;
    float p_sample = sampleNextDir(tri->normal,isSpecular,shininess,nextDir,state);
    Ray nextRay = Ray(intersect.hit,nextDir);
    // Factors
    BSDF(intersect,ray.d,nextDir,isSpecular,shininess,false,prev_factors);
    // Weight
    prev_weight *= nextDir.dot(tri->getNormal(intersect.hit));
    prev_weight *= 1. / p_sample / p_RR / ((isSpecular) ? P_SPEC : 1 - P_SPEC);
    // Ray
    ray = nextRay;

    return src;
}

__global__ void renderSample(Scene *scene, Image *image, LightEdge lightEdges[], unsigned long long seed) {

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
    float weight = 1.;
    float factors[2];
    
    ray.transform(scene->getInverseViewMatrix());
    
    int recursion = 0; 
    int dst = scene->nTriangles();
    while (dst != -1) {
    	  dst = radiance(scene, dst, image, ray, lightEdges, recursion, weight, factors, state);
	  recursion++;
    }
}

void renderScene(Scene *scene, Image *image, LightEdge lightEdges[]) {
    // Render Scene
    unsigned long long seed = (unsigned long long) time(NULL); 
    renderSample<<<NBLOCKS,BLOCKSIZE>>>(scene,image,lightEdges,seed);
    cudaDeviceSynchronize();
    //cudaError_t err = cudaGetLastError();
    //printf("%d\n",err);
}

extern "C" {

int loadScene(int *shps, float **poss, float **oris, float **scls, char **obj_fs, char **mtl_fs, int n, void **scenePtr) {

     CameraParams_t camParams(true);

     std::vector<ObjParams_t> objParams(n);
     for (int i = 0; i < n; i++) {
     	 ObjParams_t op(shps[i],poss[i],oris[i],scls[i],obj_fs[i],mtl_fs[i]);
	 objParams[i] = op;
     }
     
     Scene *scene;
     cudaMallocManaged(&scene,sizeof(Scene));
     new(scene) Scene(camParams,objParams);
     *scenePtr = scene;     

     return scene->nTriangles();
}

void createGraph(void *scenePtr, float *graph_weights) {
    Scene *scene = (Scene *) scenePtr;
    
    Image *image;
    cudaMallocManaged(&image,sizeof(Image));
    new(image) Image("/users/bblinn/pt_inv/temp.png");

    int nT = scene->nTriangles();
    int nLE = nT * nT;
    LightEdge *lightEdges;
    cudaMallocManaged(&lightEdges,sizeof(LightEdge)*nLE);
    new(lightEdges) LightEdge[nLE];

    renderScene(scene,image,lightEdges);
    Graph graph = Graph(lightEdges,nT);
    memcpy(graph_weights,graph.p_src.data(),sizeof(float)*nLE);

    cudaFree(image);
    cudaFree(lightEdges);
    scene->~Scene();
    cudaFree(scene);
}

};
/*
int main(int argc, char argv[]) {
    
    Object *objects;
    int n = 1;
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
    
    Image *image;
    cudaMallocManaged(&image,sizeof(Image));
    new(image) Image("/users/bblinn/pt_inv/temp.png");
    
    Scene *scene;
    cudaMallocManaged(&scene,sizeof(Scene));
    new(scene) Scene(camera,objects,n);

    LightEdge *lightEdges;
    int nT = scene->nTriangles();
    int nLE = nT * nT;
    cudaMallocManaged(&lightEdges,sizeof(LightEdge)*nLE);
    new(lightEdges) LightEdge[nLE];

    renderScene(scene,image,lightEdges);
    std::vector<ProcessedEdges> procEdges = processEdges(scene,lightEdges);

    cudaFree(lightEdges);
    cudaFree(objects);
    cudaFree(camera);
    cudaFree(scene);
}*/