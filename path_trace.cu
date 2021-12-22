#include "scene.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

typedef Eigen::Vector3f vecF;
typedef Eigen::Vector3i vecI;
typedef Eigen::Matrix4f mat4F;
typedef Eigen::Matrix3f mat3F;

const vecF EYE = vecF(0.,0.,0.);
const vecF LOOK = vecF(0.,0.,1.);
const vecF UP = vecF(0.,1.,0.);
const float HA = 90.f;
const float AR = 1.f;
#define IM_WIDTH 100
#define IM_HEIGHT 100
#define SAMPLE_NUM 1
#define p_RR 0.7f
#define BLOCKSIZE 512
#define NBLOCKS IM_WIDTH * IM_HEIGHT * SAMPLE_NUM / BLOCKSIZE + 1

__host__ __device__ vecF directLighting(Scene *scene, vecF d, intersection_t intersect) {
    
    Triangle *t_curr = intersection.tri;

}

__host__ __device__ vecF radiance(Scene *scene, Ray ray, int recursion) {
    intersection_t intersect;
    scene->getIntersection(ray,intersect);
    
    if (!intersection.hit) {return vecF(0,0,0);}

    Triange *tri = intersection.tri;
    mat_t mat = tri->material;

    // Emission
    vecF L_e;
    if (recursion == 0) {
       for (int i = 0; i < 3; i++) {L_e[i] = mat.emission[i];}
    }

    // Direct
    vecF L_d = directLighting(scene, ray.d, intersect);
    
}

__global__ void renderSample(Scene *scene, vecF values[]) {
    int curr = blockIdx.x * blockDim.x + threadIdx.x;
    if (curr >= IM_WIDTH * IM_HEIGHT * SAMPLE_NUM) {return;}
    
    int r = curr / SAMPLE_NUM / IM_WIDTH;
    int c = (curr / SAMPLE_NUM) % IM_WIDTH;

    vecF p = vecF::Zero();
    vecF d = vecF(2 * ((float) c + 0.5) / (float) IM_WIDTH - 1., 1. - 2 * ((float) r + 0.5) / (float) IM_HEIGHT, 1);
    d.normalize();
    Ray ray = Ray(p,d);
    
    ray.transform(scene->getInverseViewMatrix());
    values[curr] = radiance(scene,ray,0);
}

void renderScene(Scene *scene, vecF values[]) {
    // Render Scene
    renderSample<<<NBLOCKS,BLOCKSIZE>>>(scene,values);
    cudaDeviceSynchronize();
}

int main(int argc, char argv[]) {

    Object *objects;
    int n = 2;
    cudaMallocManaged(&objects,sizeof(Object)*n);
    objects[0] = Object(shape_t::Other,
			vecF(0,0,1.),
			vecF(0,0,0),
			vecF(4,4,4),
			"/users/bblinn/pt_inv/shapes/scene.obj");
    for (int i = 1; i < n; i++) {
	objects[i] = Object(shape_t::Cube);
    }
    
    Camera *camera;
    cudaMallocManaged(&camera,sizeof(Camera));
    vecF eye = vecF::Zero();
    vecF look = vecF(0,0,1);
    vecF up = vecF(0,1,0);
    float heightAngle = 90.f;
    float aspectRatio = 1.f;
    camera = new(camera) Camera(eye,look,up,heightAngle,aspectRatio);
    
    Scene *scene;
    cudaMallocManaged(&scene,sizeof(Scene));
    scene = new(scene) Scene(camera,objects,n);
    
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
	    float val = max(min(curr[j],1.),0.);
	    img[i][j] = (uint8_t)(255.f * val);
	}
    }
    stbi_write_jpg("temp.jpg", IM_WIDTH, IM_HEIGHT, 3, img, 100);

    cudaFree(gpuValues);
    cudaFree(objects);
    //delete camera;
    cudaFree(camera);
    //delete scene;
    cudaFree(scene);
}