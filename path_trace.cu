#include "scene.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

typedef Eigen::Vector3f vecF;
typedef Eigen::Vector3i vecI;
typedef Eigen::Matrix4f mat4F;
typedef Eigen::Matrix3f mat3F;

#define EYE vecF(0,0,0)
#define LOOK vecF(0,0,1)
#define UP vecF(0,1,0)
#define HA 90.f
#define AR 1.f
#define IM_WIDTH 1
#define IM_HEIGHT 1
#define SAMPLE_NUM 1
#define p_RR 0.7f
#define BLOCKSIZE 512
#define NBLOCKS IM_WIDTH * IM_HEIGHT * SAMPLE_NUM / BLOCKSIZE + 1

__host__ __device__ vecF radiance(Scene *scene, Ray ray, int recursion) {
    intersection_t intersect;
    scene->getIntersection(ray,intersect);
    return vecF(1.,1.,1.) / intersect.t;
}

__global__ void renderSample(Scene *scene, vecF values[]) {
    int curr = blockIdx.x * blockDim.x + threadIdx.x;
    if (curr >= IM_WIDTH * IM_HEIGHT * SAMPLE_NUM) {return;}
    
    int r = curr / SAMPLE_NUM / IM_WIDTH;
    int c = (curr / SAMPLE_NUM) % IM_WIDTH; 

    vecF p = vecF::Zero();
    //vecF d = vecF(2 * c / IM_WIDTH - 1, 1 - 2 * r / IM_HEIGHT, 1);
    vecF d = vecF(0,0,1.);
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
    int n = 1;
    cudaMallocManaged(&objects,sizeof(Object)*n);
    for (int i = 0; i < n; i++) {
	objects[i] = Object(shape_t::Cube);
    }

    for (int i = 0; i < objects[0].m_mesh->m_nV; i++) {
      vecF v = objects[0].m_mesh->m_vertices[i];
      printf("%f %f %f\n",v[0],v[1],v[2]);
    }
    printf("\n");

    Camera *camera;
    cudaMallocManaged(&camera,sizeof(Camera));
    *camera = Camera(EYE,LOOK,UP,HA,AR);

    Scene *scene;
    cudaMallocManaged(&scene,sizeof(Scene));
    *scene = Scene(camera,objects,n);
    
    vecF values[IM_WIDTH*IM_HEIGHT*SAMPLE_NUM];
    renderScene(scene,values);

    unsigned char img[IM_WIDTH*IM_HEIGHT*3];
    for (int i = 0; i < IM_WIDTH*IM_HEIGHT; i++) {
    	vecF curr;
    	for (int j = 0; j < SAMPLE_NUM; j++) {
	    curr += values[i*SAMPLE_NUM + j] / (float) SAMPLE_NUM;
	}
	for (int j = 0; j < 3; j++) {
	    float val = max(min(curr[j],1.),0.);
	    img[i*3+j] = (uint8_t)(255 * val);
	}
    }
    stbi_write_jpg("temp.jpg", IM_WIDTH, IM_HEIGHT, 3, img, 100);

    cudaFree(objects);
    cudaFree(camera);
    cudaFree(scene);
}