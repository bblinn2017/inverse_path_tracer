#include "scene_basics.h"

#define UNTOUCHED 0xffffffff
#define TOUCHED_TWICE 0xfffffffd

__host__ __device__ struct flatNode_t {
  bbox_t bbox;
  int start;
  int nPrims;
  int rightOffset;
};

struct buildEntry_t {
  char parent;
  int start;
  int end;
};

__host__ __device__ struct traversal_t {
  int i;
  float mint;
};

class BVH {
 public:
  __host__ __device__ BVH(int leafSize=4) {
    m_leafSize = leafSize;

    m_nNodes = 0;
    m_nLeafs = 0;
  }

  ~BVH() {
    cudaFree(m_flatTree);
  }

  __host__ __device__ void getIntersection(Ray ray, intersection_t &intersection, bool occlusion) {
    
    intersection.t = INFINITY;

    if (m_nNodes == 0) {return;}
    
    traversal_t todo[64];
    int stackptr = 0;
    
    todo[stackptr].i = 0;
    todo[stackptr].mint = INFINITY;

    int ni, closer, other;
    float near, min0, max0, min1, max1, maxCloser, maxOther;
    flatNode_t node, n0, n1;
    Object *obj;
    bool hitc0, hitc1, less;

    while (stackptr >= 0) {
      ni = todo[stackptr].i;
      near = todo[stackptr].mint;
      stackptr -= 1;
      node = m_flatTree[ni];

      if (near > intersection.t) {
        continue;
      }
      
      if (!node.rightOffset) {
        for (int o = 0; o < node.nPrims; o++) {
          obj = m_objects + node.start + o;
	  intersection_t curr;
          obj->getIntersection(ray,curr);
	  
          if (curr) {

            if (occlusion) {return;}

            if (curr.t < intersection.t) {intersection = curr;}
          }
        }
      } else {
        n0 = m_flatTree[ni+1];
        n1 = m_flatTree[ni+node.rightOffset];
        hitc0 = n0.bbox.intersect(ray,min0,max0);
        hitc1 = n1.bbox.intersect(ray,min1,max1);

        if (hitc0 && hitc1) {

          less = max0 < max1;
          closer = less ? ni+1 : ni+node.rightOffset;
          maxCloser = less ? max0 : max1;
          other = (!less) ? ni+1 : ni+node.rightOffset;
          maxOther = (!less) ? max0 : max1;

          stackptr += 1;
          todo[stackptr] = {.i=other, .mint=maxOther};
          stackptr += 1;
          todo[stackptr] = {.i=closer, .mint=maxCloser};
        } else if (hitc0) {

          stackptr += 1;
          todo[stackptr] = {.i=ni+1, max0};
        } else if (hitc1) {

          stackptr += 1;
          todo[stackptr] = {.i=ni+node.rightOffset, max1};
        }
      }
    }
  }

  void build(Object *objects, int n) {
    m_nO = n;
    m_objects = objects;
    if (!m_nO) {
      return;
    }
    
    std::vector<buildEntry_t> todo;
    todo.reserve(128);
    
    int stackptr = 0;

    todo[stackptr].start = 0;
    todo[stackptr].end = m_nO;
    todo[stackptr].parent = 0xfffffffc;
    stackptr += 1;
 
    flatNode_t node;
    std::vector<flatNode_t> buildnodes = std::vector<flatNode_t>();

    buildEntry_t bnode;
    int start, end, nPrims, split_dim, mid;
    float split_coord;
    bbox_t bb, bc;
    while (stackptr > 0) {
      stackptr -= 1;
      bnode = todo[stackptr];
      start = bnode.start;
      end = bnode.end;
      nPrims = end - start;
      
      m_nNodes += 1;
      node.start = start;
      node.nPrims = nPrims;
      node.rightOffset = UNTOUCHED;
      
      bb = m_objects[start].getBBox();
      bc = {};
      
      bc.setP(m_objects[start].getPosition());
      for (int p = start+1; p < end; p++) {
	bb.expandToInclude(m_objects[p].getBBox());
	bc.expandToInclude(m_objects[p].getPosition());
      }

      node.bbox = bb;

      if (nPrims <= m_leafSize) {
	node.rightOffset = 0;
	m_nLeafs += 1;
      }

      buildnodes.push_back(node);
      
      if (bnode.parent != 0xfffffffc) {
	buildnodes[bnode.parent].rightOffset -= 1;

	if (buildnodes[bnode.parent].rightOffset == TOUCHED_TWICE) {
	  buildnodes[bnode.parent].rightOffset = m_nNodes - 1 - bnode.parent;
	}
      }

      if (!node.rightOffset) {
	continue;
      }

      split_dim = bc.maxDimension();
      split_coord = (bc.min[split_dim] + bc.max[split_dim]);

      mid = start;
      for (int i = start; i < end; i++) {
	if (m_objects[i].getPosition()[split_dim] < split_coord) {
	  Object temp = m_objects[mid];
	  m_objects[mid] = m_objects[i];
	  m_objects[i] = temp;
	  mid += 1;
	}
      }

      if (mid == start || mid == end) {
	mid = start + ((int) (end - start)/2.);
      }

      todo[stackptr].start = mid;
      todo[stackptr].end = end;
      todo[stackptr].parent = m_nNodes - 1;
      stackptr += 1;

      todo[stackptr].start = start;
      todo[stackptr].end = mid;
      todo[stackptr].parent = m_nNodes - 1;
      stackptr += 1;
    }
    
    cudaMallocManaged(&m_flatTree,sizeof(flatNode_t)*m_nNodes);
    cudaMemcpy(m_flatTree,buildnodes.data(),sizeof(flatNode_t)*m_nNodes,cudaMemcpyHostToDevice);
  }

 private:
  Object *m_objects;
  int m_leafSize;

  int m_nNodes;
  int m_nLeafs;

  int m_nO;

  flatNode_t *m_flatTree;

};
