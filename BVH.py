import numpy
from scene_basics import *
from scipy.spatial.transform import Rotation
from copy import deepcopy
import warnings

class BBox:

    def __init__(self):

        self.minimum = None
        self.maximum = None
        self.extent = None

    def setP(self,p):

        self.minimum = p
        self.maximum = p
        self.extent = self.minimum - self.maximum

    def expandToInclude(self,o):

        if isinstance(o,np.ndarray):
            self.minimum = np.minimum(self.minimum,o)
            self.maximum = np.maximum(self.maximum,o)
        elif isinstance(o,BBox):
            self.minimum = np.minimum(self.minimum,o.minimum)
            self.maximum = np.maximum(self.maximum,o.maximum)
        self.extent = self.maximum - self.minimum

    def maxDimension(self):

        return np.argmax(self.extent)

    def surfaceArea(self):

        return 2 * np.sum(np.prod(self.extent[[0,1],[1,2],[2,0]],axis=-1))

    def intersect(self,ray):

        warnings.filterwarnings("ignore")
        l1 = (self.minimum - ray.p) / ray.d
        l2 = (self.maximum - ray.p) / ray.d

        l1nan = np.isnan(l1).nonzero()[0]
        l1sign = np.sign((self.minimum - ray.p)[l1nan]) + 0.5
        l1[l1nan] = l1sign * np.PINF
        l2nan = np.isnan(l2).nonzero()[0]
        l2sign = np.sign((self.maximum - ray.p)[l2nan]) + 0.5
        l2[l2nan] = l2sign * np.PINF

        lmax = np.maximum(l1,l2)
        lmin = np.minimum(l1,l2)

        if np.all(lmin < 0) and np.all(lmax > lmin):
            return lmin.max(), lmax.min()
        return None


class BVHFlatNode:

    def __init__(self):

        self.bbox = None
        self.start = None
        self.nPrims = None
        self.rightOffset = None

class BVHBuildEntry:

    def __init__(self):

        self.parent = None
        self.start = None
        self.end = None

class BVHTraversal:

    def __init__(self,i=None,mint=None):

        self.i = i
        self.mint = mint

UNTOUCHED = 0xffffffff
TOUCHED_TWICE = 0xfffffffd

class BVH:

    def	__init__(self,objects,leaf_size=4):

        self.objects = objects
        self.leaf_size = leaf_size

        self.nNodes = 0
        self.nLeafs = 0

        self.build()

    def getBBox(self,obj):

        box_min = obj.mesh.vertices.min(0)
        box_max = obj.mesh.vertices.max(0)

        bbox = BBox()
        bbox.minimum = box_min
        bbox.maximum = box_max
        bbox.extent = box_max - box_min

        return bbox

    def	build(self):

        if len(self.objects) == 0:
            self.flatTree = []
            return

        todo = [BVHBuildEntry() for i in range(128)]
        stackptr = 0

        todo[stackptr].start = 0
        todo[stackptr].end = len(self.objects)
        todo[stackptr].parent = 0xfffffffc
        stackptr += 1

        node = BVHFlatNode()
        buildnodes = []

        while stackptr > 0:

            stackptr -= 1
            bnode = todo[stackptr]
            start = bnode.start
            end = bnode.end
            nPrims = end - start

            self.nNodes += 1
            node.start = start
            node.nPrims = nPrims
            node.rightOffset = UNTOUCHED

            bb = self.getBBox(self.objects[start])
            bc = BBox()

            bc.setP(self.objects[start].getCentroid())
            for p in range(start+1,end):
                bb.expandToInclude(self.getBBox(self.objects[p]))
                bc.expandToInclude(self.objects[p].mesh.position)

            node.bbox = bb

            if nPrims <= self.leaf_size:
                node.rightOffset = 0
                self.nLeafs += 1

            buildnodes.append(deepcopy(node))

            if bnode.parent != 0xfffffffc:
                buildnodes[bnode.parent].rightOffset -= 1

                if buildnodes[bnode.parent].rightOffset == TOUCHED_TWICE:
                    buildnodes[bnode.parent].rightOffset = self.nNodes - 1 - bnode.parent

            if node.rightOffset == 0:
                continue

            split_dim = bc.maxDimension()

            split_coord = .5 * (bc.minimum[split_dim] + bc.maximum[split_dim])

            mid = start
            for i in range(start,end):
                if self.objects[i].mesh.position[split_dim] < split_coord:
                    temp = self.objects[mid]
                    self.objects[mid] = self.objects[i]
                    self.objects[i] = temp
                    mid += 1

            if mid == start or mid == end:
                mid = start + (end - start) // 2

            todo[stackptr].start = mid
            todo[stackptr].end = end
            todo[stackptr].parent = self.nNodes-1
            stackptr += 1

            todo[stackptr].start = start
            todo[stackptr].end = mid
            todo[stackptr].parent = self.nNodes-1
            stackptr += 1

        flatTree = []
        for n in range(self.nNodes):
            flatTree.append(buildnodes[n])
        self.flatTree = flatTree

    def getIntersection(self, ray, occlusion):

        intersection = Intersection()
        intersection.t = np.PINF

        if len(self.flatTree) == 0:
            return intersection

        bbhits = np.zeros(4)
        closer, other = None, None

        todo = [BVHTraversal() for i in range(64)]
        stackptr = 0

        todo[stackptr].i = 0
        todo[stackptr].mint = np.PINF

        while stackptr >= 0:

            ni = todo[stackptr].i
            near = todo[stackptr].mint
            stackptr -= 1
            node = self.flatTree[ni]

            if near > intersection.t:
                continue

            if node.rightOffset == 0:

                for o in range(node.nPrims):

                    obj = self.objects[node.start+o]
                    data = obj.getIntersection(ray)

                    if data is not None:

                        curr = Intersection(t=data[0],
                                            obj=obj,
                                            hit=data[1],
                                            tri=data[2])

                        if occlusion:
                            return curr

                        if curr.t < intersection.t:
                            intersection = curr

            else:

                hitc0 = self.flatTree[ni+1].bbox.intersect(ray)
                hitc1 = self.flatTree[ni+node.rightOffset].bbox.intersect(ray)

                if hitc0 is not None and hitc1 is not None:

                    closer = ni+1
                    closer_hit = hitc0
                    other = ni+node.rightOffset
                    other_hit = hitc1

                    if hitc1[0] < hitc0[0]:
                        temp = closer
                        closer = other
                        other = temp

                        temp = closer_hit
                        closer_hit = other_hit
                        other_hit = temp

                    stackptr += 1
                    todo[stackptr] = BVHTraversal(other, other_hit[0])
                    stackptr += 1
                    todo[stackptr] = BVHTraversal(closer, closer_hit[0])

                elif hitc0 is None:

                    stackptr += 1
                    todo[stackptr] = BVHTraversal(ni+1, hitc0[0])

                elif hitc1 is None:

                    stackptr += 1
                    todo[stackptr] = BVHTraversal(ni + node.rightOffset, hitc1[0])

        return intersection
