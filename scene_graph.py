import numpy as np
from scene_basics import *
from BVH import *

class Camera:

    def __init__(self,position,direction,up,heightAngle,aspectRatio):

        self.position = position
        self.direction = direction / np.linalg.norm(direction)
        self.up = up / np.linalg.norm(direction)
        self.heightAngle = heightAngle
        self.aspectRatio = aspectRatio

    def getViewMatrix(self):

        f = self.direction
        s = np.cross(f,self.up)
        u = np.cross(s,f)

        return np.stack((
            np.append(s,-np.dot(s,self.position)),
            np.append(u,-np.dot(u,self.position)),
            np.append(-f,np.dot(f,self.position)),
            np.array([0,0,0,1])
        ))

    def getScaleMatrix(self):

        heightAngleRads = np.pi * self.heightAngle / 360.
        tanThetaH = np.tan(heightAngleRads)
        tanThetaW = self.aspectRatio * tanThetaH

        return np.eye(4) * np.array([tanThetaW,tanThetaH,1,1])

    def getInverseViewMatrix(self):

        return self.getScaleMatrix() @ self.getViewMatrix().T

class Scene:

    def __init__(self,camera,objects,scene_obj,leaf_size=4):

        self.camera = camera
        self.objects = objects
        self.BVH = BVH(objects,leaf_size)
        self.scene_obj = scene_obj

        self.emissives = []
        for object in objects + [scene_obj]:
            for triangle in object.mesh.triangles:
                if np.any(triangle.material.emission != 0):
                    self.emissives.append(triangle)

    def get_intersection(self,ray,occlusion=False):

        intersection = self.BVH.getIntersection(ray,occlusion)
        if intersection.obj == None:
            data = self.scene_obj.getIntersection(ray)
            if data != None:
                t,hit,tri = self.scene_obj.getIntersection(ray)
                intersection = Intersection(t=t,obj=self.scene_obj,hit=hit,tri=tri)
        return intersection
