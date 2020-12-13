import numpy as np
import pymesh
from scipy.spatial.transform import Rotation

# Define Shapes
SHAPE_CUBE = 0
SHAPE_SPHERE = 1
SHAPE_OTHER = 2
# Define Meshes
MESH_CUBE = pymesh.generate_box_mesh(
    -0.5 * np.ones(3),
    0.5 * np.ones(3)
)
MESH_SPHERE = pymesh.generate_icosphere(
    0.5,
    np.zeros(3),
    2
)

class Material:

    def __init__(self,params):

        self.diffuse = params['diffuse'] if 'diffuse' in params else np.zeros(3)
        self.specular = params['specular'] if 'specular' in params else np.zeros(3)
        self.glossiness = params['glossiness'] if 'glossiness' in params else np.zeros(3)
        self.emission = params['emission'] if 'emission' in params else np.zeros(3)

class Intersection:

    def __init__(self,t=None,obj=None,hit=None,tri=None):

        self.t = t
        self.obj = obj
        self.hit = hit
        self.tri = tri

class Ray:

    def __init__(self,p,d):

        self.p = p
        self.d = d

    def transform(self,matrix):

        if len(matrix) == 4:
            self.p = (matrix @ np.append(self.p,1))[:3]
            self.d = (matrix @ np.append(self.d,0))[:3]
        else:
            self.p = matrix @ p
            self.d = np.linalg.inverse(matrix).T @ self.d

class Triangle:

    def __init__(self,vertices,material,idx):

        self.vertices = vertices
        n = np.cross(vertices[1]-vertices[0],vertices[2]-vertices[1])
        self.normal = n / np.linalg.norm(n)
        self.material = material
        self.idx = idx

    def area(self):

        a,b,c = self.vertices
        return 0.5 * np.linalg.norm(np.cross(a-b,b-c))

class Mesh:

    def __init__(self,
    shape_type,
    position=np.zeros(3),
    orientation=np.zeros(3),
    scale=None,
    mat_params=None,
    mesh=None):

        self.shape_type = shape_type
        self.position = position
        self.orientation = orientation
        self.scale = scale
        self.vertices,self.faces,self.triangles = self.build_mesh(mat_params,mesh)

    def build_mesh(self,mat_params,m):

        if self.shape_type == SHAPE_CUBE:
            mesh = MESH_CUBE
        elif self.shape_type == SHAPE_SPHERE:
            mesh = MESH_SPHERE
        elif self.shape_type == SHAPE_OTHER:
            mesh = m
        else:
            raise Exception("No such shape exists!")

        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        if self.scale != None:
            vertices *= self.scale

        R = Rotation.from_rotvec(self.orientation).as_matrix()
        vertices = vertices @ R.T + self.position

        triangles = []
        for i in range(len(faces)):
            f = faces[i]
            v = vertices[f]

            if i in mat_params:
                data = mat_params[i]
            else:
                data = mat_params[-1]

            mat = Material(data)

            tri = Triangle(v,mat,i)
            triangles.append(tri)
        return vertices,faces,triangles

class Object:

    def __init__(self,
    mesh):

        self.mesh = mesh

    def getCentroid(self):

        return self.mesh.position

    def getIntersection(self,ray):

        def sd(point, s0, s1, normal):

            out = np.cross(s1 - s0, normal)
            out = out / np.linalg.norm(out)
            d = -np.sum(out*(s1+s0)/2.,axis=-1)

            sd = np.sum(point * out, axis=-1) + d

            return sd

        triangles = self.mesh.triangles

        normal = np.stack([t.normal for t in triangles])
        a,b,c = np.transpose([t.vertices for t in triangles],[1,0,2])
        centroid = np.mean((a,b,c),axis=0)

        denom = np.sum(normal*ray.d,axis=-1)
        idx = (np.abs(denom) > 0.0001).nonzero()[0]

        t = np.sum((centroid[idx] - ray.p) * normal[idx],axis=-1) / denom[idx]

        point = ray.p + ray.d * np.expand_dims(t,1)

        d1 = sd(point,a[idx],b[idx],normal[idx])
        d2 = sd(point,b[idx],c[idx],normal[idx])
        d3 = sd(point,c[idx],a[idx],normal[idx])

        has_int = np.logical_and(d1 <= 0,d2 <= 0)
        has_int = np.logical_and(has_int, d3 <=0)
        has_int = np.logical_and(has_int, t > 0)

        if not np.any(has_int):
            return None

        idx = idx[has_int]
        t = t[has_int]
        point = point[has_int]

        sub_idx = np.argmin(t)

        int_t = t[sub_idx]
        int_point = point[sub_idx]
        int_tri = self.mesh.triangles[idx[sub_idx]]

        return int_t,int_point,int_tri
