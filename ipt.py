import open3d as o3d
import torch
import numpy as np
from ctypes import *
import sys,os
from scipy.sparse import csr_matrix

P_MIN = 1e-3

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

lib_ipt = cdll.LoadLibrary("./build/libipt.so")
lib_pt = cdll.LoadLibrary("./build/libpt.so")
(Cube,Sphere,Cornell,Other) = (0,1,2,3)

c_int_p = POINTER(c_int)
c_float_p = POINTER(c_float)

class ObjParams(Structure):

    _fields_= [
        ('shp', c_int),
        ('pos', (c_float * 3)),
        ('ori', (c_float * 3)),
        ('scl', (c_float * 3)),
        ('obj_file', c_char_p),
        ('mtl_file', c_char_p)
    ]

    def convert(self,arr):
        
        if isinstance(arr,list): return arr
        elif torch.is_tensor(arr): arr = arr.cpu().detach().numpy()
        return list(arr)

    def __init__(self,shp,pos,ori,scl,obj_file="",mtl_file=""):

        self.shp = shp
        self.pos = (c_float * 3)(*self.convert(pos))
        self.ori = (c_float * 3)(*self.convert(ori))
        self.scl = (c_float * 3)(*self.convert(scl))
        self.obj_file = c_char_p(obj_file.encode('utf-8'))
        self.mtl_file = c_char_p(mtl_file.encode('utf-8'))

def dereference(py_objects):

    n = len(py_objects)
    shps = np.array([o.shp for o in py_objects]).astype(np.int32).ctypes.data_as(c_int_p)
    poss = (c_float_p * n)(*[o.pos for o in py_objects])
    oris = (c_float_p * n)(*[o.ori for o in py_objects])
    scls = (c_float_p * n)(*[o.scl for o in py_objects])
    objs = (c_char_p * n)(*[o.obj_file for o in py_objects])
    mtls = (c_char_p * n)(*[o.mtl_file for o in py_objects])
    
    return shps,poss,oris,scls,objs,mtls,n

def generate_graph():

    cube = ObjParams(Cube,np.array([0,-1.5,4]),np.zeros(3),np.ones(3))
    cornell = ObjParams(Cornell,np.array([0,0,4]),np.zeros(3),np.ones(3)*2)
    
    py_objects = [cornell,cube]
    params = dereference(py_objects)
    
    for i in range(2):
        scene_ptr = c_void_p(0)
        filename_ptr = c_char_p(f"imgs/{i}.png".encode('utf-8'))
        n_t = lib_pt.loadScene(*params,pointer(scene_ptr))        

        lib_pt.createImage(scene_ptr,filename_ptr)
        lib_pt.freeScene(scene_ptr)
        """
    len_data = 7
    data_sz = (n_t + 1) * n_t * 7

    data = (c_float * data_sz)(0)
    lib_ipt.createGraph(scene_ptr,data)
    data = np.ctypeslib.as_array(data).astype(np.float64)
    
    size = (n_t + 1) * n_t
    w = data[:size].reshape(n_t+1,n_t)
    pixel = data[size:size*4].reshape(n_t+1,n_t,3)
    light = data[size*4:].reshape(n_t+1,n_t,3)
    """

def visualize(w_mat):
    t = o3d.io.read_triangle_mesh("CornellBox/CornellBox-Empty-CO.obj")

    centroids = np.asarray(t.vertices)[np.asarray(t.triangles)].mean(axis=-2)
    vs = o3d.utility.Vector3dVector(centroids)
    
    ij = np.array(w_mat.nonzero()).T
    ids = o3d.utility.Vector2iVector(ij)
    
    vals = w_mat[w_mat != 0]
    vals_c = np.array((vals,1-vals,np.zeros_like(vals))).T.reshape(-1,3)
    colors = o3d.utility.Vector3dVector(vals_c)

    lines = o3d.geometry.LineSet(vs,ids)
    lines.colors = colors
    o3d.io.write_line_set("lines.ply",lines)
    o3d.io.write_triangle_mesh("mesh.ply",t)

if __name__ == "__main__":
    generate_graph()
