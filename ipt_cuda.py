import numpy as np
from ctypes import *
import sys,os
from scipy.sparse import csr_matrix
from tqdm import tqdm

lib_ipt = cdll.LoadLibrary("./build/libipt.so")
lib_pt = cdll.LoadLibrary("./build/libpt.so")
(Cube,Sphere,Cornell,Other) = (0,1,2,3)

c_int_p = POINTER(c_int)
c_float_p = POINTER(c_float)

def rand_mtl():
    return f"""*Kd {np.random.uniform()} {np.random.uniform()} {np.random.uniform()}*"""

def to_string(shp=None,pos=None,ori=None,scl=None,obj_file=None,mtl_file=None):
    string = ""
    if pos is not None:
        string += f'POS {pos[0]} {pos[1]} {pos[2]}\n'
    if ori is not None:
        string += f'ORI {ori[0]} {ori[1]} {ori[2]}\n'
    if scl is not None:
        string += f'SCL {scl[0]} {scl[1]} {scl[2]}\n'
    if shp is Cube:
        obj_file = "./shapes/cube.obj"
        mtl_file = rand_mtl() if mtl_file is None else mtl_file
    elif shp is Sphere:
        obj_file = "./shapes/sphere.obj"
        mtl_file = rand_mtl() if mtl_file is None else mtl_file
    elif shp is Cornell:
        obj_file = "./CornellBox/CornellBox-Empty-CO.obj"
        mtl_file = "./CornellBox/CornellBox-Empty-CO.mtl"
    assert obj_file is not None and mtl_file is not None
    string += f'OBJ {obj_file}\n'
    string += f'MTL {mtl_file}\n'
    return string

def from_string(string):
    lines = string.split("\n")
    pos,ori,scl,obj_file,mtl_file = [None] * 5
    for line in lines:
        items = line.strip().split(" ")
        token,values = items[0],items[1:]
        if token == "POS":
            pos = [float(x) for x in values]
        elif token == "ORI":
            ori = [float(x) for x in values]
        elif token == "SCL":
            scl = [float(x) for x in values]
        elif token == "OBJ":
            obj_file = values[0]
        elif token == "MTL":
            mtl_file = " ".join(values)
    if pos is None: pos = [0] * 3
    if ori is None: ori = [0] * 3
    if scl is None: scl = [1] * 3
    assert obj_file is not None and mtl_file is not None
    return ObjParams(pos,ori,scl,obj_file,mtl_file)

class ObjParams(Structure):

    _fields_ = [
        ('shp', c_int),
        ('pos', (c_float * 3)),
        ('ori', (c_float * 3)),
        ('scl', (c_float * 3)),
        ('obj_file', c_char_p),
        ('mtl_file', c_char_p)
    ]

    def __init__(self,pos,ori,scl,obj_file,mtl_file):

        self.pos = (c_float * 3)(*pos)
        self.ori = (c_float * 3)(*ori)
        self.scl = (c_float * 3)(*scl)
        self.obj_file = c_char_p(obj_file.encode('utf-8'))
        self.mtl_file = c_char_p(mtl_file.encode('utf-8'))

def dereference(py_objects):

    n = len(py_objects)
    poss = (c_float_p * n)(*[o.pos for o in py_objects])
    oris = (c_float_p * n)(*[o.ori for o in py_objects])
    scls = (c_float_p * n)(*[o.scl for o in py_objects])
    objs = (c_char_p * n)(*[o.obj_file for o in py_objects])
    mtls = (c_char_p * n)(*[o.mtl_file for o in py_objects])
    
    return poss,oris,scls,objs,mtls,n

def load_params(filename):
    f = open(filename,"r")

    lines = f.readlines()
    obj_params = []
    curr = ""
    for line in lines:
        line = line.strip()
        if line == "OBJECT":
            if len(curr) > 0: obj_params.append(from_string(curr))
            curr = ""
        else:
            curr += line + "\n"
    obj_params.append(from_string(curr))
        
    params = dereference(obj_params)
    return params

def load_scene(scenefile):
    params = load_params(scenefile)
    scene_ptr = c_void_p(0)
    n_t = lib_pt.loadScene(*params,pointer(scene_ptr))
    return n_t,scene_ptr

def generate_files(n):

    for i in tqdm(range(n)):
        f = open(f"scenes/{i}.txt","w")

        f.write(f"OBJECT\n")
        f.write(to_string(
            shp=Cornell,pos=np.array([0,0,4]),scl=np.ones(3)*2
        ))
        f.write(f"OBJECT\n")
        f.write(to_string(
            shp=Cube,pos=np.array([0,-1.5,4])
        ))
        f.close()

        # Load Scene
        n_t,scene_ptr = load_scene(f'scenes/{i}.txt')
        filename_ptr = c_char_p(f'imgs/{i}.png'.encode('utf-8'))
        lib_pt.createImage(scene_ptr,filename_ptr)
        lib_ipt.freeScene(scene_ptr)

def generate_data(scenefile,imgfile):

    len_data = 7

    # Load Scene
    n_t,scene_ptr = load_scene(scenefile)
    filename_ptr = c_char_p(imgfile.encode('utf-8'))
    
    # Create Data
    size = (n_t + 1) * n_t
    data_sz = size * len_data
    data = (c_float * data_sz)(0)
    lib_ipt.createGraph(scene_ptr,filename_ptr,data)

    # Get Labels
    labels = (c_float * (n_t * 3))(0)
    lib_ipt.getMaterials(scene_ptr,labels)
    labels = np.ctypeslib.as_array(labels).astype(np.float64).reshape(n_t,3)

    # Free Scene
    lib_ipt.freeScene(scene_ptr)
    
    # Populate Graph
    data = np.ctypeslib.as_array(data).astype(np.float64)
    w = data[:size].reshape(n_t+1,n_t)
    pixel = data[size:size*4].reshape(n_t+1,n_t,3)
    assert not np.isnan(pixel).any()
    light = data[size*4:].reshape(n_t+1,n_t,3)
        
    return w,pixel,light,labels

def render_with_materials(scenefile,imgfile,materials):
    
    # Load Scene
    params = load_params(scenefile)
    scene_ptr = c_void_p(0)
    filename_ptr = c_char_p(imgfile.encode('utf-8'))
    n_t = lib_pt.loadScene(*params,pointer(scene_ptr))

    # Set Materials
    materials = materials.numpy().astype(np.float32).ctypes.data_as(c_float_p)
    lib_ipt.setMaterials(scene_ptr,materials)
    
    # Create Image
    lib_pt.createImage(scene_ptr,filename_ptr)

    # Free Scene
    lib_ipt.freeScene(scene_ptr)

