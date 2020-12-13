import numpy as np
from scene_graph import *
from scene_basics import *
import pymesh

SCENE_SIZE = 4

def CornellBox():

    vertices = np.array([
        [0,0,0], # 0
        [1,0,0], # 1
        [0,1,0], # 2
        [1,1,0], # 3
        [0,0,1], # 4
        [1,0,1], # 5
        [0,1,1], # 6
        [1,1,1], # 7
        [.25,1,.25], # 8
        [.75,1,.25], # 9
        [.25,1,.75], # 10
        [.75,1,.75]  # 11
    ])

    faces = np.array([
        [0,2,4],
        [4,2,6],
        [1,5,7],
        [1,7,3],
        [0,4,1],
        [1,4,5],
        [5,4,6],
        [5,6,7],
        [8,2,3],
        [8,3,9],
        [9,3,7],
        [9,7,11],
        [11,7,6],
        [11,6,10],
        [10,6,2],
        [10,2,8],
        [10,8,9],
        [10,9,11]
    ])

    # Normalize
    vertices *= SCENE_SIZE
    vertices += np.array([-.5*SCENE_SIZE,-.5*SCENE_SIZE,2])
    new_mesh = pymesh.form_mesh(vertices,faces)

    mat_params = {}
    num_faces = len(new_mesh.faces)
    for i in range(num_faces):
        if i < num_faces - 2:
            mat_params[i] = {'diffuse':np.ones(3)}
        else:
            mat_params[i] = {'emission':np.ones(3)}

    mesh = Mesh(
    SHAPE_OTHER,
    scale=None,
    mat_params=mat_params,
    mesh=new_mesh
    )

    scene_obj = Object(mesh)
    return scene_obj

def initialize():

    num_obj = 2
    radius = 0.5

    eye = np.array([0,0,0])
    look = np.array([0,0,-1])
    up = np.array([0,1,0])

    objects = []
    for i in range(num_obj):

        rand = np.random.uniform(size=3)
        pos = (SCENE_SIZE -1) * rand + np.array([-1.5,-1.5,2.5])

        mat_params = {-1:{
        'diffuse':np.random.uniform(3),
        'specular':np.ones(3),
        'glossiness':np.random.uniform()}}

        mesh = Mesh(
        np.random.choice(2),
        position=pos,
        mat_params=mat_params,
        mesh=None)
        obj = Object(mesh)

        objects.append(obj)

    scene_obj = CornellBox()
    camera = Camera(pos,look,up,90,1)

    scene = Scene(camera,objects,scene_obj)
    return scene
