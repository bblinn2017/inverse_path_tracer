import open3d as o3d
import torch
import numpy as np
from ctypes import *
import sys,os

lib = cdll.LoadLibrary("./build/libipt.so")

def generate_graph():
    obj_files = [
        'Cornell',
        'Cube'
    ]
    mtl_files = ["",""]

    assert(len(obj_files) == len(mtl_files))
    obj_files = (c_char_p * len(obj_files))(*[s.encode('utf-8') for s in obj_files])
    mtl_files = (c_char_p * len(mtl_files))(*[s.encode('utf-8') for s in mtl_files])

    lib.loadScene(obj_files,mtl_files,len(obj_files))

if __name__ == "__main__":
    generate_graph()
