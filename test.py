import torch
import numpy as np
import pymesh
from utils import *
from scene_basics import *
from scene_graph import *
from path_trace import *
from BVH import *

if __name__ == "__main__":

    scene = initialize()
    image = render_scene(scene)
