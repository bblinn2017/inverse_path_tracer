import numpy as np
from scipy.spatial.transform import Rotation
from scene_basics import *
from scene_graph import *
from path_trace	import *
from BVH import *

import threading
import queue
from tqdm import tqdm
from PIL import Image

IM_WIDTH = 200
IM_HEIGHT = 200
SAMPLE_NUM = 5
p_RR = 0.7
num_threads = 10

class RenderThread(threading.Thread):

    def __init__(self, threadID, **kwargs):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.queue = kwargs['work_queue']
        self.queue_lock = kwargs['work_queue_lock']
        self.image = kwargs['image']
        self.image_lock = kwargs['image_lock']
        self.pbar = kwargs['pbar']
        self.scene = kwargs['scene']

    def run(self):

        while not self.queue.empty():
            self.queue_lock.acquire()
            r,c = self.queue.get()
            self.queue_lock.release()

            val = render_pixel(self.scene,r,c)
            self.image_lock.acquire()
            self.image[r,c] = val
            self.pbar.update(1)
            self.image_lock.release()

def BSDF(intersection, w, w_i, isDirect):

    mat = intersection.tri.material

    diffuse = mat.diffuse
    if not isDirect:
        diffuse /= np.pi

    n = mat.glossiness
    norm = intersection.tri.normal
    refl = w_i - 2. * np.dot(norm,w_i) * norm
    coeff = (n+2.)/2./np.pi * np.dot(refl,w) ** n
    specular = mat.specular * coeff

    return diffuse + specular

def directLighting(intersection, d, scene):

    t_curr = intersection.tri
    emissives = scene.emissives

    L_o = np.zeros(3)

    areas = []
    for e in emissives:
        areas.append(e.area())
    p = np.array(areas) / np.sum(areas)

    idx = np.random.choice(len(emissives),p=p)
    area = areas[idx]
    t_emm = emissives[idx]

    r1,r2 = np.random.uniform(size=2)
    v1,v2,v3 = t_emm.vertices

    emm_point = (1 - (r1 ** .5)) * v1 \
        + (r1 ** .5 * (1 - r2)) * v2 \
        + (r2 * r1 ** .5) * v3

    curr_int = intersection.hit
    to_light = emm_point - curr_int
    to_light = to_light / np.linalg.norm(to_light)

    light_ray = Ray(curr_int,to_light)

    cos_theta = np.dot(t_curr.normal,to_light)
    if cos_theta < 0.:
        return np.zeros(3)

    i = scene.get_intersection(light_ray)
    if i.obj == None:
        return np.zeros(3)

    cos_theta_prime = -np.dot(t_emm.normal,to_light)
    if cos_theta_prime < 0.:
        return np.zeros(3)

    t_int = i.tri
    if t_int.idx != t_emm.idx:
        return np.zeros(3)

    L_o_j = t_emm.material.emission

    L_o += L_o_j * cos_theta * cos_theta_prime / (i.t ** 2) / (1./area)

    L_o *= BSDF(intersection, d, to_light, True)

    return L_o

def sampleNextDir(normDir, isSpecular, shininess):

    phi = 2. * np.pi * np.random.uniform()
    if not isSpecular:
        theta = np.arccos(np.random.uniform() ** .5)
    else:
        theta = np.arccos(np.random.uniform() ** (1./(shininess+1)))

    hemiDir = np.array([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ])
    normDir0 = np.array([0,0,1])

    angle_axis = np.cross(normDir0,normDir)
    R = Rotation.from_rotvec(angle_axis).as_matrix()

    dir = R @ hemiDir
    dir /= np.linalg.norm(dir)

    if not isSpecular:
        pdf = 1 / np.pi
    else:
        pdf = (shininess+1) * np.cos(theta) ** shininess

    return {'dir':dir,'pdf':pdf}

def radiance(ray, scene, recursion):

    intersection = scene.get_intersection(ray)

    if intersection.obj == None:
        return np.zeros(3)

    tri = intersection.tri
    mat = tri.material

    # Emissive lighting TBD
    L_e = np.zeros(3)
    if recursion == 0:
        L_e += mat.emission

    # Mirror lighting TBD

    # Direct lighting
    L_d = directLighting(intersection, ray.d, scene)

    # Next lighting
    L_i = np.zeros(3)
    isSpecular = (mat.specular > 0).any()
    sample = sampleNextDir(tri.normal,isSpecular,mat.glossiness)

    next_ray = Ray(intersection.hit,sample['dir'])
    if np.random.uniform() < p_RR:
        L_i += radiance(next_ray, scene, recursion + 1)
    coeff = BSDF(intersection,ray.d,sample['dir'],False) * np.dot(sample['dir'],tri.normal) / sample['pdf'] / p_RR
    L_i *= coeff

    return L_e + L_d + L_i

def trace_ray(ray, scene):

    return radiance(ray, scene, 0)

def render_pixel(scene,r,c):

    p = np.zeros(3)
    d = np.array([2 * c / IM_WIDTH - 1,
                      1 - 2 * r / IM_HEIGHT,
                      1])
    d /= np.linalg.norm(d)

    ray = Ray(p,d)
    ray.transform(scene.camera.getInverseViewMatrix())

    output = np.zeros(3)

    for i in range(SAMPLE_NUM):
        output += trace_ray(ray, scene) / SAMPLE_NUM

    return output

def toneMap(image):

    new_image = np.zeros((IM_HEIGHT,IM_WIDTH,3))

    for r in range(IM_HEIGHT):
        for c in range(IM_WIDTH):
            new_image[r,c] = 255 * image[r,c] / (1 + image[r,c])
    return new_image.astype(np.uint8)

def render_scene(scene):

    q_lock = threading.Lock()
    image_lock = threading.Lock()
    q = queue.Queue()
    for r in range(IM_HEIGHT):
        for c in range(IM_WIDTH):
            q.put((r,c))
    image = np.zeros((IM_HEIGHT,IM_WIDTH,3))

    pbar = tqdm(total=IM_HEIGHT*IM_WIDTH)

    args = {
        'work_queue':q,
        'work_queue_lock':q_lock,
        'image':image,
        'image_lock':image_lock,
        'pbar':pbar,
        'scene':scene
    }
    threads = []
    for i in range(num_threads):
        thread = RenderThread(i,**args)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    new_image = Image.fromarray(toneMap(image))
    new_image = new_image.resize((500,500), Image.ANTIALIAS)
    new_image.save("RENDERED.png")
