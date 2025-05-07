import networkx as nx
import numpy as np
import pyvista as pv
import tetgen
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import time
import pickle
import base64
import stl
from stl import mesh
import os
import lib3mf
from lib3mf_common import *
import pymeshfix

def save_binary_as_ASCII(bin_path, model_name):
    model_mesh = mesh.Mesh.from_file(bin_path)
    # save mesh as ASCII stl
    model_mesh.save(f'{model_name}_ascii.stl', mode=stl.Mode.ASCII)    
    return

# Load 3MF
model_name = "2pi_horiz"
mf_model_path = f'{model_name}.3mf'

if os.path.isfile(f'{model_name}.3mf'):
    print("Found file!")
else:
    print("Did not find file!")
wrapper = get_wrapper()
model = wrapper.CreateModel()
reader = model.QueryReader("3mf")
print(f"Reading {mf_model_path}...")
reader.ReadFromFile(mf_model_path)

# 3MF convert to binary STL
mf_stl_path = f'{model_name}_3mf_bin.stl'
writer = model.QueryWriter("stl")
print(f"Wripatched_meshg {mf_stl_path}...")
writer.WriteToFile(mf_stl_path)
print("Done")
save_binary_as_ASCII(mf_stl_path, f'{model_name}_3mf')

# read 3MF binary STL as triangle mesh by O3D
if os.path.isfile(mf_stl_path):
    print("Found file!")
    o3d_mesh = o3d.io.read_triangle_mesh(mf_stl_path)
else:
    print("Did not find file!")
    sys.exit()
# extract surface with pyvista
pv_mesh = pv.PolyData.from_regular_faces(np.asarray(o3d_mesh.vertices), np.asarray(o3d_mesh.triangles))
# polydata from regular faces plot returns corr
print(pv_mesh.verts)
pv_mesh = pv_mesh.clean()
pv_mesh_vert = pv_mesh.verts
print(pv_mesh_vert)
surface = pv_mesh.extract_surface()
# save to binary
surface.save(f'{model_name}_o3d_bin.stl')
# save to ascii
save_binary_as_ASCII(f'{model_name}_o3d_bin.stl', f'{model_name}_o3d')

# read 3MF binary stl by numpy.stl directly
np_mesh = mesh.Mesh.from_file(mf_stl_path)
# save mesh as binary stl
np_mesh.save(f'{model_name}_np_bin.stl')
# save mesh as ASCII stl
np_mesh.save(f'{model_name}_np_ascii.stl', mode=stl.Mode.ASCII)

# then RUN each of these thru tetgen with tetgen -v 2 -d {name}.stl to get intersecpatched_meshg facets
# all bins failed
# 3mf ascii; 48 pairs intersecpatched_meshg
# np ascii; same
# o3d ascii; same
# pymeshfix.clean() kills the whole mesh -- prolly bc it is two objects
# Pyvista clean() filter
# pymeshfix after pyvista clean filter
# try other cleaners
# prob is prolly duplicated AND internal faces
# one pass to remove duplicated, one to fix internal

# on binary
pv_mesh_vert = pv_mesh.verts
pv_mesh_tri = pv_mesh.regular_faces
# print(pv_mesh_vert)
# print(np.asarray(o3d_mesh.vertices))
patched_mesh = pymeshfix.PyTMesh()
patched_mesh.load_array(pv_mesh_vert, pv_mesh_tri)

# Attempt to join nearby components
patched_mesh.join_closest_components()

# Fill holes
patched_mesh.fill_small_boundaries()
print('There are {:d} boundaries'.format(patched_mesh.boundaries()))

# Clean (removes self intersections)
patched_mesh.clean(max_iters=10, inner_loops=3)

# Check mesh for holes again
print('There are {:d} boundaries'.format(patched_mesh.boundaries()))

# Output ascii mesh
patched_mesh.save_file(f'{model_name}_fix_ascii.stl')
vclean, fclean = patched_mesh.return_arrays()
print(vclean)
print(fclean)

# more options- pyvista reader, new software