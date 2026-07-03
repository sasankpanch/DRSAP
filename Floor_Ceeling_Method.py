
import sys
import numpy as np
import trimesh

verts = np.asarray(trimesh.load(sys.argv[1], process=False).vertices)

mn = verts.min(axis=0)          # [min_x, min_y, min_z]
mx = verts.max(axis=0)          # [max_x, max_y, max_z]
ext = mx - mn                   # [width_x, depth_y, height_z]

height = ext[2]
length, width = sorted([ext[0], ext[1]], reverse=True)
volume = length * width * height

print(f"X:  min {mn[0]:8.3f}   max {mx[0]:8.3f}   extent {ext[0]:7.3f} m")
print(f"Y:  min {mn[1]:8.3f}   max {mx[1]:8.3f}   extent {ext[1]:7.3f} m")
print(f"Z:  min {mn[2]:8.3f}   max {mx[2]:8.3f}   extent {ext[2]:7.3f} m  (height)")
print()
print(f"length x width x height = {length:.3f} x {width:.3f} x {height:.3f} m")
print(f"VOLUME = {volume:.3f} m^3")