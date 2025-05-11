#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# ==== Патч numpy для обратной совместимости с geomdl ====
setattr(np, 'float', float)
setattr(np, 'int', int)
# ========================================================
import trimesh
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from geomdl import fitting, exchange
from geomdl.visualization import VisMPL
import matplotlib.pyplot as plt


# === Параметры: подставь свои! ===
input_obj     = "bodies/cube_quads.obj"   # имя входного OBJ
output_obj    = "nurbs_result.obj" # имя выходного OBJ
grid_size_u   = 50                 # число точек сетки по U
grid_size_v   = 50                 # число точек сетки по V
degree_u      = 3                  # степень NURBS по U
degree_v      = 3                  # степень NURBS по V
visualize     = True               # показать результат? (требуется matplotlib)
# ==================================

# 1) Загрузка меша и получение массива вершин
mesh = trimesh.load(input_obj)
if not isinstance(mesh, trimesh.Trimesh):
    raise RuntimeError(f"{input_obj} не распознан как корректный Trimesh")
points = np.asarray(mesh.vertices)  # (N,3)

# 2) PCA для «развёртки» точек на плоскость (u,v) и высоту w
pca = PCA(n_components=3)
coords_uvw = pca.fit_transform(points)
u, v, w = coords_uvw[:,0], coords_uvw[:,1], coords_uvw[:,2]

# 3) Генерация равномерной сетки (grid_u, grid_v) и интерполяция w
u_lin = np.linspace(u.min(), u.max(), grid_size_u)
v_lin = np.linspace(v.min(), v.max(), grid_size_v)
grid_u, grid_v = np.meshgrid(u_lin, v_lin, indexing='ij')

# метод cubic даст гладкую поверхность, но может выдавать NaN у краёв:
grid_w = griddata((u, v), w, (grid_u, grid_v), method='cubic')

# Заполняем NaN ближайшим значением для надёжности
mask = np.isnan(grid_w)
if np.any(mask):
    grid_w[mask] = griddata((u, v), w, (grid_u[mask], grid_v[mask]),
                            method='nearest')

# 4) Собираем pts2d для geomdl: из (grid_u, grid_v, grid_w) обратно в 3D
print('grid_u')
print(grid_u)
print('\n\n\n\n')

print('grid_v')
print(grid_v)
print('\n\n\n\n')

print('grid_w')
print(grid_w)
print('\n\n\n\n')

print('pts')
pts2d = []
for i in range(grid_size_u):
    row = []
    for j in range(grid_size_v):
        uvw = np.array([grid_u[i,j], grid_v[i,j], grid_w[i,j]])
        xyz = pca.inverse_transform(uvw)
        row.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    pts2d.append(row)

print(pts2d)

flat_pts = [pt for row in pts2d for pt in row]    # теперь len(flat_pts) == grid_size_u * grid_size_v

# 5) Аппроксимация NURBS-поверхности
surf = fitting.approximate_surface(
    flat_pts,
    size_u=grid_size_u,
    size_v=grid_size_v,
    degree_u=degree_u,
    degree_v=degree_v,
    periodic_u=True,
    chlen= True,
)
surf.delta = 0.05  # детализация при визуализации

pts = np.array(surf.evalpts)
print("Z-разброс:", pts[:,2].min(), "…", pts[:,2].max())

# 6) Экспорт результата
exchange.export_obj(surf, output_obj)
print(f"NURBS-поверхность сохранена в «{output_obj}»")

# 7) Визуализация (опционально)
if visualize:
    vis = VisMPL.VisSurface(ctrlpts=True, legend=False, axes=True)
    surf.vis = vis
    surf.render()
