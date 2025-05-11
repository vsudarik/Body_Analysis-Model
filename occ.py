import trimesh, numpy as np
import matplotlib.pyplot as plt
# 1. Загрузка
file = 'bodies/sphere.obj'
file = 'bodies/cube_quads.obj'

mesh = trimesh.load(file, force='mesh')
center = mesh.center_mass
mesh.apply_translation(-center)

# 2. Параметры сетки
n_theta, n_phi = 50, 100
theta_min, theta_max = np.deg2rad(10), np.deg2rad(170)
thetas = np.linspace(theta_min, theta_max, n_theta)
phis   = np.linspace(0, 2*np.pi, n_phi, endpoint=False)

# 3. Формируем лучи
origins    = np.zeros((n_theta * n_phi, 3))
directions = np.array([
    [np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)]
    for p in phis for t in thetas
])

# 4. Пересечения
loc, ray_idx, _ = mesh.ray.intersects_location(
    ray_origins=origins,
    ray_directions=directions,
    multiple_hits=False
)
print(loc)
print(ray_idx)

# 5. Собираем в массив
points = np.full((n_theta*n_phi, 3), np.nan)
points[ray_idx] = loc
points = points.reshape((n_theta, n_phi, 3))

# 2. Визуализация массива точек
coords = points.reshape(-1, 3)
mask = ~np.isnan(coords).any(axis=1)
valid_coords = coords[mask]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Точки пересечения лучей с поверхностью объекта')
plt.show()

coords_matrix = np.where(np.isnan(points), np.nan, points)
print('cord')
print(coords_matrix)


num_u, num_v, dim = coords_matrix.shape
print(num_u)
print(num_v)
print(dim)

from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.gp import gp_Pnt
points = TColgp_Array2OfPnt(1, num_u, 1, num_v)
# Копируем данные
for i in range(num_u):
    for j in range(num_v):
        x, y, z = coords_matrix[i, j]
        points.SetValue(i + 1, j + 1, gp_Pnt(x, y, z))






from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.Approx import Approx_ChordLength
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.Quantity import Quantity_NOC_RED

'''
import math
# 1. Параметры сетки
num_u = 12  # по U (периодическая)
num_v = 6   # по V
radius = 10
height = 5.0
# 2. Генерация точек: окружность по U, подъём по V
points = TColgp_Array2OfPnt(1, num_u, 1, num_v)
for i in range(num_u):
    angle = 2 * math.pi * i / num_u
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    for j in range(num_v):
        z = j * height / (num_v - 1)
        points.SetValue(i + 1, j + 1, gp_Pnt(x, y, z))
'''

# 3. Построение B-spline поверхности (первый алгоритм) с циклом по U
deg_min = 3
deg_max = 3
continuity = GeomAbs_C2
tol3d = 1e-3
u_periodic = True


builder = GeomAPI_PointsToBSplineSurface()
builder.Init(
    points,
    Approx_ChordLength,  # метод параметризации
    3, 3,                 # DegMin, DegMax
    GeomAbs_C2,
    1e-3,
    True                 # !!! это и есть thePeriodic
)
surface = builder.Surface()


def print_surface_info(surface):
    print("=== B-spline Surface Info ===")

    # Степени
    print("UDegree:", surface.UDegree())
    print("VDegree:", surface.VDegree())

    # Количество контрольных точек
    nb_u = surface.NbUPoles()
    nb_v = surface.NbVPoles()
    print("NbUPoles:", nb_u)
    print("NbVPoles:", nb_v)

    # Периодичность и рациональность
    print("IsUPeriodic:", surface.IsUPeriodic())
    print("IsVPeriodic:", surface.IsVPeriodic())
    print("IsURational:", surface.IsURational())
    print("IsVRational:", surface.IsVRational())

    # Контрольные точки
    print("Poles:")
    poles = surface.Poles()
    for i in range(1, nb_u + 1):
        for j in range(1, nb_v + 1):
            pt = poles.Value(i, j)
            print(f"  P[{i},{j}] = ({pt.X():.3f}, {pt.Y():.3f}, {pt.Z():.3f})")

    # Веса (если рациональная)
    if surface.IsURational() or surface.IsVRational():
        print("Weights:")
        weights = surface.Weights()
        for i in range(1, nb_u + 1):
            for j in range(1, nb_v + 1):
                w = weights.Value(i, j)
                print(f"  w[{i},{j}] = {w:.5f}")

    # Узлы и кратности по U
    print("U Knots and Multiplicities:")
    nb_uknots = surface.NbUKnots()
    for i in range(1, nb_uknots + 1):
        knot = surface.UKnot(i)
        mult = surface.UMultiplicity(i)
        print(f"  KnotU[{i}] = {knot:.4f}, multiplicity = {mult}")

    # Узлы и кратности по V
    print("V Knots and Multiplicities:")
    nb_vknots = surface.NbVKnots()
    for i in range(1, nb_vknots + 1):
        knot = surface.VKnot(i)
        mult = surface.VMultiplicity(i)
        print(f"  KnotV[{i}] = {knot:.4f}, multiplicity = {mult}")
    print("---end surface info----")

print_surface_info(surface)
# 4. Визуализация
display, start_display, _, _ = init_display()
face = BRepBuilderAPI_MakeFace(surface, 1e-6).Face()
display.DisplayShape(face, update=True)
# Отображение контрольных точек
poles = surface.Poles()
nb_u = surface.NbUPoles()
nb_v = surface.NbVPoles()


vertex_map = {}

for i in range(1, nb_u + 1):
    for j in range(1, nb_v + 1):
        pt = poles.Value(i, j)
        vertex = BRepBuilderAPI_MakeVertex(pt).Vertex()
        display.DisplayShape(vertex, color=Quantity_NOC_RED, update=False)
        vertex_map[vertex] = (pt.X(), pt.Y(), pt.Z(), i, j)

# 2. Callback для обработки клика по точке
def on_click(shp, *args, **kwargs):
    print(shp)
    first_shp = shp[0]
    if first_shp in vertex_map:
        x, y, z, i, j = vertex_map[first_shp]
        print(f"Clicked point at: ({x:.3f}, {y:.3f}, {z:.3f}), and order in massiv i: {i}, j:{j}")

display.register_select_callback(on_click)


# 3. Обновление экрана
display.FitAll()
start_display()

######################### new surface ############3

poles = surface.Poles()
new_point = gp_Pnt(-0.032, -1, 0.016) #Clicked point at: (-0.032, -0.500, 0.016), and order in massiv i: 39, j:24
poles.SetValue(39, 24, new_point)
display, start_display, _, _ = init_display()
face = BRepBuilderAPI_MakeFace(surface, 1e-6).Face()
display.DisplayShape(face, update=True)
# Отображение контрольных точек
poles = surface.Poles()
nb_u = surface.NbUPoles()
nb_v = surface.NbVPoles()


vertex_map = {}

for i in range(1, nb_u + 1):
    for j in range(1, nb_v + 1):
        pt = poles.Value(i, j)
        vertex = BRepBuilderAPI_MakeVertex(pt).Vertex()
        display.DisplayShape(vertex, color=Quantity_NOC_RED, update=False)
        vertex_map[vertex] = (pt.X(), pt.Y(), pt.Z(), i, j)

# 2. Callback для обработки клика по точке
def on_click(shp, *args, **kwargs):
    print(shp)
    first_shp = shp[0]
    if first_shp in vertex_map:
        x, y, z, i, j = vertex_map[first_shp]
        print(f"Clicked point at: ({x:.3f}, {y:.3f}, {z:.3f}), and order in massiv i: {i}, j:{j}")

display.register_select_callback(on_click)

# 3. Обновление экрана
display.FitAll()
start_display()
