# Путь к .obj файлу

import os

import trimesh

obj_path = "bodies/2_norm.obj"
obj_path = "bodies/cube_quads.obj"
#obj_path = "bodies/cube_with_normals.obj"
#obj_path = "bodies/cube_corrected_normals.obj"
obj_path = "nurbs_result.obj"
obj_path = "bodies/sphere.obj"
# Проверка на наличие файла
if not os.path.exists(obj_path):
    raise FileNotFoundError(f"OBJ файл не найден: {obj_path}")

# Загружаем меш (автоматически подгружает .mtl, если прописан в OBJ)
mesh = trimesh.load(obj_path, force='scene')  # используем сцену, чтобы подтянуть материалы
mesh = trimesh.load(obj_path)  # используем сцену, чтобы подтянуть материалы

# Визуализация
mesh.show()
