# Путь к .obj файлу

import trimesh
import os

obj_path = "bodies/2_norm.obj"
# Проверка на наличие файла
if not os.path.exists(obj_path):
    raise FileNotFoundError(f"OBJ файл не найден: {obj_path}")

# Загружаем меш (автоматически подгружает .mtl, если прописан в OBJ)
mesh = trimesh.load(obj_path, force='scene')  # используем сцену, чтобы подтянуть материалы

# Визуализация
mesh.show()
