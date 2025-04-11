"""
Created on Tue Mar  7 16:15:00 2023

@author: 79175
"""

import numpy as np
import os
import open3d as o3d
import nrrd
from quantimpy import minkowski as mk
from skimage.measure import label, regionprops
from skimage.segmentation import flood_fill
from skimage.morphology import closing

from datetime import datetime


# просто функция для вывода сообщений в консоль с штампом текущего времени
def now(message=''):
    print(datetime.now().strftime('%H:%M:%S') + ' ' + str(message))


def obj_to_nrrd(fn, voxel_size=0.5):
    '''
    Вокселизировать obj с заданным размером вокселя voxel_size.
    Файл сохраняется в той же папке в формате nrrd.
    '''
    skull = o3d.io.read_triangle_mesh(fn)
    aabb = skull.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()  # найти размеры бокса, в который заключён объект;
    # я здесь молюсь на то, что open3d и binvox эти боксы одинаково рисует, но по моему опыту выходило ок.

    voxel_number = round(
        extent.max() / voxel_size)  # вычислить количество вокселей по заданному размеру вокселя (у меня было в мм)
    cmd = f"binvox -t nrrd -e -ri -d {voxel_number} \"{fn}\" "
    now(cmd)
    os.system(cmd)


def obj_to_nrrd_dir(path, voxel_size=0.5):
    '''
    Применить obj_to_nrrd ко всем obj файлам в папке.
    '''
    filelist = os.listdir(path)
    for fn in filelist:
        if fn.endswith(".obj"):
            process(os.path.join(path, fn), voxel_size)


def cutdown(data):
    '''
    Обрезает пустые края куба до прямоугольника.
    Binvox всегда сохраняет объект в кубическом боксе, там может быть много пустого пространства,
    что раздувает размер файла и может замедлить алгоритмы, поэтому я от них избавляюсь.
    '''
    ix, iy, iz = np.where(data > 0)
    x0, x1 = min(ix), max(ix)
    y0, y1 = min(iy), max(iy)
    z0, z1 = min(iz), max(iz)
    res = np.zeros((x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1))
    res = data[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
    res = np.pad(res, 1)
    return res


def cutdown_dir(path, writepath):
    '''
    Обрезать у всех nrrd файлов в папке path, сохранить в writepath
    '''
    filelist = os.listdir(path)
    for fn in filelist:
        if fn.endswith(".nrrd"):
            now(fn)
            data, header = nrrd.read(os.path.join(path, fn))
            res = cutdown(data)
            header['sizes'] = np.array(res.shape)
            nrrd.write(os.path.join(writepath, fn), res, header)


def process(data):
    # data = data.copy(order = 'C')
    # data = np.pad(data, 1)
    # res = data > 0

    # фильтрация шума, из-за которого могут быть несвязанные с основным телом компоненты
    res = label(data)
    rps = regionprops(res)
    areas = [x.area for x in rps]  # список "размера" компонент (количества вокселей в них)
    idxs = np.argsort(areas)[
           ::-1]  # сортируем так, чтобы на первом месте была наше тело; всё остальное будем игнорировать
    res = np.full(shape=res.shape, fill_value=False)
    res[tuple(rps[idxs[0]].coords.T)] = True  # в индексах, соответсвующим телу, заполнили True, остальное False

    # заполнение внутренности
    center = tuple(
        np.int64(np.array(res.shape) / 2))  # начинаем закраску с центра массива, надеясь, что он попадём внутрь тела
    res = flood_fill(res, center, True,
                     connectivity=1)  # connectivity связность можно поменять, если закраска просачивается через диагонали

    # после закраски могли остаться диагональные щели, их мы убираем с помощью морфологического закрытия
    res = closing(res)

    # расширяем массив, чтобы тело не упиралось в стенки массива
    res = np.pad(res, 1)
    return res


def example(fname):
    fname = "мой файл.nrrd"
    data, header = nrrd.read(fname)
    res = process(res)
    scalings = (
    abs(header['space directions'][0, 0]), abs(header['space directions'][1, 1]), abs(header['space directions'][2, 2]))
    m = mk.functionals(res, scalings)
    now(m)


if __name__ == '__main__':
    voxel_size = 0.5  # размер вокселя
    obj_to_nrrd_dir('папка с obj', voxel_size)
    # cutdown_dir('папка с nrrd', 'папка куда сохранить') #необязательно, но файлы будут меньшего размера


