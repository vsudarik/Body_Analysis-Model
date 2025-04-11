import numpy as np
import trimesh
from scipy.special import comb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ---------------------------
# Бернштейновы полиномы
# ---------------------------
def bernstein_basis(n, i, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def fit_bernstein_surface(u, v, r, degree_u=6, degree_v=6):
    A = np.zeros((len(u), (degree_u + 1) * (degree_v + 1)))
    for k in range(len(u)):
        row = []
        for i in range(degree_u + 1):
            for j in range(degree_v + 1):
                row.append(bernstein_basis(degree_u, i, u[k]) * bernstein_basis(degree_v, j, v[k]))
        A[k] = row
    coeffs, _, _, _ = np.linalg.lstsq(A, r, rcond=None)
    return coeffs.reshape((degree_u + 1, degree_v + 1))


def bernstein_surface(u, v, coeffs):
    n, m = coeffs.shape[0] - 1, coeffs.shape[1] - 1
    result = np.zeros_like(u)
    for i in range(n + 1):
        for j in range(m + 1):
            result += coeffs[i, j] * bernstein_basis(n, i, u) * bernstein_basis(m, j, v)
    return result


# ---------------------------
# Перевод декартовых координат в сферические
# ---------------------------
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


# ---------------------------
# Основная часть
# ---------------------------
def main():
    # Путь к .obj
    obj_path = 'bodies/2_norm.obj'

    # Загружаем модель с учетом .mtl
    mesh = trimesh.load(obj_path, force='mesh')
    points = mesh.vertices

    # Переход в сферические координаты
    r, theta, phi = cartesian_to_spherical(points[:, 0], points[:, 1], points[:, 2])

    # Нормализованные углы
    u = theta / np.pi                 # θ ∈ [0, π] → u ∈ [0, 1]
    v = (phi + np.pi) / (2 * np.pi)  # φ ∈ [-π, π] → v ∈ [0, 1]

    # Аппроксимация
    degree_u, degree_v = 8, 8
    coeffs = fit_bernstein_surface(u, v, r, degree_u, degree_v)

    # Сетка для визуализации
    U, V = np.meshgrid(np.linspace(0, 1, 150), np.linspace(0, 1, 300))
    R = bernstein_surface(U, V, coeffs)

    # Обратно в декартовы координаты
    Theta = U * np.pi
    Phi = V * 2 * np.pi - np.pi
    X = R * np.sin(Theta) * np.cos(Phi)
    Y = R * np.sin(Theta) * np.sin(Phi)
    Z = R * np.cos(Theta)

    # Визуализация
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='lightblue', edgecolor='gray', linewidth=0.2, alpha=1.0)
    ax.set_title("Аппроксимация полиномами Берштейна", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
