import numpy as np
from quaddtype import QuadPrecDType, QuadPrecision
import matplotlib.pyplot as plt

def get_color(t, interior_t):
    epsilon = QuadPrecision("1e-10")

    if abs(t - QuadPrecision(1.0)) < epsilon:
        value = int(255 * float(interior_t))
        return np.array([value, value, value], dtype=np.uint8)

    t = np.power(t, 0.5)
    t = np.mod(t * 20, 1.0)

    if t < 0.16:
        return np.array([0, int(255 * (t / 0.16)), int(128 + 127 * (t / 0.16))], dtype=np.uint8)
    elif t < 0.33:
        return np.array([0, 255, int(255 * (1 - (t - 0.16) / 0.17))], dtype=np.uint8)
    elif t < 0.5:
        return np.array([int(255 * ((t - 0.33) / 0.17)), 255, 0], dtype=np.uint8)
    elif t < 0.66:
        return np.array([255, int(255 * (1 - (t - 0.5) / 0.16)), 0], dtype=np.uint8)
    elif t < 0.83:
        return np.array([255, 0, int(255 * ((t - 0.66) / 0.17))], dtype=np.uint8)
    else:
        return np.array([int(255 * (1 - (t - 0.83) / 0.17)), 0, int(128 * ((t - 0.83) / 0.17))], dtype=np.uint8)

def iterate_and_compute_derivatives(c, max_iter):
    z = 0
    dz = 1
    dc = 0
    dzdz = 0

    for _ in range(max_iter):
        dzdz = 2 * (z * dzdz + dz * dz)
        dz = 2 * z * dz + dc
        z = z * z + c
        dc = 1

    return z, dz, dc, dzdz

def estimate_interior_distance(c, max_iter):
    z, dz, dc, dzdz = iterate_and_compute_derivatives(c, max_iter)

    dz_abs_sq = np.abs(dz) ** 2
    numerator = 1 - dz_abs_sq

    denominator = np.abs(dc * dz + dzdz * z * dc)

    return numerator / denominator

def mandelbrot(c, max_iter, radius2):
    z = 0
    for i in range(max_iter):
        z = z * z + c
        if np.abs(z) ** 2 > radius2:
            log_zn = np.log(np.abs(z))
            nu = np.log(log_zn / np.log(2)) / np.log(2)
            return i + 1 - nu, z
    return max_iter, z

def mandelbrot_set(width, height, max_iter, center_r, center_i, zoom):
    radius = 2.0
    radius2 = radius * radius
    zoom_q = 1 / zoom

    x = np.linspace(center_r - radius / zoom, center_r + radius / zoom, width)
    y = np.linspace(center_i - radius / zoom, center_i + radius / zoom, height)
    c = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    smooth_iter, final_z = np.frompyfunc(lambda c: mandelbrot(c, max_iter, radius2), 1, 2)(c)
    smooth_iter = smooth_iter.astype(np.float64)
    final_z = final_z.astype(np.complex128)

    img = np.zeros((height, width, 3), dtype=np.uint8)

    interior_mask = smooth_iter == max_iter
    interior_c = c[interior_mask]
    interior_distance = np.frompyfunc(lambda c: estimate_interior_distance(c, max_iter), 1, 1)(interior_c)
    interior_distance = interior_distance.astype(np.float64)
    interior_t = interior_distance - np.floor(interior_distance)

    exterior_mask = ~interior_mask
    t = smooth_iter[exterior_mask] / max_iter

    interior_colors = np.array(list(map(lambda t: get_color(1.0, t), interior_t)))
    exterior_colors = np.array(list(map(lambda t: get_color(t, 0.0), t)))

    img[interior_mask] = interior_colors
    img[exterior_mask] = exterior_colors

    return img

def plot_mandelbrot(width, height, max_iter, center_r, center_i, zoom):
    img_array = mandelbrot_set(width, height, max_iter, center_r, center_i, zoom)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_array)
    plt.axis('off')
    plt.title(f'Mandelbrot Set (zoom: {zoom}, center: {center_r} + {center_i}i, iterations: {max_iter}, dtype: numpy.float64)')
    plt.show()

if __name__ == "__main__":
    width = 800
    height = 800
    max_iter = 1000
    center_r = -0.75
    center_i = 0.0
    zoom = 1.0

    plot_mandelbrot(width, height, max_iter, center_r, center_i, zoom)