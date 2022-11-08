import imageio
import matplotlib.pyplot as plt
import numpy as np
import os

FONTSIZE = 16
LINSPACE = np.linspace(-5, 5, 2)
CMAP     = 'jet'
FILENAME = 'fractal.gif'

# Mandelbrot fractal
def fractal(a, b, p=1000, s=4, num_iter=20):
    img = np.zeros((p, p))
    for i in range(p):
        x_0 = -s / 2 + i * s / p;
        for j in range(p):
            y_0 = -s / 2 + j * s / p;
            x, y, z = x_0, y_0, 0
            iter = 1
            while iter < num_iter:
                iter += 1
                x_1 = x ** 2 - y ** 2 + a
                y_1 = 2 * x * y + b
                x, y, z = x_1, y_1, x_1 ** 2 + y_1 ** 2
                if z < 4:
                    img[i, j] = np.sqrt(z)
                else:
                    break
    return img

filenames = []

# a+0i
for a in LINSPACE:
    plt.figure(figsize=(12, 12))
    plt.imshow(fractal(a, 0), cmap=CMAP)
    plt.title(f'a+bi = {a:.3f}+0.000i', fontsize=FONTSIZE)
    filename = f'a_{a:.3f}b_0.png'
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
    filenames.append(filename)

# 0+bi
for b in LINSPACE:
    plt.figure(figsize=(12, 12))
    plt.imshow(fractal(0, b), cmap=CMAP)
    if b >= 0:
        plt.title(f'a+bi = 0.000+{b:.3f}i', fontsize=FONTSIZE)
    else:
        plt.title(f'a+bi = 0.000{b:.3f}i', fontsize=FONTSIZE)
    filename = f'a_0b_{b:.3f}.png'
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
    filenames.append(filename)

# x+xi
for x in LINSPACE:
    plt.figure(figsize=(12, 12))
    plt.imshow(fractal(x, x), cmap=CMAP)
    if x >= 0:
        plt.title(f'a+bi = {x:.3f}+{x:.3f}i', fontsize=FONTSIZE)
    else:
        plt.title(f'a+bi = {x:.3f}{x:.3f}i', fontsize=FONTSIZE)
    filename = f'a_{x:.3f}b_{x:.3f}.png'
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
    filenames.append(filename)

# x-xi
for x in LINSPACE:
    plt.figure(figsize=(12, 12))
    plt.imshow(fractal(-x, x), cmap=CMAP)
    if x >= 0:
        plt.title(f'a+bi = {-x:.3f}+{x:.3f}i', fontsize=FONTSIZE)
    else:
        plt.title(f'a+bi = {-x:.3f}{x:.3f}i', fontsize=FONTSIZE)
    filename = f'a_{-x:.3f}b_{x:.3f}.png'
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
    filenames.append(filename)

# save to gif file
with imageio.get_writer(FILENAME, mode='I') as writer:
    for filename in filenames:
       image = imageio.imread(filename)
       writer.append_data(image)
       writer.append_data(image)

# remove tmp files
for filename in set(filenames):
    os.remove(filename)