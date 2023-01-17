# -*- coding:utf-8 -*-

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#画像の読み込み
im = Image.open("../data/image.png")

im_list = np.asarray(im)
plt.imshow(im_list)

def plot_rect(sy, sx, height, width, c):    
    (y_min, x_min) = (sy, sx)
    (y_max, x_max) = (sy + height, sx + width)
    
    plt.plot([x_min, x_max], [y_min, y_min], '-', color=c, lw=1)
    plt.plot([x_min, x_max], [y_max, y_max], '-', color=c, lw=1)
    plt.plot([x_min, x_min], [y_min, y_max], '-', color=c, lw=1)
    plt.plot([x_max, x_max], [y_min, y_max], '-', color=c, lw=1)

plt.plot([0, 256], [128, 128], '-', lw=1)
plt.plot([128, 128], [0, 256], '-', lw=1)

plot_rect(3, 65, 129, 129, 'b')
plt.plot([140], [132], 'r.', ms=3)
plot_rect(169, 30, 129 - 64, 129 + 64, 'r')

#表示
plt.savefig("image-plt.png", dpi=500)
