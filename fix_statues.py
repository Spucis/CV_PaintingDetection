import os
import cv2
import numpy as np

src_path = "/home/daniele/Immagini/statue/"
dst_path = "/home/daniele/Immagini/fixed/"

counter = 1

for name in os.listdir(src_path):
    img = cv2.imread(src_path + name)

    # rimuove interfaccia grafica di ubuntu
    # img = img[132:, 130:, :]

    # cerca gli indici dove inizia l'img senza bande nere
    ri = 0
    ci = 0
    found = False
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if (not found and np.sum(img[r, c, :]) != 0):
                ri = r
                ci = c
                found = True
                break
        if (found):
            break

    rf = -ri
    cf = -ci

    # ritaglio l'img
    if (ri == rf):
        if (ci == cf):
            dst_img = img[ri:, ci:, :]
        else:
            dst_img = img[ri:, ci:cf, :]
    else:
        if (ci == cf):
            dst_img = img[ri:rf, ci:, :]
        else:
            dst_img = img[ri:rf, ci:cf, :]

    cv2.imwrite(dst_path + name, dst_img)
    print("IMG_{}".format(counter))
    counter += 1
