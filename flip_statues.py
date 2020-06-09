import os
import cv2

src_statue = "/home/daniele/Immagini/statue/"
src_labels = "/home/daniele/Immagini/labels/"
dst_statue = "/home/daniele/Immagini/statue_flip/"
dst_labels = "/home/daniele/Immagini/labels_flip/"

suffix = "_flip"

# immagini
counter = 1
for name in os.listdir(src_statue):
    img = cv2.imread(src_statue + name)

    """
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """

    dst_img = cv2.flip(img, 1)

    """
    cv2.imshow('flip', dst_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """

    title, ext = os.path.splitext(name)
    cv2.imwrite("{}{}{}{}".format(dst_statue, title, suffix, ext), dst_img)

    print("IMG_{}".format(counter))
    counter += 1

# labels
counter = 1
for name in os.listdir(src_labels):
    title, ext = os.path.splitext(name)
    with open(src_labels + name, "r") as src:
        with open("{}{}{}{}".format(dst_labels, title, suffix, ext), "w") as dst:
            lines = src.readlines()
            for row in lines:
                vals = row.split(" ")
                cx = 1 - float(vals[1])
                # class cx cy w h
                dst.write("{} {} {} {} {}".format(vals[0], cx, vals[2], vals[3], vals[4]))
            dst.close()
        src.close()

    print("TXT_{}".format(counter))
    counter += 1
