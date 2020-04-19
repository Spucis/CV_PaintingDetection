from source import globals
import numpy as np
import cv2
import matplotlib.pyplot as plt

conf = globals.conf

def quantize(frame, nbin):
    histogram = np.zeros([256], np.float64)
    for h in range(frame.shape[0]):
        for w in range(frame.shape[1]):
            p = frame[h, w]
            histogram[p] += 1

    bin_size = 256 // nbin
    bins = np.zeros([nbin], np.uint8)
    for bin in range(bins.shape[0]):
        bins[bin] = (bin * bin_size) + np.argmax(histogram[bin * bin_size: (bin+1)*bin_size])

    ret = np.zeros(frame.shape, np.uint8)
    for h in range(frame.shape[0]):
        for w in range(frame.shape[1]):
            p = frame[h, w]
            ret[h, w] = bins[p // bin_size]

    return ret

def edge_detection(frame, debug = False, corners = False):
    print("Edge detection function.")

    #Gaussian filter
    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    #Median
    #dst = cv2.medianBlur(frame, 13)
    #Bilateral
    #dst = cv2.bilateralFilter(frame, 5, 5, 20)

    #Try threshold to eliminate white
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    #th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    #th3 = cv2.medianBlur(th3, 5)

    ret, th2 = cv2.threshold(gray, 0, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #thresholded_image = frame
    gray[gray < ret] = 0

    #Quantize
    q_frame = quantize(gray, 4)
    if debug:
        # cv2.imshow('Threshold', th3)
        cv2.imshow('Gray', gray)
        cv2.imshow('Quantized', q_frame)
        cv2.waitKey()
        cv2.destroyAllWindows()

    #ret, th2 = cv2.threshold(gray, 0, 200,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #thresholded_image = frame
    #thresholded_image[gray > ret] = [0,0,0]
    #thresholded_image[gray > 60] = [0,0,0]

    thresholded_image = q_frame

    if debug:
        cv2.imshow('Blurred image', blur)
        #cv2.imshow('Thresholded image - Value: {}'.format(ret), thresholded_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    #Canny

    factor = 7
    th = 100
    TH = th * factor

    # """
    v = np.median(thresholded_image.astype(np.uint8))
    # apply automatic Canny edge detection using the computed median
    th = int(max(0, (1.0 - 0.30) * v))
    TH = int(min(255, (1.0 + 0.30) * v))

    dst = thresholded_image

    #with aperture 5
    #th = 500
    #TH = 700
    # """

    #with aperture 3
    th = 100
    TH = 200
    mod_f = cv2.Canny(dst.astype(np.uint8), th, TH, apertureSize=3, L2gradient=True)

    if debug:
       cv2.imshow('Edge detection image', mod_f)
       cv2.waitKey()
       cv2.destroyAllWindows()

    #CORNER DETECTION
    if corners:
        mod_f = np.float32(mod_f)
        dst = cv2.cornerMinEigenVal(mod_f, 10) # , 5, 0.04)

        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)

    #conversione colore
    mod_f2 = cv2.cvtColor(mod_f.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if corners:
        # PRINT THE CORNERS
        #No threshold for corners
        #mod_f2[dst > 0] = [0, 0, 255]

        # Threshold for an optimal value, it may vary depending on the image. (original was 0.01 * dst.max())
        #mod_f2[dst > 0.3* dst.max()]= [0, 0, 255]

        # Threshold between two values
        mod_f2[ np.where((0.1* dst.max() <  dst) & (dst< 0.15 * dst.max()))] = [0, 0, 255]

        if debug:
            cv2.imshow('Corner detection image', mod_f2)
            cv2.waitKey()
            cv2.destroyAllWindows()

    print("Fine edge_detection.")

    return mod_f2.astype(np.uint8)
