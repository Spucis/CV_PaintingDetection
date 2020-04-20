from source import globals

import numpy as np
import cv2
import matplotlib.pyplot as plt

conf = globals.conf

def quantize(frame, nbin):
    """
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
    """
    #Uniform quatization
    #return np.round(frame*(nbin/255))*(255/nbin)

    # USING KMEANS

    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning

    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    img = frame#image
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = nbin
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)

    #res2 = cv2.cvtColor(res2, cv2.COLOR_LAB2BGR)
    
    return res2


def edge_detection(frame, debug = False, corners = False, frame_number = 0):

    selected_frame = 0
    blur = frame

    #Gaussian filter
    blur = cv2.GaussianBlur(blur, (9, 9), 0)

    """
        Median -- UTILE PER LE STATUE?
        
        Sembra che funzioni meglio sul riconoscimento degli edge delle statue.
    """
    #blur = cv2.medianBlur(blur, 11)

    #Bilateral
    #blur = cv2.bilateralFilter(blur, 5, 5, 20)



    if debug and selected_frame == frame_number:
        cv2.imshow('Blurred image', blur)
        cv2.waitKey()
        cv2.destroyAllWindows()

    #Try threshold to eliminate white
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    #th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    #th3 = cv2.medianBlur(th3, 5)

    #ret, th2 = cv2.threshold(gray, 0, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #thresholded_image = frame
    #gray[gray < ret] = 0

    #Quantize
    #q_frame = quantize(gray, 4)
    #q_frame = quantize(blur, 3)
    if debug and selected_frame == frame_number:
        # cv2.imshow('Threshold', th3)
        cv2.imshow('Gray', gray)
        #cv2.imshow('Quantized', q_frame)
        cv2.waitKey()
        cv2.destroyAllWindows()

    #ret, th2 = cv2.threshold(gray, 0, 200,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #thresholded_image = frame
    #thresholded_image[gray > ret] = [0,0,0]
    #thresholded_image[gray > 60] = [0,0,0]

    #thresholded_image = q_frame
    """
    ret = 0
    if debug:
        cv2.imshow('Thresholded image - Value: {}'.format(ret), thresholded_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    """

    #dst = thresholded_image

    dst = blur


    #with aperture 3
    # thresholds with K-MEANS
    #th = 175
    #TH = 196

    # Canny
    # Threshold ottimali per aperture 5
    # th = 550
    # TH = 1750

    # threshold con un po' più di rumore
    th = 500
    TH = 1300

    mod_f = cv2.Canny(dst.astype(np.uint8), th, TH, apertureSize=5, L2gradient=True)

    vis = frame.copy()
    vis = np.uint8(vis / 2.)
    vis[mod_f != 0] = (0, 255, 0)

    if debug and selected_frame == frame_number:
       cv2.imshow('Edge detection image', mod_f)
       cv2.imshow('Edge detection image', vis)
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

        if debug and selected_frame == frame_number:
            cv2.imshow('Corner detection image', mod_f2)
            cv2.waitKey()
            cv2.destroyAllWindows()



    #return mod_f2.astype(np.uint8)
    return vis