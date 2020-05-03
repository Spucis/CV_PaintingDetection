from source import globals

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

from json import *
import random

conf = globals.conf


def show_frame(d_frames):
    for label, frame in d_frames.items():
        cv2.imshow(label, frame)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Ritorna l'intersezione di un segmento a0-a1 e un altro b0-b1
def calculateIntersection(a0, a1, b0, b1):
    if a0 >= b0 and a1 <= b1: # Contained
        intersection = a1 - a0
    elif a0 < b0 and a1 > b1: # Contains
        intersection = b1 - b0
    elif a0 < b0 and a1 > b0: # Intersects right
        intersection = a1 - b0
    elif a1 > b1 and a0 < b1: # Intersects left
        intersection = b1 - a0
    else: # No intersection (either side)
        intersection = 0

    return intersection

def draw_ROI(frame, ROI, text=None, color=(0,0,255), text_color=(0,0,0), copy=False, only_text=False):

    if copy:
        img = frame.copy()
    else:
        img = frame

    x = ROI[0]
    y = ROI[1]
    if only_text:
        w = 0
        h = 0
    else:
        w = ROI[2]
        h = ROI[3]

    #LABEL
    if text:
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = .5
        thickness = 1
        textSize, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)

        # To show the label also when at the upper corner of the image
        label_rect_h = 5
        label_offset = -textSize[1] - label_rect_h if y - textSize[1] - label_rect_h > 0 else textSize[1] + label_rect_h
        text_offset = -label_rect_h if label_offset < 0 else textSize[1] + 2  # 1 of the border itself and 1 of spacing

        img = cv2.rectangle(img, (x, y), (x + textSize[0], y + label_offset), color, cv2.FILLED)
        img = cv2.putText(img, text, (x, y + text_offset), fontFace, fontScale, text_color, thickness)

    return cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

def ccl_detection(or_frame, gray_frame, frame, frame_number):



    """
        Marco
        Provo cv.connectedComponents(	image[, labels[, connectivity[, ltype]]]	)
    """

    # Parametri delle modifiche apportate all'immagine. Verranno elencate nell'immagine finale.
    # nome -> valore
    params = {}
    out = frame

    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    out = cv2.drawContours(out, contours, -1, (255,255,255), 3)
    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #show_frame({"CANNY CONTOURS DILATED": out})

    #print(hierarchy.shape)
    # Tolgo la prima dimensione -- Da controllare, in generale
    hierarchy = hierarchy[0]

    i = 0
    ROIs = []

    # CLEANING ROIs RULES
    # 1. se il rapporto di aspetto è più di 'ratio_max' volte, considero la ROI non ammissibile
    # 2. se l'area è minore di min_area, non è ammissibile
    # 3. se è contenuto in un'altra ROI o contiene un'altra ROI, quello con area minore non è ammissibile -- TODO
    # 4. (opzionale?) se l'overlap supera una certa soglia, la ROI con area minore non è ammissibile -- TODO
    ratio_max = 4.5
    global_area = frame.shape[0]*frame.shape[1]
    min_area = 0.03 * global_area #almeno una % dell'area totale

    img = or_frame.copy()

    for component in zip(contours, hierarchy):
        currentContour = component[0]
        #currentHierarchy = component[1]
        x,y,w,h = cv2.boundingRect(currentContour)

        #hull.append(cv2.convexHull(currentContour))



        cleaning_boxes = True
        if cleaning_boxes:
            #1.
            feasible_ratio = abs(h) < abs(ratio_max * (w)) and abs(w) < abs(ratio_max * (h))
            area = w * h
            #2.
            feasible_area = area > min_area

            params["Maximum ROI Aspect ratio"] = ratio_max
            params["Minimun ROI Area"] = min_area

            if feasible_ratio and feasible_area:
                ROIs.append([(x, y), (x + w, y + h), w, h])
            else:
                continue
                text += "RATIO:{}-L_AREA:{}({})".format((max(h,w)/min(h,w)),h * w,h * w / (frame.shape[0] * frame.shape[1]) * 100)
                img = draw_ROI(img, (x, y, w, h), color=(255, 0, 0))


        else: #NO CLEANING BOXES
            img = cv2.rectangle(or_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        i += 1
    i = 0

    # ARLGORITMO QUADRATICO RISPETTO AL NUMERO DELLE ROI, MIGLIORABILE? TODO
    max_overlap = 0.6
    params["Max ROI overlap"] = max_overlap

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # True ROIs array
    trueROIs = []

    global_thres, _ = cv2.threshold(gray_frame, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for roi in ROIs:
        curr_roi = []

        # Se "roi" verrà riconosciuto come contenuto in altre ROI, sarà non drawable
        drawable = True
        j = 0
        for roi2 in ROIs:

            if roi[0][0] == roi2[0][0] and roi[0][1] == roi2[0][1] and roi[1][0] == roi2[1][0] and roi[1][1] == roi2[1][1]:
                j += 1
                continue

            width = calculateIntersection(roi[0][0],roi[1][0], roi2[0][0], roi2[1][0])
            height = calculateIntersection(roi[0][1],roi[1][1], roi2[0][1], roi2[1][1])
            area = width * height
            percent = area / (roi[2] * roi[3])
            #print("C{} over C{} overlap: {}".format(i,j,percent))

            # must go over the max_overlap AND minor area
            if percent >= max_overlap and roi[2] * roi[3] < roi2[2] * roi2[3]:
                drawable = False
                #print("C{} over C{} overlap: {} ---- BREAK!".format(i, j, percent))
                break
            j += 1

        text = "C{}".format(i)

        x = roi[0][0]
        y = roi[0][1]
        w = roi[2]
        h = roi[3]

        # THRESHOLD
        thres, thres_image = cv2.threshold(gray_frame[y:y + h, x:x + w], 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text += "-G_TH:{}-L_TH:{}-L_AREA:{}({})".format(global_thres, thres, h * w,
                                                        h * w / (frame.shape[0] * frame.shape[1]) * 100)

        if thres > global_thres: # Escludo le roi che hanno local thresholding di otsu più alta della global.
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            drawable = False

        if not drawable:
            img = draw_ROI(img, (x,y,w,h), text=text, color=(0, 255, 0))
            continue
        else:
            img = draw_ROI(img, (x,y,w,h), text=text, color=(0, 0, 255))
            trueROIs.append((x, y, w, h))

        #PRINT GLOBAL AREA
        glob_text = "GLOBAL AREA: {}".format(frame.shape[0] * frame.shape[1])
        img = draw_ROI(img, (10, 10, 0, 0), text = glob_text, color=(0,0,255), only_text=True )



        show_plots = False
        total_var = 0
        if frame_number == 150 and show_plots:
            # in LAB = L a b
            color = ('r', 'g', 'b')
            color = ('r')
            total_var = []
            for index, col in enumerate(color):
                histr = cv2.calcHist(gray_frame[y:y + h, x:x + w], [index], None, [256], [0, 256])

                #print(histr.shape)
                #histr = histr / (w * h)
                var = np.std(histr, axis=0)
                total_var.append(var)

                #print("Var of {} in ROI {}, frame {}:{}".format(col,i,frame_number,var))
                plt.plot(histr, color=col)
                plt.title("Histogram of colors ROI{}_frame{}".format(i, frame_number))
                plt.xlim([0, 256])
            plt.show()
            print("Average var of image: {}".format(total_var))

            thres, thres_image = cv2.threshold(gray_frame[y:y + h, x:x + w], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            show_frame({"GRAY_FRAME": gray_frame})
            show_frame({"OTSU_FRAME_TH:{}".format(thres): thres_image})

        """
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = .5
        thickness = 1
        textSize, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)

        # To show the label also when at the upper corner of the image
        label_rect_h = 5
        label_offset = -textSize[1] - label_rect_h if y - textSize[1] - label_rect_h > 0 else textSize[1] + label_rect_h
        text_offset = -label_rect_h if label_offset < 0 else textSize[1] + 2  # 1 of the border itself and 1 of spacing

        img = cv2.rectangle(img, (x, y), (x + textSize[0], y + label_offset), (0, 0, 255), cv2.FILLED)
        img = cv2.putText(img, text, (x, y + text_offset), fontFace, fontScale, (0, 0, 0), thickness)
        """
        i += 1

    final_string = ""
    for name, value in params.items():
        final_string += "{}: {}; ".format(name, value)

    #show_frame({"Relevant ROIs: " + final_string : img})
    return img, trueROIs

def edge_detection(frame, debug = False, frame_number = 0):

    selected_frame = 0
    d_frames = {}

    # To apply more blur without modifing frame
    blur = frame

    # Gaussian filter
    blur = cv2.GaussianBlur(blur, (9, 9), 0)
    """
        Median -- UTILE PER LE STATUE?
        
        Sembra che funzioni meglio sul riconoscimento degli edge delle statue.
    """
    #blur = cv2.medianBlur(blur, 11)
    # Bilateral
    # blur = cv2.bilateralFilter(blur, 5, 5, 20)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    if debug and selected_frame == frame_number:
        d_frames['Blurred image'] = blur
        d_frames['Gray'] = gray

    dst = blur
    # Canny
    # Threshold ottimali per aperture 5
    # th = 550
    # TH = 1750
    # Threshold con un po' più di rumore
    th = 500
    TH = 1300

    #Threshold rumorose
    th = 400
    TH = 800

    mod_f = cv2.Canny(dst.astype(np.uint8), th, TH, apertureSize=5, L2gradient=True)

    # Frame with the edges highlighted in green
    vis = frame.copy()
    vis = np.uint8(vis / 2.)
    vis[mod_f != 0] = (0, 255, 0)

    if debug and selected_frame == frame_number:
        d_frames['Vis'] = vis
        d_frames['Canny_OutPut'] = mod_f

    # conversione colore
    mod_f2 = cv2.cvtColor(mod_f.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    show_frame(d_frames)

    return gray, vis, mod_f

def keypoints_detection(frame, show=True):

    start = time.time()
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(frame, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(frame, kp)
    # draw only keypoints location,not size and orientation
    kp_frame = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
    if show:
        show_frame({"KEY_POINTS_elapsedTime: {}".format(time.time()-start): kp_frame})

    return kp_frame

def find_keypoint(img):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(img,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        # draw only keypoints location,not size and orientation
        # in una funzione a parte per separare le cose?

        return kp, des

# ritorna un array di matches (ritornato da BFMatch) e un array di nomi di img per ogni roi
def paint_retrival(frame, ROIs, kp, des):
    pass

# prendo l'array di matches per ogni ROI e stampo a video i quadri raddrizzati
def paint_rectification(frame, ROIs, kp, des, matches):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # quadro centrale
    d1 = des['085.png']

    # ROI del quadro 085
    roi = ROIs[1]

    crop = frame[roi[1]:roi[3], roi[0]:roi[2]]

    k, d2 = find_keypoint(crop)

    # Match descriptors.
    match = bf.match(d1, d2)
    pass