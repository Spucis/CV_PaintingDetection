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

# Connected Components Labeling
# Tranforms a binary input image into a simbolic one, in which all the pixel of the same components
# have the same label
def ccl_detection(or_frame, gray_frame, frame, frame_number):

    """
        Marco
        Provo cv.connectedComponents(	image[, labels[, connectivity[, ltype]]]	)
    """

    # Parametri delle modifiche apportate all'immagine. Verranno elencate nell'immagine finale.
    # nome -> valore
    params = {}
    out = frame

    """
    # Dilatation of the borders to connect near edges
    input = frame.copy()
    H, W = input.shape  # input -> (n, iC, H, W).. so i have to expand dims like: (n, 1, iC, H, W)
    k_dim = 5
    params["Dilation kernel dim"] = "{}x{}".format(k_dim, k_dim)
    kernel = np.full((k_dim, k_dim), 1)

    # (1, oC, iC, kH, kW)
    kH = kernel.shape[0]
    kW = kernel.shape[1]

    out = np.zeros((H - kH + 1, W - kW + 1))

    start_conv = time.time()
    for i in range(H - kH + 1):
        for j in range(W - kW + 1):
            new_input = input[i:i + kH, j:j + kW]
            out[i, j] = np.sum(new_input * kernel)

    print("Conv time: {}".format(time.time() - start_conv))

    out[out > 0] = 1
    show_frame({"Dilated Canny. Kernel: {}x{}".format(k_dim,k_dim) : out})
    out = out.astype(np.uint8)
    """

    """ CONNECTED COMPONENTS AND LABELING
    
    threshold_frame = cv2.adaptiveThreshold(gray_frame ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 1)
    
    input = frame.copy()
    H, W = input.shape  # input -> (n, iC, H, W).. so i have to expand dims like: (n, 1, iC, H, W)
    kernel = np.full((17,17), 1)

    # (1, oC, iC, kH, kW)
    kH = kernel.shape[0]
    kW = kernel.shape[1]

    out = np.zeros((H - kH + 1, W - kW + 1))

    for i in range(H - kH + 1):
        for j in range(W - kW + 1):
            new_input =input[i:i + kH, j:j + kW]
            out[i, j] = np.sum(new_input * kernel)

    out[out > 0] = 1
    out = out.astype(np.uint8)
    print("OUT: {}\nshape: {}, kernel: {}".format(out, out.shape, kernel))
    components, labels = cv2.connectedComponents(out)
    print("Num. components: {}".format(components))

    #show_frame({"Dilatated CANNY" : out, "Normal CANNY": frame})
    show_frame({"Thresholded_image": threshold_frame})

    labeled_frame = or_frame.copy()
    colors = []
    example_point = []

    for label in range(components):
        colors.append([random.randint(0,255),random.randint(0,255),random.randint(0,255)])
        #Trovo le ccordinate di un punto del cluster
        found = False
        for r in range(labels.shape[0]):
            for c in range(labels.shape[1]):
                if labels[r, c] == label:
                    example_point.append((r, c))
                    found = True
                if found:
                    break
            if found:
                break
    colors[0] = [0,0,0]
    for r in range(labels.shape[0]):
        for c in range(labels.shape[1]):
            labeled_frame[r,c] = colors[labels[r,c]]

    for label in range(components):
        text = "L{}".format(label)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = .4
        thickness = 1

        textSize, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
        x = 0
        y = label * textSize[1] + textSize[1] + 1
        labeled_frame = cv2.rectangle(labeled_frame, (x, y), (x + textSize[0], y - textSize[1] - 5),
                                      colors[label], cv2.FILLED)
        labeled_frame = cv2.putText(labeled_frame, text, (x, y - 5), fontFace, fontScale, (255, 255, 255), thickness)


    show_frame({"LABELED_FRAME": labeled_frame})

    labeled_frame_2 = cv2.cvtColor(labeled_frame, cv2.COLOR_BGR2GRAY)
    labeled_frame_2[labeled_frame_2 > 0] = 1
    """

    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    out = cv2.drawContours(out, contours, -1, (255,255,255), 3)
    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #show_frame({"CANNY CONTOURS DILATED": out})

    #print(hierarchy.shape)
    # Tolgo la prima dimensione -- Da controllare, in generale
    hierarchy = hierarchy[0]
    #for i in range(hierarchy.shape[0]):
    #    mod_f = cv2.drawContours(or_frame.copy(), contours, hierarchy[i][0], (0, 255, 0))
        # per adesso tolgo il frame normale 'Canny_Frame':m_frame,
    #    show_frame({'CCL_Frame_ContourHierarchy: {}'.format(i): mod_f})
    i = 0
    #hull = []
    ROIs = []

    """ CLEANING ROIs RULES """
    # 1. se il rapporto di aspetto è più di 'ratio_max' volte, considero la ROI non ammissibile
    # 2. se l'area è minore di min_area, non è ammissibile
    # 3. se è contenuto in un'altra ROI o contiene un'altra ROI, quello con area minore non è ammissibile -- TODO
    # 4. (opzionale?) se l'overlap supera una certa soglia, la ROI con area minore non è ammissibile -- TODO
    ratio_max = 4.5
    min_area = 4000

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
                text = "C{} DISCARDED".format(i)
                #img = cv2.rectangle(or_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        else: #NO CLEANING BOXES
            img = cv2.rectangle(or_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        """
        CONSIDER HIERARCHIES
        
        if currentHierarchy[0] > 0:
            _x, _y, _w, _h = cv2.boundingRect(contours[currentHierarchy[0]])
            # these are the innermost child components
            img = cv2.rectangle(or_frame, (_x-1, _y-1), (_x + _w + 1, _y + _h + 1), (0, 255, 0), 2)

        if currentHierarchy[1] > 0:
            _x, _y, _w, _h = cv2.boundingRect(contours[currentHierarchy[1]])
            # these are the innermost child components
            img = cv2.rectangle(or_frame, (_x-1, _y-1), (_x + _w + 1, _y + _h + 1), (255, 0, 0), 2)
        """
        """
        #print(x,y,w,h)
        print("{}".format(currentHierarchy))

        if currentHierarchy[2] < 0:
            # these are the innermost child components
            img = cv2.rectangle(or_frame,(x,y),(x+w,y+h),(0,0,255),2)
        elif currentHierarchy[3] < 0:
            # these are the outermost parent components
            img = cv2.rectangle(or_frame,(x,y),(x+w,y+h),(0,255,0),2)

        #show_frame({'CCL_Frame_ContourHierarchy: {}'.format(i): hull})
        
        """
        i += 1
    """
    CONVEX HULL DRAWING
    
    drawing = np.uint8(or_frame.copy()/2.)

    for i in range(len(contours)):
        color = (0, 0, 255)  # blue - color for convex hull
        # draw ith contour
        #cv2.drawContours(drawing, contours, i, color_contours, 1, 8)#, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)

    show_frame({"convex_hull": drawing})
    """

    i = 0
    # ARLGORITMO QUADRATICO RISPETTO AL NUMERO DELLE ROI, MIGLIORABILE? TODO

    max_overlap = 0.6
    params["Max ROI overlap"] = max_overlap

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # True ROIs array
    trueROIs = []

    for roi in ROIs:
                   # 0[0, 1]    1[0 , 1]      2      3
        # roi -> [ (x00, y00) , (x01, y01) , width, height ]

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

        """ DRAW KEYPOINTS
        start = time.time()

        # find the keypoints with ORB
        kp = orb.detect(or_frame, None)
        # compute the descriptors with ORB
        kp, des = orb.compute(or_frame, kp)
        # draw only keypoints location,not size and orientation
        or_frame = cv2.drawKeypoints(or_frame, kp, None, color=(0, 255, 0), flags=0)
        """

        x = roi[0][0]
        y = roi[0][1]
        w = roi[2]
        h = roi[3]

        if not drawable:
            #img = cv2.rectangle(or_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue
        else:
            img = cv2.rectangle(or_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            trueROIs.append((x, y, w, h))

        show_plots = True
        total_var = 0
        if frame_number == 150 and show_plots:
            color = ('b', 'g', 'r')
            total_var = []
            for index, col in enumerate(color):
                histr = cv2.calcHist([or_frame[y:y + h, x:x + w, :]], [index], None, [256], [0, 256])
                #print(histr.shape)
                var = np.std(histr, axis=0)
                total_var.append(var/(w*h))

                #print("Var of {} in ROI {}, frame {}:{}".format(col,i,frame_number,var))
                plt.plot(histr, color=col)
                plt.title("Histogram of colors ROI{}_frame{}".format(i, frame_number))
                plt.xlim([0, 256])
            plt.show()
            print("Average var of image: {}".format(np.std(total_var)))

            show_frame({"ROI{}".format(i): or_frame[y:y + h, x:x + w, :]})

        #if np.var(total_var) > 10:
        #    cv2.rectangle(or_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #   show_frame({"SOMETHING?": or_frame})



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