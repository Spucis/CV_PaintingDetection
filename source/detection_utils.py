from source import globals

import numpy as np
import cv2
import time
import random
import matplotlib.pyplot as plt

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

        texts = text.split("\n")
        textSizes = []
        for curr_text in texts:
            textSize, baseLine = cv2.getTextSize(curr_text, fontFace, fontScale, thickness)
            textSizes.append(textSize)

        # To show the label also when at the upper corner of the image
        label_rect_h = 5
        label_offset = -textSizes[0][1] - label_rect_h if y - textSizes[0][1]*len(texts) - label_rect_h > 0 else textSizes[0][1] + label_rect_h
        text_offset = -label_rect_h if label_offset < 0 else textSizes[0][1] + 2  # 1 of the border itself and 1 of spacing


        for index, curr_text in enumerate(texts):
            img = cv2.rectangle(img, (x, y + label_offset*(index)), (x + textSizes[index][0], y + label_offset*(index+1)), color, cv2.FILLED)
        for index, curr_text in enumerate(texts):
            add_offset = label_offset*(index) #if label_offset < 0 else label_offset*(index)
            img = cv2.putText(img, curr_text, (x, y + text_offset + add_offset), fontFace, fontScale, text_color, thickness)

    return cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

def ccl_detection(or_frame, gray_frame, frame, frame_number, otsu_opt_enabled=False):
    """
        Marco
        Provo cv.connectedComponents(	image[, labels[, connectivity[, ltype]]]	)
    """
    #gray_frame = cv2.bilateralFilter(gray_frame, 7, 7, 150)
    #
    #gray_frame = cv2.medianBlur(gray_frame, 5)
    #gray_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #show_frame({"adaptive": gray_frame})
    #global_thres, th_frame = cv2.threshold(gray_frame, 0, 255,
    #                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #return gray_frame, None
    # Parametri delle modifiche apportate all'immagine. Verranno elencate nell'immagine finale.
    # nome -> valore

    img = or_frame.copy()

    #Dilation and erosion phase over Canny, to connect the borders
    dilations = 3
    dilate_size = 3
    dilate_kernel = np.full((dilate_size, dilate_size), 1, dtype=np.uint8)
    cycles = 1
    new_canny = frame.copy()

    for _ in range(cycles):
        new_canny = cv2.dilate(new_canny, dilate_kernel, iterations=dilations)
        # show_frame({"dilation_{}ites".format(dilations): new_canny, "old": frame})
        new_canny = cv2.erode(new_canny, dilate_kernel, iterations=dilations)
        # show_frame({"erode_{}ites".format(dilations): new_canny})
    #show_frame({"cycles_Canny": new_canny, "old": frame})
    #return new_canny, None

    # Determine the roi by finding the connected components in dilated and eroded canny
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(new_canny)

    for label in range(1, num_labels):
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        label_x = stats[label, cv2.CC_STAT_LEFT]
        label_y = stats[label, cv2.CC_STAT_TOP]
        label_w = stats[label, cv2.CC_STAT_WIDTH]
        label_h = stats[label, cv2.CC_STAT_HEIGHT]
        centroid = (int(round(centroids[label, 0])), int(round(centroids[label, 1])))
        #print("x{}\ny{}\nw{}\nh{}\ncent.{}".format(label_x, label_y, label_w, label_h, centroid))
        img = cv2.drawMarker(img, centroid, color)
        cv2.rectangle(img,(label_x, label_y), (label_x+label_w, label_y+label_h), color, 2)
        cv2.arrowedLine(img, centroid, (label_x, label_y), color)

    #show_frame({"connectedComponents_BBOX": img})
    #return img, None
    params = {}
    """
    out = frame.copy()
    contours_iters = 1

    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for iter in range(contours_iters):
        new_out = np.zeros(out.shape, dtype=np.uint8)
        contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(new_out, contours, -1, (255,255,255), 2)
        contours, hierarchy = cv2.findContours(new_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        new_out = np.zeros(out.shape, dtype=np.uint8)
        cv2.drawContours(new_out, contours, -1, (255, 255, 255), 1)
        out = new_out.copy()
        #show_frame({"Contours_iter{}".format(iter): out, "Original CANNY": frame})

    #show_frame({"CANNY CONTOURS DILATED": out})

    #print(hierarchy.shape)
    # Tolgo la prima dimensione -- Da controllare, in generale
    hierarchy = hierarchy[0]
    """

    i = 0
    ROIs = []

    # CLEANING ROIs RULES
    # 1. se il rapporto di aspetto è più di 'ratio_max' volte, considero la ROI non ammissibile
    # 2. se l'area è minore di min_area, non è ammissibile
    # 3. se è contenuto in un'altra ROI o contiene un'altra ROI, quello con area minore non è ammissibile (solo se area maggiore è >50% del frame)
    # 4. (opzionale?) se l'overlap supera una certa soglia, la ROI con area minore non è ammissibile
    ratio_max = 3
    global_area = frame.shape[0]*frame.shape[1]
    min_area = 0.015 * global_area #almeno una % dell'area totale
    img = or_frame.copy()

    for label in range(1, num_labels):

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        cleaning_boxes = True
        if cleaning_boxes:
            # 1.
            feasible_ratio = abs(w) < abs(ratio_max * (h)) and abs(h) < abs(ratio_max * (w))
            area = w * h
            # 2.
            feasible_area = area > min_area

            params["Maximum ROI Aspect ratio"] = ratio_max
            params["Minimun ROI Area"] = min_area

            if feasible_ratio and feasible_area:
                ROIs.append([(x, y), (x + w, y + h), w, h])
            else:
                #text = "RATIO:{:.2}-L_AREA:{}({:.2%})".format((max(h, w) / min(h, w)), h * w,h * w / (frame.shape[0] * frame.shape[1]))
                #img = draw_ROI(img, (x, y, w, h), color=(255, 0, 0), text=text)
                continue

    i = 0

    # ARLGORITMO QUADRATICO RISPETTO AL NUMERO DELLE ROI, MIGLIORABILE? TODO
    max_overlap = 0.80
    params["Max ROI overlap"] = max_overlap

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # True ROIs array
    trueROIs = []
    big_ROI_discarded = []

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
            if percent >= max_overlap and roi[2] * roi[3] < roi2[2] * roi2[3] and roi2[2]*roi2[3] < 0.5*global_area:
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

        # text += "-G_TH:{}-L_TH:{}-L_AREA:{}({})".format(global_thres, thres, h * w,
        #                                               h * w / (frame.shape[0] * frame.shape[1]) * 100)

        otsu_factor = 1.3 #1.3
        if otsu_opt_enabled and thres >  otsu_factor * global_thres: # # Escludo le roi che hanno local thresholding di otsu più alta della global.
            img = draw_ROI(img, (x-2, y-2, w+4, h+4), text=text+"G_TH{}-ROI_TH{}".format(global_thres, thres), text_color=(255, 255, 255), color=(255, 0, 0))
            drawable = False

        if not drawable:
            img = draw_ROI(img, (x,y,w,h), text=text, color=(0, 255, 0))
            continue
        else:
            img = draw_ROI(img, (x,y,w,h), text=text, color=(0, 0, 255))
            trueROIs.append((x, y, w, h))

        # PRINT GLOBAL AREA
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

        i += 1

    final_string = ""
    for name, value in params.items():
        final_string += "{}: {}; ".format(name, value)



    # show_frame({"Relevant ROIs: " + final_string : img})
    return img, trueROIs

def edge_detection(frame, th=400, TH=400, debug = False, frame_number = 0, equalizehist=False):
    selected_frame = 0
    d_frames = {}
    gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    # To apply more blur without modifing frame
    blur = gray.copy()

    # Gaussian filter
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #blur = cv2.medianBlur(blur, 9)
    blur = cv2.bilateralFilter(blur, 7, 500, 500)
    """
        Median -- UTILE PER LE STATUE?
        
        Sembra che funzioni meglio sul riconoscimento degli edge delle statue.
    """
    # blur = cv2.medianBlur(blur, 11)
    # Bilateral
    # blur = cv2.bilateralFilter(blur, 5, 5, 20)


    if debug and selected_frame == frame_number:
        d_frames['Blurred image'] = blur
        d_frames['Gray'] = gray

    dst = blur
    # Canny
    # Threshold ottimali per aperture 5
    # th = 550
    # TH = 1750
    # Threshold con un po' più di rumore
    #th = 500
    #TH = 1300

    #Threshold rumorose
    #th = 400
    #TH = 800

    #Threshold molto rumorose
    #th = 400
    #TH = 400

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

def find_keypoint(img):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(img,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        return kp, des


def matcher(des_crop, des_or):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        try:
            matches = bf.match(des_crop, des_or)
        except:
            return 100
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x:x.distance)

        if len(matches) < globals.match_th:
            return 100

        sum = 0
        for el in matches:
            if el.distance < 45:
                sum += el.distance - (el.distance*0.3)
            elif el.distance > 65:
                sum += el.distance + (el.distance*0.3)
            else:
                sum += el.distance

        av_1 = sum / len(matches)
        return av_1
