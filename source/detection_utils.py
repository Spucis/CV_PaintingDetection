from source import globals

import numpy as np
import cv2
from json import *
import random

conf = globals.conf


def show_frame(d_frames):
    for label, frame in d_frames.items():
        cv2.imshow(label, frame)
    cv2.waitKey()
    cv2.waitKey()
    cv2.waitKey()
    cv2.destroyAllWindows()

# Connected Components Labeling
# Tranforms a binary input image into a simbolic one, in which all the pixel of the same components
# have the same label
def ccl_detection(or_frame, gray_frame, frame):

    """
        Marco
        Provo cv.connectedComponents(	image[, labels[, connectivity[, ltype]]]	)
    """

    """
    CONNECTED COMPONENTS AND LABELING
    
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

    contours, hierarchy = cv2.findContours(frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(hierarchy.shape)
    # Tolgo la prima dimensione -- Da controllare, in generale
    hierarchy = hierarchy[0]
    #for i in range(hierarchy.shape[0]):
    #    mod_f = cv2.drawContours(or_frame.copy(), contours, hierarchy[i][0], (0, 255, 0))
        # per adesso tolgo il frame normale 'Canny_Frame':m_frame,
    #    show_frame({'CCL_Frame_ContourHierarchy: {}'.format(i): mod_f})
    i = 0
    #hull = []
    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        x,y,w,h = cv2.boundingRect(currentContour)

        #hull.append(cv2.convexHull(currentContour))
        text = "C{}".format(i)
        fontFace  =  cv2.FONT_HERSHEY_SIMPLEX
        fontScale = .5
        thickness = 1
        textSize, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
        img = cv2.rectangle(or_frame, (x, y), (x + textSize[0], y - textSize[1] - 5), (0,0,255), cv2.FILLED)
        img = cv2.putText(img, text , (x, y - 5), fontFace, fontScale, (0,0,0), thickness)

        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
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
    return img

def edge_detection(frame, debug = False, corners = False, frame_number = 0):

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

    mod_f = cv2.Canny(dst.astype(np.uint8), th, TH, apertureSize=5, L2gradient=True)

    # Frame with the edges highlighted in green
    vis = frame.copy()
    vis = np.uint8(vis / 2.)
    vis[mod_f != 0] = (0, 255, 0)

    if debug and selected_frame == frame_number:
        d_frames['Vis'] = vis
        d_frames['Canny_OutPut'] = mod_f

    # CORNER DETECTION
    if corners:
        mod_f = np.float32(mod_f)
        dst = cv2.cornerMinEigenVal(mod_f, 10) # , 5, 0.04)

        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)

    # conversione colore
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
            d_frames['Corner detection image'] = mod_f2

    show_frame(d_frames)
    # return mod_f2.astype(np.uint8)
    return gray, vis, mod_f


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

