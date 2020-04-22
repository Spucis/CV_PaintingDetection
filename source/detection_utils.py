from source import globals

import numpy as np
import cv2
import random

conf = globals.conf

def show_frame(d_frames):
    for label, frame in d_frames.items():
        cv2.imshow(label, frame)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Connected Components Labeling
# Tranforms a binary input image into a simbolic one, in which all the pixel of the same components
# have the same label
def ccl_detection(or_frame, m_frame, frame):
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(hierarchy.shape[0])
    for i in range(hierarchy.shape[0]):
        mod_f = cv2.drawContours(or_frame, contours, -1, (0, 255, 0))
        show_frame({'CCL_Frame': mod_f, 'Canny_Frame':m_frame})
    return mod_f

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
    return vis, mod_f.astype(np.uint8)
