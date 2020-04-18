
import numpy as np
import cv2
import matplotlib.pyplot as plt

class edge_detector():
    def __init__(self):
        pass
    #@staticmethod
    def edge_detection(self):
        print("Edge detection function.")


def edge_detection(debug = False, force_all_video = True, corners = False):
    print("Second Edge detection function.")

    basepath = "C:\\Users\\Marco\\Desktop\\Universita\\Magistrale\\Secondo anno\\Computer Vision and Cognitive Systems\\Project material\\videos\\000\\"
    video_name = "VIRB0401"
    cap = cv2.VideoCapture()
    cap.open("{}{}.MP4".format(basepath,video_name))
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    in_codec = cap.get(cv2.CAP_PROP_FOURCC)
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    in_frameSize = np.around((cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))).astype(np.uint32)
    n_frame = np.around(cap.get(cv2.CAP_PROP_FRAME_COUNT)).astype(np.uint32)

    print("Frame count: {}".format(n_frame))
    print("Codec number: {}\nFPS: {}\nFrame size: {}".format(int(in_codec), np.around(in_fps).astype(np.uint32),
                                                             np.around(in_frameSize).astype(np.uint32)))

    out = cv2.VideoWriter(".\\output\\videos\\{}.mp4".format(video_name), cv2.VideoWriter_fourcc(*'mp4v'), np.around(in_fps).astype(np.uint32),
                          tuple(np.around(in_frameSize).astype(np.uint32)), True)
    out.open(".\\output\\videos\\{}.mp4".format(video_name), cv2.VideoWriter_fourcc(*'mp4v'), np.around(in_fps).astype(np.uint32),
             tuple(np.around(in_frameSize).astype(np.uint32)))

    if (out.isOpened() == False):
        print("Error opening out video stream or file")

    # print("io")
    # print(cap.isOpened())
    # print(out.isOpened())

    #frames = np.zeros((n_frame, in_frameSize[1], in_frameSize[0], 3))
    i = 0
    # Read until video is completed
    while (cap.isOpened() and out.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            #Gaussian filter
            dst = cv2.GaussianBlur(frame, (5, 5), 0)

            #Median
            #dst = cv2.medianBlur(frame, 13)
            #Bilateral
            #dst = cv2.bilateralFilter(frame, 5, 5, 20)

            #Try threshold to eliminate white
            gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            ret, th2 = cv2.threshold(gray, 0, 200,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresholded_image = frame
            thresholded_image[gray > th2] = [0,0,0]
            #thresholded_image[gray > 60] = [0,0,0]

            if debug and i == 0:
                cv2.imshow('Blurred image', dst)
                cv2.imshow('Thresholded image', thresholded_image)
                cv2.waitKey()
                cv2.destroyAllWindows()
            #Canny

            factor = 7
            th = 100
            TH = th * factor

            # """
            v = np.median(dst.astype(np.uint8))
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

            if debug and i == 0:
                cv2.imshow('Edge detection image', mod_f)
                cv2.waitKey()
                cv2.destroyAllWindows()

            #CORNER DETECTION
            if corners:
                mod_f = np.float32(mod_f)
                dst = cv2.cornerHarris(mod_f, 10, 5, 0.04)

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

                if debug and i == 0:
                    cv2.imshow('Corner detection image', mod_f2)
                    cv2.waitKey()
                    cv2.destroyAllWindows()


                if debug and not force_all_video:
                    break
            out.write(mod_f2.astype(np.uint8))

            #frames[i] = frame
            i += 1
            # out.write(frame_edge)
        else:
            break

    cap.release()
    out.release()

    print("Fine edge_detection.")
