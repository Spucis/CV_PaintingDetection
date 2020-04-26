from source import globals
from source.detection_utils import *

conf = globals.conf

class PaintingManager:
    def __init__(self, video_manager):
        self.video_manager = video_manager
        self.input_path = conf['input_path'] + '000' + conf['slash']
        self.count = 0
        self.cap = None
        self.out = None

    def open_video(self, video_name):
        self.cap, self.out = self.video_manager.open_video(video_name, self.input_path, conf['output_path'])

    def close_video(self):
        self.cap.release()
        self.out.release()

    def ROI_detection(self, or_frame):
        gray_frame, marked_frame, ed_frame = edge_detection(or_frame, debug=True, corners=False, frame_number=self.count)
        ccl_frame = ccl_detection(or_frame, gray_frame, ed_frame)

        #ed_frame = cv2.cvtColor(ed_frame, cv2.COLOR_GRAY2BGR)

        return ccl_frame

    def paint_detection(self):
        # Read until video is completed
        while (self.cap.isOpened() and self.out.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                if self.count == 0:
                    print("Edge detection function.")
                mod_frame = self.ROI_detection(frame)
                # hough_transform()
                self.count += 1
                if self.count % 100 == 0:
                    print("Frame count: {}/{}".format(self.count, self.video_manager.n_frame))
                self.out.write(mod_frame)

                all_video = True
                if not all_video:
                    break
            else:
                break
        print("Fine edge_detection.")
