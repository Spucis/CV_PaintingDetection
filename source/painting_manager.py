from source import globals, edge_detection
conf = globals.conf


class PaintingManager:
    def __init__(self, video_manager):
        self.video_manager = video_manager
        self.input_path = conf['input_path'] + '000' + conf['slash']
        self.cap = None
        self.out = None

    def open_video(self, video_name):
        self.cap, self.out = self.video_manager.open_video(video_name, self.input_path, conf['output_path'])

    def close_video(self):
        self.cap.release()
        self.out.release()

    def paint_detection(self):
        self.ROI_detection()

    def ROI_detection(self):
        # Read until video is completed
        count = 0
        while (self.cap.isOpened() and self.out.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                if count == 0:
                    print("Edge detection function.")
                mod_frame = edge_detection(frame, debug=True, corners=False, frame_number = count)
                # hough_transform()
                count += 1
                if count % 100 == 0:
                    print("Frame count: {}/{}".format(count, self.video_manager.n_frame))
                self.out.write(mod_frame)

                all_video = True
                if not all_video:
                    break
            else:
                break
        print("Fine edge_detection.")
