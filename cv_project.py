from source import edge_detection, hough_transform, globals
from source.painting_manager import *

conf = globals.conf

if __name__ == '__main__':
    video_name = "VIRB0401"
    #video_name = "VIRB0391"
    #video_name = "VIRB0400"
    #video_name = "VIRB0402"

    p_manager = PaintingManager(globals.VideoManager())
    p_manager.open_video(video_name)

    p_manager.paint_detection()

    p_manager.close_video()
