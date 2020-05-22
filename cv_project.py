from source import edge_detection, hough_transform, globals
from source.painting_manager import *

conf = globals.conf

if __name__ == '__main__':
    #video_name = "VIRB0401" #000
    video_name = "VIRB0391.mp4"

    video_name = "VIRB0400.mp4"
    #video_name = "VIRB0402"

    video_name = "20180206_114720.mp4" # 002 # IMP 1
    #video_name = "20180206_113716.mp4"
    #video_name = "20180206_112930.mp4"

    #video_name = "GOPR1926" #003
    #video_name = "VIRB0420.mp4" #008
    #video_name = "VIRB0421" #008
    #video_name = "VIRB0413.mp4" #008


    #video_name = "VID_20180529_112706.mp4" # 010 # IMP 2 Primo da fare vedere
    #video_name = "VID_20180529_112951.mp4" # 010

    #video_name = "VID_20180529_112849.mp4" # 010
    #video_name = "VID_20180529_112828" # 010

    # invece che PaintingManager, FrameManager? Così inseriamo anche le persone e concettualmente rimane giusto
    p_manager = PaintingManager(globals.VideoManager())
    p_manager.db_keypoints()
    p_manager.open_video(video_name)
    p_manager.paint_detection()
    p_manager.close_video()
