from source import edge_detection, hough_transform, globals
from source.painting_manager import *

conf = globals.conf

if __name__ == '__main__':
    video_name = "VIRB0401"     #000
    video_name = "VIRB0391"

    #video_name = "VIRB0400"
    #video_name = "VIRB0402"

    video_name = "20180206_114720" # 002

    #video_name = "GOPR1926" #003

    #video_name = "VIRB0420" #008
    #video_name = "VIRB0421" #008

    #video_name = "VID_20180529_112706" # 010
    #video_name = "VID_20180529_112951" # 010
    #video_name = "VID_20180529_112849" # 010
    #video_name = "VID_20180529_112828" # 010

    p_manager = PaintingManager(globals.VideoManager())

    #p_manager.keypoint_writedb()
    #p_manager.keypoint_readdb()

    p_manager.open_video(video_name)

    p_manager.paint_detection()

    p_manager.close_video()
