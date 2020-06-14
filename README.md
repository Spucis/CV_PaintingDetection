# CV_PaintingDetection
Computer Vision - Museum Painting Detection

Authors: *Marco Cagrandi*, *Alessio Ruggi* and *Daniele Lombardo*.

GitHub Repository: https://github.com/Spucis/CV_PaintingDetection

# Project structure
The file `cv_project.py` is the entry point of the application. It contains the video selection and it is where you can change the *step* of the program.
The file `conf.json` is the configuration file and it's were you can change the input directory and (optionally) all the paths of the directories of the project.
- **input** directory contains all the provided inputs such as the images, the videos, the .csv file and the map
- **output** directory contains all the produced outputs (e.g. the videos)
- **source** contains the source files:
  - `detection_utils.py` contains all the utilities to perform the detection phase (e.g. edge detection and connected components detection)
  - `painting_manager.py` contains all the functions needed to fulfill the requested work
  - `globals.py` contains all the global definitions and the utilities used to manage the input video itself (such as opening and closing the video).

An output file `output_details.json` is provided: it summarize all the information provided by the program.

# How it works and how to enable all the options available
Each `step` frames the program will stop and process the frame, producing a modified frame `mod_frame` and writing it on the disk.
In `cv_project.py` is possible to enable the output of the details with the option `json_output_details`, enable the segmentation of both statues and 
detected paintings with the option `en_segmentation`, and modify the `step`.

Instead, inside the `painting_manager.py` file and `paint_detection(...)` function you can manipulate other options:
- `verbose` option which will enable more output over the standard output
- `show_details` will stop the execution to let you see the detected ROI, the rectified attempt and the proposed selected painting in the DB
- `otsu_opt_enabled` will enable the *otsu_optimization*, as discussed in the report 

# Link to Google Drive to download YOLO weights
https://drive.google.com/open?id=1Na6uMDc_ST1179GNjGjJ5ywywlBoYXNj

