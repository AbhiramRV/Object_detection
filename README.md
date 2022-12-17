# Object_detection
2D object detection

VGG Net, Res Net, Inception
Fast RCNN
SSD,YOLOv4


3D object detection can be done using 2D object detection+ Depth estimation using sterio vision

A very good resource to refer concepts of deep learning http://cs231n.stanford.edu/schedule.html


Multiple Object tracking is done using YOLO and DeepSORT(Simple Online Realtime Tracking) 
(Detection, prediction, association)
1. Perform object detection using YOLOV4
2. Deep SORT (Deep sort is extension of sort algorithm)
    --Kalman filtering: Process the correlation frame by frame
    --Hungarian Algorithm: correlation measurement
    --CNN Networks: Training and feature extraction
    
Good resource to implement Tracking is https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet

YOLOv7 which was launched recently surpassed all the earlier known object detector in speed and accuracy.
Good medium article to implement perception problems is:
https://medium.com/@shahrullo/visual-perception-for-self-driving-cars-bb500f8c6adc
