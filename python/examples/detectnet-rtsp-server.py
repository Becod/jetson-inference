#!/usr/bin/python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import cv2
import numpy as np

class Pipe:
    def __init__(self, camera, udpout, width, height):
        self.cameraPath = camera
        self.camera = jetson.utils.gstCamera(width, height, camera)
        self.width = width
        self.height = height
        self.udpout = udpout
        self.gst_str_rtp = "appsrc is-live=true ! videoconvert ! x264enc ! h264parse config-interval=1 ! rtph264pay pt=96 ! multiudpsink clients=" + udpout
        self.out = None
    def write(self, frame, fps):
        if (self.out == None):
            self.out = cv2.VideoWriter(self.gst_str_rtp, 0, fps, (self.width, self.height), True)
        self.out.write(frame)

    def __repr__(self):
        return "Pipe({:s})".format(self.__str__())
    def __str__(self):
        return self.cameraPath + " >> " + self.udpout


# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("--network", type=str, default="pednet", help="pre-trained model to load, see below for options")
parser.add_argument("--threshold", type=float, default=0.45, help="minimum detection threshold to use")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (NULL for CSI camera 0)\nor for VL42 cameras the /dev/video node to use.\nby default, MIPI CSI camera 0 will be used.")

parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

parser.add_argument("--udpout", type=str, default="127.0.0.1:5000", help="")

opt, argv = parser.parse_known_args()

# load the object detection network
net = jetson.inference.detectNet(opt.network, argv, opt.threshold)

# create the camera and display
cameras = opt.camera.split(";")
udpouts = opt.udpout.split(";")
pipes = []
for i in range(len(cameras)):
    pipes.append(Pipe(cameras[i], udpouts[i], opt.width, opt.height))
print(pipes)

channelId = 0
font = None	
	
# process frames until user exits
while True:
    # capture the image
    img, width, height = pipes[channelId].camera.CaptureRGBA(zeroCopy=True)

    # detect objects in the image (with overlay)
    detections = net.Detect(img, width, height)
    fps = None
    try:
        fps = 1000.0 / net.GetNetworkTime()
        fps = fps / len(pipes)
    except ZeroDivisionError:
        continue

    text = "CH:{:d} | {:s} | {:.0f} FPS | {:d} detected".format(channelId, opt.network, fps, len(detections))
    #print(text)
    if (font == None):
        font = jetson.utils.cudaFont(size=jetson.utils.adaptFontSize(width))
    font.OverlayText(img, width, height, text, 10, 10, font.White, font.Gray40)
	
    # synchronize with the GPU
    jetson.utils.cudaDeviceSynchronize()

    # convert cudaim to im
    frame = jetson.utils.cudaToNumpy(img, width, height, 4)
    frame = np.uint8(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


    for detection in detections:
        class_idx = detection.ClassID
        class_desc = net.GetClassDesc(class_idx)
        label = "{:s}, {:.2f}%".format(class_desc, 100 * detection.Confidence)
        cv2.rectangle(frame, (int(detection.Left), int(detection.Top)), (int(detection.Right), int(detection.Bottom)), (255, 0, 0), 1)
        cv2.putText(frame, label, (int(detection.Left), int(detection.Top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)



    pipes[channelId].write(frame, fps)

    channelId = (channelId + 1) % len(pipes)


    # print out performance info
    #net.PrintProfilerTimes()

