"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

import ffmpeg
import sys

from imutils.video import FPS

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
# MQTT_HOST = IPADDRESS
MQTT_HOST = "13.229.206.192"
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
MQTT_TOPIC1 = "person"
MQTT_TOPIC2 = "person/duration"

font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.5
color = (255,0,0)
thk = 1

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=False, type=str, default="WEBCAM",
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="MYRIAD",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def preprocessing(frame, frame_size=300):
    # model format
    n, c, h, w = [1, 3, frame_size, frame_size]
    # resize frame to fit model spec
    dim = (frame_size, frame_size)
    image = np.copy(frame)
    image = cv2.resize(image, dim)
    # Rearrange image to CHW format
    image = image.transpose((2,0,1))
    # Reshape image to fit model
    image = image.reshape(n,c,h,w)

    return image


def infer_on_stream(args, client):
# def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    people_count = 0
    people_onframe = 0
    list_time_onframe = []
    avg_dur_on_frame = 0

    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device)

    ### TODO: Handle the input stream ###
    net_input_stream = infer_network.get_input_shape()

    if "WEBCAM" in args.input:
        cap = cv2.VideoCapture(0)
        cap.open(0)
    else:
        cap = cv2.VideoCapture(args.input)
        cap.open(args.input)

    width = int(cap.get(3))
    height = int(cap.get(4))

    last_count = 0
    frame_refresh = 0

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    # out = cv2.VideoWriter('out.mp4', fourcc, 30, (width,height))

    ### TODO: Loop until stream is over ###
    while cap.isOpened:
        fps = FPS().start()
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(30)

        ### TODO: Pre-process the image as needed ###
        prep_frame = preprocessing(frame)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(prep_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            res = result[0][0]
            preds = [pred for pred in res if pred[1] == 1 and pred[2] > prob_threshold]
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            current_count = len(preds)

            # Detection Box
            for pred in preds:
                x = int(pred[3] * width)
                y = int(pred[4] * height)

                x2 = int(pred[5] * width)
                y2 = int(pred[6] * height)

                cv2.rectangle(frame, (x,y), (x2,y2), (255,0,0), 2)

            # people enter the frame
            if current_count > last_count and frame_refresh == 0: 
                start_time = time.perf_counter()
                people_count = people_count + current_count - last_count
                people_onframe = current_count
                
                # Send to MQTT Server
                client.publish(MQTT_TOPIC1, json.dumps({"count" : people_onframe, "total" : people_count}))
                
                last_count = current_count

            # people leave the frame    
            if current_count < last_count and frame_refresh == 0:
                duration = time.perf_counter() - start_time
                # print("Duration : {}".format(duration))
                
                client.publish(MQTT_TOPIC2, json.dumps({"duration" : duration}))
                
                people_onframe = current_count
                last_count = current_count
            
            frame_refresh = frame_refresh + 1
            
            if frame_refresh == 49:
                frame_refresh = 0


            
        
        # Draw some information
        fps.update()
        fps.stop()
        infer = "Approx FPS : {:.2f}".format(fps.fps())
        people_count_txt = "People Count : {}".format(people_count)
        people_on_frame = "People on Frame : {}".format(people_onframe)

        cv2.putText(frame, infer, (5, 20), font, fontscale, color, thk, cv2.LINE_AA, False)
        cv2.putText(frame, people_count_txt, (5, 40), font, fontscale, color, thk, cv2.LINE_AA, False)
        cv2.putText(frame, people_on_frame, (5, 60), font, fontscale, color, thk, cv2.LINE_AA, False)
        
        
        # cv2.imshow('capture', frame)
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
    
        
        # out.write(frame)
        if key_pressed == 27:
            break

    # out.release()
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
    # infer_on_stream(args)


if __name__ == '__main__':
    main()
