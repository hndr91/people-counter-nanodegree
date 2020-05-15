#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.img_info_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="MYRIAD", cpu_extension=None):
        ### TODO: Load the model ###
        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        # Load the IENetwork into the plugin
        # This is my experimental on HETERO plugin only using MYRIAD
        if "HETERO" in device:
            devices = self.plugin.available_devices
            devices = [d for d in devices if "MYRIAD" in d]
            devices = ','.join(devices)
            devices = 'HETERO:' + devices

            self.exec_network = self.plugin.load_network(self.network, devices)
        else:
            self.exec_network = self.plugin.load_network(self.network, device)


        # Get the output layer
        self.output_blob = next(iter(self.network.outputs))

        return

    def get_input_blob_name(self):
        # Get all possible input layer
        for blob_name in self.network.inputs:
            if len(self.network.inputs[blob_name].shape) == 4:
                self.input_blob = blob_name
            elif len(self.network.inputs[blob_name].shape) == 2:
                self.img_info_blob = blob_name
            else:
                raise RuntimeError("Unsuppored input layer dimension. Only 2D and 4D input layers are supported")
        
        return self.input_blob, self.img_info_blob

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        
        # Do some modification to handle 2 input for model Faster RCNN like
        if self.img_info_blob:
            return self.network.inputs[self.input_blob].shape, self.network.inputs[self.img_info_blob].shape
        else:
            return self.network.inputs[self.input_blob].shape


    # def exec_net(self, image):
    def exec_net(self, input_dict):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        
        # Update input to handle dictionary parameters.
        return self.exec_network.start_async(request_id=0, inputs=input_dict)

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].outputs[self.output_blob]
