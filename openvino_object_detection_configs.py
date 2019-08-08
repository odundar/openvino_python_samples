# MIT License
#
# Copyright (c) 2019 Onur Dundar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import cv2 as cv
import argparse
import psutil
import os

# Import OpenVINO Inference Engine
from openvino.inference_engine import IECore, IENetwork


mobilenet_ssd_labels = {0:'background',
                        1: 'aeroplane',
                        2: 'bicycle',
                        3: 'bird',
                        4: 'boat',
                        5: 'bottle',
                        6: 'bus',
                        7: 'car',
                        8: 'cat',
                        9: 'chair',
                        10: 'cow',
                        11: 'diningtable',
                        12: 'dog',
                        13: 'horse',
                        14: 'motorbike',
                        15: 'person',
                        16: 'pottedplant',
                        17: 'sheep',
                        18: 'sofa',
                        19: 'train',
                        20: 'tvmonitor' }


def run_app():
    """
    Run Object Detection Application
    :return:
    """

    frame_count = 0

    # Load Network
    OpenVinoNetwork = IENetwork(model=arguments.model_xml, weights=arguments.model_bin)

    # Get Input Layer Information
    InputLayer = next(iter(OpenVinoNetwork.inputs))
    print("Input Layer: ", InputLayer)

    # Get Output Layer Information
    OutputLayer = next(iter(OpenVinoNetwork.outputs))
    print("Output Layer: ", OutputLayer)

    # Get Input Shape of Model
    InputShape = OpenVinoNetwork.inputs[InputLayer].shape
    print("Input Shape: ", InputShape)

    # Get Output Shape of Model
    OutputShape = OpenVinoNetwork.outputs[OutputLayer].shape
    print("Output Shape: ", OutputShape)

    # Load IECore Object
    OpenVinoIE = IECore()
    print("Available Devices: ", OpenVinoIE.available_devices)

    # Load CPU Extensions if Necessary
    if 'CPU' in arguments.target_device:
        OpenVinoIE.add_extension('/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.so', "CPU")

    # Configs
    if "CPU" in arguments.target_device:
        if int(arguments.cpu_num_threads) > 0:
            print("Setting CPU Threads to {}".format(arguments.cpu_num_threads))
            OpenVinoIE.set_config({"CPU_THREADS_NUM" : arguments.cpu_num_threads}, "CPU")

        if "GPU" not in arguments.target_device:
            if arguments.cpu_bind_thread:
                print("Setting CPU Threads Binding")
                OpenVinoIE.set_config({"CPU_BIND_THREAD": "YES"}, "CPU")
            else:
                OpenVinoIE.set_config({"CPU_BIND_THREAD": "NO"}, "CPU")

        if arguments.async:
            print("Setting CPU Stream {}".format(arguments.cpu_throughput_streams))
            try:
                stream = int(arguments.cpu_throughput_streams)
                if stream > 0:
                    OpenVinoIE.set_config({"CPU_THROUGHPUT_STREAMS": arguments.cpu_throughput_streams}, "CPU")
                else:
                    OpenVinoIE.set_config({"CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_AUTO"}, "CPU")
            except ValueError:
                if arguments.cpu_throughput_streams == "CPU_THROUGHPUT_NUMA":
                    OpenVinoIE.set_config({"CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_NUMA"}, "CPU")
                else:
                    OpenVinoIE.set_config({"CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_AUTO"}, "CPU")

    if "GPU" in arguments.target_device:
        if arguments.async:
            print("Setting GPU Stream {}".format(arguments.gpu_throughput_streams))
            try:
                stream = int(arguments.cpu_throughput_streams)
                if stream > 0:
                    OpenVinoIE.set_config({"GPU_THROUGHPUT_STREAMS": arguments.gpu_throughput_streams}, "GPU")
                else:
                    OpenVinoIE.set_config({"GPU_THROUGHPUT_STREAMS": "GPU_THROUGHPUT_AUTO"}, "GPU")
            except ValueError:
                OpenVinoIE.set_config({"GPU_THROUGHPUT_STREAMS": "GPU_THROUGHPUT_AUTO"}, "GPU")

        if "MULTI" in arguments.target_device and arguments.gpu_throttle:
            print("Enabling GPU Throttle")
            OpenVinoIE.set_config({"CLDNN_PLUGIN_THROTTLE": "1"}, "GPU")

    config = {}

    if arguments.pc:
        print("Setting Performance Counters")
        config["PERF_COUNT"] = "YES"

    # Create Executable Network
    if arguments.async:
        print("Async Mode Enabled")
        OpenVinoExecutable = OpenVinoIE.load_network(network=OpenVinoNetwork, config=config, device_name=arguments.target_device, num_requests=number_of_async_req)
    else:
        OpenVinoExecutable = OpenVinoIE.load_network(network=OpenVinoNetwork, config=config, device_name=arguments.target_device)

    # Generate a Named Window
    cv.namedWindow('Window', cv.WINDOW_NORMAL)
    cv.resizeWindow('Window', 800, 600)

    start_time = time.time()

    if arguments.input_type == 'image':
        frame_count += 1
        # Read Image
        image = cv.imread(arguments.input)

        # Get Shape Values
        N, C, H, W = OpenVinoNetwork.inputs[InputLayer].shape

        # Pre-process Image
        resized = cv.resize(image, (W, H))
        resized = resized.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        input_image = resized.reshape((N, C, H, W))

        # Start Inference
        start = time.time()
        results = OpenVinoExecutable.infer(inputs={InputLayer: input_image})
        end = time.time()
        inf_time = end - start
        print('Inference Time: {} Seconds'.format(inf_time))

        fps = 1./(end-start)
        print('Estimated FPS: {} FPS'.format(fps))

        fh = image.shape[0]
        fw = image.shape[1]

        # Write Information on Image
        text = 'FPS: {}, INF: {}'.format(round(fps, 2), round(inf_time, 2))
        cv.putText(image, text, (0, 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 125, 255), 1)

        # Print Bounding Boxes on Image
        detections = results[OutputLayer][0][0]
        for detection in detections:
            if detection[2] > arguments.detection_threshold:
                print('Original Frame Shape: ', fw, fh)
                xmin = int(detection[3] * fw)
                ymin = int(detection[4] * fh)
                xmax = int(detection[5] * fw)
                ymax = int(detection[6] * fh)
                cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 125, 255), 3)
                text = '{}, %: {}'.format(mobilenet_ssd_labels[int(detection[1])], round(detection[2], 2))
                cv.putText(image, text, (xmin, ymin - 7), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)

        cv.imshow('Window', image)
        cv.waitKey(0)

    else:
        print("Running Inference for {} - {}".format(arguments.input_type, arguments.input))

        process_id = os.getpid()
        process = psutil.Process(process_id)

        total_inference_time = 0.0
        # Implementation for CAM or Video File
        # Read Image
        capture = cv.VideoCapture(arguments.input)
        has_frame, frame = capture.read()
        frame_count += 1

        if not has_frame:
            print("Can't Open Input Video Source {}".format(arguments.input))
            exit(-1)

        # Get Shape Values
        N, C, H, W = OpenVinoNetwork.inputs[InputLayer].shape
        fh = frame.shape[0]
        fw = frame.shape[1]
        print('Original Frame Shape: ', fw, fh)

        request_order = list()
        process_order = list()
        frame_order = list()
        if arguments.async:
            print("Async Mode Set")
            for i in range(number_of_async_req):
                request_order.append(i)
                print('Request Id {} Created'.format(i))

            print('Request Ids {}'.format(request_order))

        while has_frame:
            if arguments.async:
                if len(request_order) > 0:
                    resized = cv.resize(frame, (W, H))
                    resized = resized.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                    input_data = resized.reshape((N, C, H, W))
                    req_id = request_order[0]
                    request_order.pop(0)
                    OpenVinoExecutable.start_async(req_id, inputs={InputLayer: input_data})
                    process_order.append(req_id)
                    frame_order.append(frame)

                if len(process_order) > 0:
                    first = process_order[0]
                    if OpenVinoExecutable.requests[first].wait(0) == 0:
                        results = OpenVinoExecutable.requests[first].outputs[OutputLayer]
                        process_order.pop(0)
                        request_order.append(first)
                        show_frame = frame_order[0]
                        frame_order.pop(0)

                        detections = results[0][0]
                        for detection in detections:
                            if detection[2] > arguments.detection_threshold:
                                xmin = int(detection[3] * fw)
                                ymin = int(detection[4] * fh)
                                xmax = int(detection[5] * fw)
                                ymax = int(detection[6] * fh)
                                cv.rectangle(show_frame, (xmin, ymin), (xmax, ymax), (0, 125, 255), 3)
                                text = '{}, %: {}'.format(mobilenet_ssd_labels[int(detection[1])],
                                                          round(detection[2], 3))
                                cv.putText(show_frame, text, (xmin, ymin - 7), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)

                        fps = frame_count / (time.time() - start_time)
                        # Write Information on Image
                        text = 'FPS: {}, INF: {} ms'.format(round(fps, 3), "-")
                        cv.putText(show_frame, text, (0, 20), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 125, 255), 1)

                        text = "SYS CPU% {} SYS MEM% {} \n " \
                               "PROC CPU Affinity {} \n " \
                               "NUM Threads {} \n " \
                               "PROC CPU% {} \n " \
                               "PROC MEM% {}".format(psutil.cpu_percent(),
                                                     psutil.virtual_memory()[2],
                                                     process.cpu_affinity(),
                                                     process.num_threads(),
                                                     process.cpu_percent(),
                                                     round(process.memory_percent(), 4))

                        cv.putText(show_frame, text, (0, 50), cv.FONT_HERSHEY_COMPLEX, 0.8, (250, 0, 255), 1)

                        if arguments.pc:
                            perf_counts = OpenVinoExecutable.requests[0].get_perf_counts()
                            print("Pefrormance counts for infer request")
                            for layer, stats in perf_counts.items():
                                max_layer_name = 30
                                print("{:<30}{:<15}{:<30}{:<20}{:<20}{:<20}".format(
                                    layer[:max_layer_name - 4] + '...' if (len(layer) >= max_layer_name) else layer,
                                    stats['status'],
                                    'layerType: ' + str(stats['layer_type']),
                                    'realTime: ' + str(stats['real_time']),
                                    'cpu: ' + str(stats['cpu_time']),
                                    'execType: ' + str(stats['exec_type'])))


                        cv.imshow('Window', show_frame)
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            break

                if len(process_order) > 0:
                    has_frame, frame = capture.read()
                    frame_count += 1
            else:
                frame_count += 1
                resized = cv.resize(frame, (W, H))
                resized = resized.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                input_data = resized.reshape((N, C, H, W))
                # Start Inference
                results = OpenVinoExecutable.infer(inputs={InputLayer: input_data})

                fps = frame_count / (time.time() - start_time)
                inf_time = (time.time() - start_time) / frame_count
                # Write Information on Image
                text = 'FPS: {}, INF: {} ms'.format(round(fps, 3), round(inf_time, 3))
                cv.putText(frame, text, (0, 20), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 125, 255), 1)

                # Print Bounding Boxes on Image
                detections = results[OutputLayer][0][0]
                for detection in detections:
                    if detection[2] > arguments.detection_threshold:
                        xmin = int(detection[3] * fw)
                        ymin = int(detection[4] * fh)
                        xmax = int(detection[5] * fw)
                        ymax = int(detection[6] * fh)
                        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 125, 255), 3)
                        detection_percentage = round(detection[2], 4)
                        text = '{}, %: {}'.format(mobilenet_ssd_labels[int(detection[1])], detection_percentage)
                        cv.putText(frame, text, (xmin, ymin - 7), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)

                text = "SYS CPU% {} SYS MEM% {} \n " \
                       "PROC CPU Affinity {} \n " \
                       "NUM Threads {} \n " \
                       "PROC CPU% {} \n " \
                       "PROC MEM% {}".format(psutil.cpu_percent(),
                                             psutil.virtual_memory()[2],
                                             process.cpu_affinity(),
                                             process.num_threads(),
                                             process.cpu_percent(),
                                             round(process.memory_percent(), 4))

                cv.putText(frame, text, (0, 50), cv.FONT_HERSHEY_COMPLEX, 0.8, (250, 0, 250), 1)
                cv.imshow('Window', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                has_frame, frame = capture.read()

                if arguments.pc:
                    perf_counts = OpenVinoExecutable.requests[0].get_perf_counts()
                    print("Pefrormance counts for infer request")
                    for layer, stats in perf_counts.items():
                        max_layer_name = 30
                        print("{:<30}{:<15}{:<30}{:<20}{:<20}{:<20}".format(
                            layer[:max_layer_name - 4] + '...' if (len(layer) >= max_layer_name) else layer,
                            stats['status'],
                            'layerType: ' + str(stats['layer_type']),
                            'realTime: ' + str(stats['real_time']),
                            'cpu: ' + str(stats['cpu_time']),
                            'execType: ' + str(stats['exec_type'])))

    end_time = time.time()
    print('Elapsed Time: {} Seconds'.format(end_time - start_time))
    print('Number of Frames: {} '.format(frame_count))
    print('Estimated FPS: {}'.format(frame_count / (end_time - start_time)))


global arguments
global number_of_async_req

"""
Entry Point of Application
"""
if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Basic OpenVINO Example to Test Object Detection Model')
    parser.add_argument('--model-xml',
                        default='/home/intel/openvino_models/object_detection/common/mobilenet-ssd/FP32/mobilenet-ssd.xml',
                        help='XML File')
    parser.add_argument('--model-bin',
                        default='/home/intel/openvino_models/object_detection/common/mobilenet-ssd/FP32/mobilenet-ssd.bin',
                        help='BIN File')
    parser.add_argument('--target-device', default='CPU',
                        help='Target Plugin: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA,CPU')
    parser.add_argument('--input-type', default='image', help='Type of Input: image, video, cam')
    parser.add_argument('--input', default='/home/intel/Pictures/faces.jpg',
                        help='Path to Input: WebCam: 0, Video File or Image file')
    parser.add_argument('--detection-threshold', default=0.6, help='Object Detection Accuracy Threshold')

    parser.add_argument('--async', action="store_true", default=False, help='Run Async Mode')
    parser.add_argument('--request-number', default=1, help='Number of Requests')

    parser.add_argument('--pc', action="store_true", default=False, help='Enable Performance Counters')

    parser.add_argument('--cpu-num-threads', default=0, help='Limit CPU Threads')
    parser.add_argument('--cpu-bind-thread', action="store_true", default=False, help='Bind Threads to CPU')
    parser.add_argument('--cpu-throughput-streams', default="CPU_THROUGHPUT_AUTO",
                        help="Int Values or CPU_THROUGHPUT_NUMA if not set CPU_THROUGHPUT_AUTO")
    parser.add_argument('--gpu-throughput-streams', default="GPU_THROUGHPUT_AUTO",
                        help="Int Values if not set GPU_THROUGHPUT_AUTO")
    parser.add_argument('--gpu-throttle', action="store_true", default=False,
                        help="multi-device execution with the CPU+GPU performs best with GPU trottling hint")

    global arguments
    arguments = parser.parse_args()

    global number_of_async_req
    number_of_async_req = int(arguments.request_number)

    print('WARNING: No Argument Control Done, You Can GET Runtime Errors')
    run_app()
