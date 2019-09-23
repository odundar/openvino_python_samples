# MIT License
#
# Copyright (c) 2019 Onur Dundar onur.dundar1@gmail.com
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

# Required Libraries Imported
import sys
import os
import time
import cv2 as cv
import numpy as np
import psutil
from math import exp as exp

# Import OpenVINO
from openvino.inference_engine import IENetwork, IEPlugin

# OpenCV Backends and Targets (Inference Engine is opensource OpenVINO Distribution
# opencv_backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV)
# opencv_targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD)

# Performance Counters
global inference_time_duration
global resize_time_durations
global resize_time_duration
global post_process_durations
post_process_durations = 0.
global inferred_frame_count
global frame_read_times


class Config:
    """
    Config Model to Store Arguments
    """
    # Detection Threshold
    CONFIDENCE_THRESHOLD = 0.6

    # Source Type of Video
    VIDEOSOURCE = str()

    # Inference Framework if caffe or tensorflow opencv will be used
    INFERENCE_FRAMEWORK = str()

    # Source of Video, Full path of video file or index of Camera
    VIDEOPATH = str()  # full path or 0 for live

    # Platform to
    INFERENCE_PLATFORM = "CPU"  # cpu, gpu, movidius
    OPENCV_INFERENCE_BACKEND = "OPENCV_DNN"

    # Labels List which is created by reading labels file
    MODEL_LABELS = list()

    # Random Colors When Rectangle to be drawn for detected objects
    LABEL_COLORS = list()  # colors to draw at post process

    # Label Names File for the detection
    MODEL_LABELS_FILE = str()

    # Path to protobuf file or .xml file for openvino
    MODEL_FILE = str()
    # Path to weights/biases file or .bin file for openvino
    MODEL_WEIGHT_FILE = str()

    # Manually Set Number of Frames to Infer, if not set app tries to infer all the frames
    INFERENCE_FRAMERATE = -1

    # Model Input Expected Image Size
    MODEL_IMAGE_HEIGHT = 300
    MODEL_IMAGE_WIDTH = 300

    # Actual Frame Size from Video Source
    IMAGE_HEIGHT = 300
    IMAGE_WIDTH = 300

    # Used to Determine Output Window Size
    OUTPUT_IMAGE_HEIGHT = 600
    OUTPUT_IMAGE_WIDTH = 800

    # Model Mean/Scale
    MODEL_MEANS = [127.5, 127.5, 127.5]
    MODEL_SCALE = (1. / 127.5)

    # used when reading from a video file, according to system speed it can be fast
    FPS_DELAY = 100

    # OpenVINO Has Feature to Run Async Inferrence
    ASYNC = False

    # OpenVINO Has Feature to Run Inference on Multiple Batches
    BATCH_SIZE = 1

    # Number of simultaneous Requests to Handle
    OPENVINO_NUM_REQUESTS = 1

    # Enable Performance Counter Report
    OPENVINO_PERFORMANCE_COUNTER = False

    # If Given Object Detection Model is YOLOv3
    YOLO_MODEL_DEFINED = False
    IOU_THRESHOLD = 0.4

    # Libray Paths
    OPENVINO_CPU_LIBPATH = '/home/intel/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so'
    OPENVINO_LIBPATH = '/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64'

    @staticmethod
    def print_summary():
        """Prints values of Config Fields

        Args:

        Returns:
            None
        """
        print("Inference Framework             :{}".format(Config.INFERENCE_FRAMEWORK))
        print("Inference Device                :{}".format(Config.INFERENCE_PLATFORM))
        print("OpenCV Inference Backend        :{}".format(Config.OPENCV_INFERENCE_BACKEND))
        print("Video Source                    :{}".format(Config.VIDEOSOURCE))
        print("Video Path                      :{}".format(Config.VIDEOPATH))
        print("Model Network Path              :{}".format(Config.MODEL_FILE))
        print("Model Weights Path              :{}".format(Config.MODEL_WEIGHT_FILE))
        print("Model Labels Path               :{}".format(Config.MODEL_LABELS_FILE))
        print("Detection Confidence Threshold  :{}".format(Config.CONFIDENCE_THRESHOLD))
        print("Inference Frame Rate            :{}".format(Config.INFERENCE_FRAMERATE))
        print("Inference Async                 :{}".format(Config.ASYNC))
        print("FPS Delay                       :{}".format(Config.FPS_DELAY))
        print("Performance Counter Report      :{}".format(Config.OPENVINO_PERFORMANCE_COUNTER))
        print("Is It YOLOv3 Model              :{}".format(Config.YOLO_MODEL_DEFINED))
        print("Intersection Over Union Thres   :{}".format(Config.IOU_THRESHOLD))
        print("Batch Size                      :{}".format(Config.BATCH_SIZE))
        print("Number of Async Requests        :{}".format(Config.OPENVINO_NUM_REQUESTS))
        print("Model Image Width               :{}".format(Config.MODEL_IMAGE_WIDTH))
        print("Model Image Height              :{}".format(Config.MODEL_IMAGE_HEIGHT))
        print("Model Mean Substract            :{}".format(Config.MODEL_MEANS))
        print("Model Scale                     :{}".format(Config.MODEL_SCALE))
        print("Image Width                     :{}".format(Config.IMAGE_WIDTH))
        print("Image Height                    :{}".format(Config.IMAGE_HEIGHT))
        print("Image Output Width              :{}".format(Config.OUTPUT_IMAGE_WIDTH))
        print("Image Output Height             :{}".format(Config.OUTPUT_IMAGE_HEIGHT))
        print("OpenVINO CPU Lib Path           :{}".format(Config.OPENVINO_CPU_LIBPATH))
        print("OpenVINO Lib Path               :{}".format(Config.OPENVINO_LIBPATH))

        return None


class YoloV3Params:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    # Code from: https://docs.openvinotoolkit.org/latest/_inference_engine_ie_bridges_python_sample_object_detection_demo_yolov3_async_README.html
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else len(param['mask'].split(',')) if 'mask' in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]
        self.side = side
        if self.side == 13:
            self.anchor_offset = 2 * 6
        elif self.side == 26:
            self.anchor_offset = 2 * 3
        elif self.side == 52:
            self.anchor_offset = 2 * 0
        else:
            assert False, "Invalid output size. Only 13, 26 and 52 sizes are supported for output spatial dimensions"

    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [print("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


def parse_yolo_region(blob, resized_image_shape, originial_height, original_width, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    # _, _, out_blob_h, out_blob_w = blob.shape
    # assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
    #                                 "be equal to width. Current height = {}, current width = {}" \
    #                                 "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h = originial_height
    orig_im_w = original_width
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            x = (col + predictions[box_index + 0 * side_square]) / params.side * resized_image_w
            y = (row + predictions[box_index + 1 * side_square]) / params.side * resized_image_h
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            w = w_exp * params.anchors[params.anchor_offset + 2 * n]
            h = h_exp * params.anchors[params.anchor_offset + 2 * n + 1]
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h / resized_image_h, w_scale=orig_im_w / resized_image_w))

    return objects


def help_menu():
    """Help Menu for the application

    Args:

    Returns:
        None
    """

    print('This script helps you to run real time object detection sample using OpenCV '
          'and NCSDK APIs using Tensorflow or Caffe DNN Models')

    print('./RealTimeObjectDetection [option] [value]')

    print('options:')
    print('--help print this menu')
    print('-d, --device CPU|GPU|MYRIAD|HETERO:CPU,GPU|HETERO:GPU,CPU|HETERO:MYRIAD,GPU : hw platform for openvino')
    print('-d, --device CPU|OPENCL|OPENCL_FP16|MYRIAD : hw platform for opencv, CPU default')
    print('-b, --backend OPENCV_DNN|OPENVINO : hw platform for opencv, OPENCV_DNN default')
    print('-i, --input live|offline : source of video, either webcam or video on disk')
    print('-s, --source <full name to video> : video file full e.g. /home/intel/videos/test.mp4')
    print('-f, --framework openvino|caffe|tensorflow : framework of models being used')
    print('--yolomodel True|False if object detection model is YOLOv3')
    print('--iou_threshold True|False if object detection model is YOLOv3')
    print('--mconfig <full name of caffe prototxt, tensoflow pbtxt, openvino xml> file')
    print('--mweight <full name of caffe caffemodel, tensoflow pb, openvino bin> file')
    print('--mlabels <full name of labels file, each line will have one label>')
    print('--model_image_height DNN model image height to be used for inferrence')
    print('--model_image_width DNN model image width to be used for inferrence')
    print('--model_image_scale DNN model image input scale for inference')
    print('--model_image_mean DNN model image mean substract eg. 127.5,127.5,127.5 ')
    print('-c, --confidence confidence threshold value, default 0.6')
    print('--pc True|False, report performance counters for Deep Learning Layers')
    print('--infer_fc <1, 2 ..> number of frames to infer, by default program tries to infer as much as it can')
    print('--async True|False determine if request is async or not')
    print('--number_of_async_requests 1 to N, enabled if async is being used')
    print('--batch_size 1,2 .. number of frames to infer at once by OpenVINO, default 1')
    print('--openvino_cpulib_path path to libcpu_extension.so')
    print('--openvino_libpath path to openvino libraries')

    return None


def parse_model_labels_from_file(label_file_name=str()):
    """Method used to read each line as a label for a given model

    Args:
        label_file_name: File name of labels

    Returns:
        Returns list of string
    """

    if not os.path.isfile(label_file_name):
        print('Label File Not Found ...')
        help_menu()
        sys.exit(2)

    label_list = list()
    print('Labels:                      ')
    with open(label_file_name) as f:
        lines = f.readlines()
        for line in lines:
            label = line.replace('\n', '')
            label_list.append(label)

    print(label_list)

    return label_list


def get_label_colors(labels_len=0):
    """Method used to generate random RGB values for drawing on the frames for detection layers

    Args:
        labels_len: Length of labels list

    Returns:
        Return list RGB Values [[R,G,B] ...]
    """
    return np.random.uniform(0, 255, (labels_len, 3))


def parse_cli_arguments(argv):
    """Method used to parse command line arguments and set Config fields

    Args:
        argv: Arguments and Options list

    Returns:
        None
    """

    opts = list()

    for i in range(0, len(argv), 2):
        if argv[0] in ('-h', '--help'):
            help_menu()
            sys.exit(0)
        else:
            opts.append((argv[i], argv[i + 1]))

    # Iterate over arguments
    for opt, arg in opts:
        if opt in ('-i', '--input'):
            Config.VIDEOSOURCE = arg
        elif opt in ('-s', '--source'):
            Config.VIDEOPATH = arg
        elif opt in ('-f', '--framework'):
            Config.INFERENCE_FRAMEWORK = arg
        elif opt in ('-d', '--device'):
            Config.INFERENCE_PLATFORM = arg
        elif opt in ('-b', '--backend'):
            Config.OPENCV_INFERENCE_BACKEND = arg
        elif opt in ('-c', '--confidence'):
            Config.CONFIDENCE_THRESHOLD = float(arg)
        elif opt == '--mconfig':
            Config.MODEL_FILE = arg
        elif opt == '--mweight':
            Config.MODEL_WEIGHT_FILE = arg
        elif opt == '--mlabels':
            Config.MODEL_LABELS_FILE = arg
        elif opt == '--infer_fc':
            Config.INFERENCE_FRAMERATE = int(arg)
        elif opt == '--model_image_height':
            Config.MODEL_IMAGE_HEIGHT = int(arg)
        elif opt == '--model_image_width':
            Config.MODEL_IMAGE_WIDTH = int(arg)
        elif opt == '--model_image_mean':
            vals = arg.split(',')
            Config.MODEL_MEANS = [float(vals[0]), float(vals[1]), float(vals[2])]
        elif opt == '--model_image_scale':
            Config.MODEL_IMAGE_SCALE = float(arg)
        elif opt == '--batch_size':
            Config.BATCH_SIZE = int(arg)
        elif opt == '--number_of_async_requests':
            Config.OPENVINO_NUM_REQUESTS = int(arg)
        elif opt == '--async':
            Config.ASYNC = (arg == 'True')
        elif opt == '--openvino_cpulib_path':
            Config.OPENVINO_CPU_LIBPATH = arg
        elif opt == '--openvino_libpath':
            Config.OPENVINO_LIBPATH = arg
        elif opt == '--pc':
            Config.OPENVINO_PERFORMANCE_COUNTER = (arg == 'True')
        elif opt == '--yolomodel':
            Config.YOLO_MODEL_DEFINED = (arg == 'True')
        elif opt == '--iou_threshold':
            Config.YOLO_MODEL_DEFINED = float(arg)
        else:
            print('Unknown argument {} exiting ...'.format(arg))
            sys.exit(2)

    return None


def load_video(source, path):
    """Method used to load given video source with OpenCV VideoCapture

        Args:
            source: Type of video source live (webcam) or file
            path: Path to video source

        Returns:
            cv.VideoCapture object
    """

    if source == 'live':
        print('Loading {} Video '.format(source))
        cap = cv.VideoCapture(0)
        print('Video FPS                       :{}'.format(cap.get(cv.CAP_PROP_FPS)))
        if cap.get(cv.CAP_PROP_FPS) > 0.0:
            Config.FPS_DELAY = int(1000 / cap.get(cv.CAP_PROP_FPS))
        else:
            Config.FPS_DELAY = int(1)
        return cap
    elif source == 'offline':
        print('Loading {} Video '.format(source))
        if not os.path.isfile(path):
            print('Video File Not Found, Exiting ...')
            sys.exit(2)

        cap = cv.VideoCapture(path)
        print('Video FPS                       :{}'.format(cap.get(cv.CAP_PROP_FPS)))
        Config.FPS_DELAY = int(1000 / cap.get(cv.CAP_PROP_FPS))
        return cap
    else:
        print("Unidentified Source Input       :{}".format(source))
        print('-i, --input live|offline : source of video, either webcam or video on disk, Exiting ...')
        sys.exit(2)


def opencv_inference(blob, network):
    """Method used to run inference on given frame with given network

        Args:
            blob: Resized and objectified frame to run forward propagation on the given DNN object
            network: DNN Network loaded

        Returns:
            cv.VideoCapture object
    """

    # Send blob data to Network
    network.setInput(blob)

    # Make network do a forward propagation to get recognition matrix
    out = network.forward()

    return out[0, 0, :, :]


def post_process(frame, detections):
    """Method used to draw rectangels on the detected objects according to defined confidence

        Args:
            frame: Original frame to be shown
            detections: detection list [N, 7] shape

        Returns:
            Processed frame
    """

    for detection in detections:
        # confidence score
        score = float(detection[2])

        # draw rectangle and write the name of the object if above given confidence
        if score > Config.CONFIDENCE_THRESHOLD:
            # label index
            # print(score)

            # print(detection)

            label_index = int(detection[1])

            if label_index >= len(Config.MODEL_LABELS):
                label_text = str(label_index)
            else:
                label_text = Config.MODEL_LABELS[label_index] + " " + str(round(score, 4))

            # print('Possibly Detected: ', label_text, score)

            xmin = detection[3]
            ymin = detection[4]
            xmax = detection[5]
            ymax = detection[6]

            col_factor = Config.IMAGE_WIDTH
            row_factor = Config.IMAGE_HEIGHT

            xmin = (xmin * col_factor)
            xmax = (xmax * col_factor)
            ymin = (ymin * row_factor)
            ymax = (ymax * row_factor)

            cv.putText(frame,
                       label_text,
                       (int(xmin), int(ymin)),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       Config.LABEL_COLORS[label_index],
                       2)

            cv.rectangle(frame,
                         (int(xmin), int(ymin)),
                         (int(xmax), int(ymax)),
                         Config.LABEL_COLORS[label_index],
                         thickness=3)

    return frame


def post_process_yolo(frame, objects):
    """Method used to draw rectangels on the detected objects according to defined confidence when Yolo Model Used

        Args:
            frame: Original frame to be shown
            objects:

        Returns:
            Processed frame
    """
    # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter

    print(objects)

    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > Config.IOU_THRESHOLD:
                objects[j]['confidence'] = 0

    # Drawing objects with respect to the --prob_threshold CLI parameter
    objects = [obj for obj in objects if obj['confidence'] >= Config.CONFIDENCE_THRESHOLD]

    for obj in objects:
        # Validation bbox of detected object
        if obj['xmax'] > frame.shape[1] or obj['ymax'] > frame.shape[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
            continue

        label_index = obj['class_id']

        label_text = Config.MODEL_LABELS[label_index] if Config.MODEL_LABELS and len(
            Config.MODEL_LABELS) >= label_index else \
            str(obj['class_id'])

        label_text += ' ' + str(round(obj['confidence'], 4))

        print(label_text)

        cv.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), Config.LABEL_COLORS[label_index], 3)
        cv.putText(frame, label_text, (obj['xmin'], obj['ymin'] - 7), cv.FONT_HERSHEY_COMPLEX, 0.5,
                   Config.LABEL_COLORS[label_index], 2)

    return frame


def get_openvino_plugin(openvino_network, inference_platform, library_path, cpu_libpath):
    """
    Method used to load IEPlugin according to given target platform
    :param openvino_network: IENetwork object
    :param inference_platform: Target Device Plugin name (CPU, GPU, HETERO:MYRIAD,GPU,CPU etc.
    :param library_path: Lib path to Shared Libraries /opt/intel/openvino/deployment_tools/inference_engine/lib/ubuntu..
    :return: IEPlugin object
    """
    openvino_plugin = None

    # If OpenVINO Selected, Check for Hardware (GPU, MYRIAD or CPU) is supported for this example
    # Load corresponding device library from the indicated paths, this application requires the environment
    # variables are already set correctly
    # source /opt/intel/openvino/bin/setupvars.sh
    if inference_platform == 'GPU':
        print('Trying to Load OpenVINO GPU Plugin')
        openvino_plugin = IEPlugin(device=inference_platform, plugin_dirs=library_path)
    elif inference_platform == 'MYRIAD':
        print('Trying to Load OpenVINO Myriad Plugin')
        openvino_plugin = IEPlugin(device=inference_platform, plugin_dirs=library_path)
        openvino_plugin.set_config({"VPU_FORCE_RESET": "NO"})
    elif inference_platform == 'HETERO:CPU,GPU' or inference_platform == 'HETERO:GPU,CPU':
        openvino_plugin = IEPlugin(device=inference_platform, plugin_dirs=library_path)
        openvino_plugin.add_cpu_extension(cpu_libpath)
        openvino_plugin.set_config({"TARGET_FALLBACK": inference_platform.split(':')[1]})
        # Enable graph visualization
        # openvino_plugin.set_config({"HETERO_DUMP_GRAPH_DOT": "YES"})
        openvino_plugin.set_initial_affinity(openvino_network)
        supported_layers = openvino_plugin.get_supported_layers(openvino_network)
        print('Supported Layers')
        # [print(layer) for layer in supported_layers]
        not_supported_layers = [l for l in openvino_network.layers.keys() if l not in supported_layers]
        print('UnSupported Layers')
        # [print(layer) for layer in not_supported_layers]
    elif inference_platform == 'HETERO:MYRIAD,GPU' or inference_platform == 'HETERO:GPU,MYRIAD':
        openvino_plugin = IEPlugin(device=inference_platform, plugin_dirs=library_path)
        openvino_plugin.set_config({"TARGET_FALLBACK": inference_platform.split(':')[1]})
        # Enable graph visualization
        # openvino_plugin.set_config({"HETERO_DUMP_GRAPH_DOT": "YES"})
        openvino_plugin.set_initial_affinity(openvino_network)
        supported_layers = openvino_plugin.get_supported_layers(openvino_network)
        print('Supported Layers')
        # [print(layer) for layer in supported_layers]
        not_supported_layers = [l for l in openvino_network.layers.keys() if l not in supported_layers]
        print('UnSupported Layers')
        # [print(layer) for layer in not_supported_layers]
    elif inference_platform == 'HETERO:MYRIAD,CPU' or inference_platform == 'HETERO:CPU,MYRIAD':
        openvino_plugin = IEPlugin(device=inference_platform, plugin_dirs=library_path)
        openvino_plugin.add_cpu_extension(cpu_libpath)
        openvino_plugin.set_config({"TARGET_FALLBACK": inference_platform.split(':')[1]})
        # Enable graph visualization
        # openvino_plugin.set_config({"HETERO_DUMP_GRAPH_DOT": "YES"})
        openvino_plugin.set_initial_affinity(openvino_network)
        supported_layers = openvino_plugin.get_supported_layers(openvino_network)
        print('Supported Layers')
        # [print(layer) for layer in supported_layers]
        not_supported_layers = [l for l in openvino_network.layers.keys() if l not in supported_layers]
        print('UnSupported Layers')
        # [print(layer) for layer in not_supported_layers]
    elif inference_platform == "CPU":
        # By default try to load CPU library
        print('Trying to Load OpenVINO CPU Plugin')
        openvino_plugin = IEPlugin(device=inference_platform, plugin_dirs=library_path)
        openvino_plugin.add_cpu_extension(cpu_libpath)
    else:
        print('Undefined Target Platform for OpenVINO: {}'.format(inference_platform))
        help_menu()
        exit(-2)

    return openvino_plugin


def main(argv):
    """Main method runs the application logic

        Args:
            argv: Command line arguments

        Returns:
            None
    """
    # Get process id , required to show CPU load
    process = psutil.Process(os.getpid())

    global inference_time_duration
    inference_time_duration = 0.
    global resize_time_durations
    resize_time_durations = dict()
    global resize_time_duration
    resize_time_duration = 0.
    global inferred_frame_count
    inferred_frame_count = 0
    global frame_read_times
    frame_read_times = 0.
    global frame_display_times
    frame_display_times = 0.
    global post_process_durations

    # Parse cli arguments
    parse_cli_arguments(argv)

    # Read Labels From Given Text File
    Config.MODEL_LABELS = parse_model_labels_from_file(Config.MODEL_LABELS_FILE)
    # Generate Random Colors for each Label
    Config.LABEL_COLORS = get_label_colors(len(Config.MODEL_LABELS))

    # Print Config Summary
    Config.print_summary()

    # Open Video with OpenCV
    cap = load_video(Config.VIDEOSOURCE, Config.VIDEOPATH)

    print("Loaded Video                    :{}".format(Config.VIDEOSOURCE))
    print("Video Path                      :{}".format(Config.VIDEOPATH))

    # Actual Frame Width/Height
    Config.IMAGE_WIDTH = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    Config.IMAGE_HEIGHT = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    print("Video Resolution                :{} x {}".format(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))

    # Deep Learning Network Object
    openvino_net = None
    openvino_plugin = None

    # OpenVINO Input/Output Definitions
    input_blob = None
    out_blob = None

    # OpenVINO Input Properties (Number of Inputs, Channels, Height, Width)
    n = 0
    c = 0
    h = 0
    w = 0

    # Request ID Queue for Async Inference
    request_ids = list()

    network_load_time_start = time.time()
    # Select Framework according to Options
    if Config.INFERENCE_FRAMEWORK == 'openvino':
        print('OpenVINO Framework Selected ...')

        # Read Inference Engine Network with given .bin/.xml files
        print('Loading DL Model Files          : {} - {}'.format(Config.MODEL_FILE, Config.MODEL_WEIGHT_FILE))
        network = IENetwork(model=Config.MODEL_FILE, weights=Config.MODEL_WEIGHT_FILE)

        openvino_plugin = get_openvino_plugin(network,
                                              Config.INFERENCE_PLATFORM,
                                              Config.OPENVINO_LIBPATH,
                                              Config.OPENVINO_CPU_LIBPATH)

        input_blob = next(iter(network.inputs))
        print('OpenVINO Model Input Blob       :', type(input_blob))

        n, c, h, w = network.inputs[input_blob].shape
        Config.MODEL_IMAGE_HEIGHT = h
        Config.MODEL_IMAGE_WIDTH = w
        print('Input Properties')
        print('Batch                           :{}'.format(n))
        print('Channels                        :{}'.format(c))
        print('Height                          :{}'.format(h))
        print('Width                           :{}'.format(w))

        out_blob = next(iter(network.outputs))
        print('OpenVINO Model Output Blob      :', type(out_blob))

        network.batch_size = Config.BATCH_SIZE
        print('Batch Size                      :', network.batch_size)

        print("Loading Given Model with IEPlugin ...")
        openvino_net = openvino_plugin.load(network=network, num_requests=Config.OPENVINO_NUM_REQUESTS)

        if Config.ASYNC:
            request_ids = list(np.arange(0, Config.OPENVINO_NUM_REQUESTS))
            print("Number of Requests to Handle    :", Config.OPENVINO_NUM_REQUESTS)
        else:
            request_ids.append(0)

        if openvino_net is None:
            print('Can not Load Given Network, Exiting ....')
            exit(-2)

    elif Config.INFERENCE_FRAMEWORK == 'tensorflow':
        print('OpenCV DNN will use Tensoflow Models for Inference')
        network = cv.dnn.readNetFromTensorflow(Config.MODEL_WEIGHT_FILE, Config.MODEL_FILE)

    elif Config.INFERENCE_FRAMEWORK == 'caffe':
        print('OpenCV DNN will use Caffe Models for Inference')
        network = cv.dnn.readNetFromCaffe(Config.MODEL_FILE, Config.MODEL_WEIGHT_FILE)

    else:
        print("{} Framework Not Supported, Exiting ...".format(Config.INFERENCE_FRAMEWORK))
        help_menu()
        sys.exit(2)

    if Config.INFERENCE_FRAMEWORK == 'tensorflow' or Config.INFERENCE_FRAMEWORK == 'caffe':
        print('Setting OpenCV Backend and Target Device ...')
        if Config.OPENCV_INFERENCE_BACKEND == 'OPENVINO':
            network.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        elif Config.OPENCV_INFERENCE_BACKEND == 'OPENCV_DNN':
            network.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        else:
            print('Undefined OpenCV Backend: {}'.format(Config.OPENCV_INFERENCE_BACKEND))
            help_menu()
            sys.exit(2)

        if Config.INFERENCE_PLATFORM == 'OPENCL':
            network.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
        elif Config.INFERENCE_PLATFORM == 'OPENCL_FP16':
            network.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL_FP16)
        elif Config.INFERENCE_PLATFORM == 'MYRIAD':
            network.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
        elif Config.INFERENCE_PLATFORM == 'CPU':
            network.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        else:
            print('Undefined OpenCV Target Device: {}'.format(Config.INFERENCE_PLATFORM))
            help_menu()
            sys.exit(2)

    network_load_time_end = time.time()
    # Start Counting frames to Calculate FPS

    detections = None

    cur_request_id = 0
    next_request_id = 1
    # Queue to be used for request ids
    if Config.INFERENCE_FRAMEWORK == 'openvino' and Config.ASYNC:
        cur_request_id = request_ids.pop(0)
        next_request_id = request_ids.pop(0)

    next_frame = None
    # Continuous loop to read frames
    has_frame, frame = cap.read()

    openvino_detection_starts = dict()
    frame_count = 0
    start_time = time.time()

    cpu_count = psutil.cpu_count()

    # Start Reading Frames
    while True:
        # read frame from capture
        frame_read_start = time.time()
        if Config.ASYNC:
            has_frame, next_frame = cap.read()
        else:
            has_frame, frame = cap.read()
        frame_read_end = time.time()
        frame_read_times += (frame_read_end - frame_read_start)

        if not has_frame:
            break

        yolo_objects = list()

        if Config.INFERENCE_FRAMEWORK == 'openvino':
            if Config.ASYNC:
                # Read and pre-process input images
                resize_start = time.time()
                resized_frame = cv.resize(next_frame, (Config.MODEL_IMAGE_HEIGHT, Config.MODEL_IMAGE_WIDTH))
                resized_frame = resized_frame.transpose((2, 0, 1))  # Change layout to HWC
                resized_frame = resized_frame.reshape((n, c, h, w))
                resize_end = time.time()
                resize_time_duration += (resize_end - resize_start)

                infer_start = time.time()
                openvino_net.start_async(request_id=next_request_id, inputs={input_blob: resized_frame})
                openvino_detection_starts[next_request_id] = infer_start

            else:
                resize_start = time.time()
                resized_frame = cv.resize(frame, (Config.MODEL_IMAGE_HEIGHT, Config.MODEL_IMAGE_WIDTH))
                resized_frame = resized_frame.transpose((2, 0, 1))  # Change layout to HWC
                resized_frame = resized_frame.reshape((n, c, h, w))
                resize_end = time.time()
                resize_time_duration += (resize_end - resize_start)

                infer_start = time.time()
                openvino_net.infer(inputs={input_blob: resized_frame})
                openvino_detection_starts[cur_request_id] = infer_start

            if openvino_net.requests[cur_request_id].wait(-1) == 0:
                if not Config.YOLO_MODEL_DEFINED:
                    openvino_detections = openvino_net.requests[cur_request_id].outputs[out_blob]
                    detections = openvino_detections[0][0]

                else:
                    output = openvino_net.requests[cur_request_id].outputs

                    for layer_name, out_blob in output.items():
                        layer_params = YoloV3Params(network.layers[layer_name].params, out_blob.shape[2])
                        # print("Layer {} parameters: ".format(layer_name))
                        layer_params.log_params()
                        yolo_objects += parse_yolo_region(out_blob,
                                                          resized_frame.shape[2:],
                                                          Config.IMAGE_HEIGHT,
                                                          Config.IMAGE_WIDTH,
                                                          layer_params,
                                                          Config.CONFIDENCE_THRESHOLD)
                detection_ends = time.time()
                inference_time_duration += (detection_ends - openvino_detection_starts[cur_request_id])
                inferred_frame_count += 1

        elif Config.INFERENCE_FRAMEWORK == 'tensorflow' or Config.INFERENCE_FRAMEWORK == 'caffe':
            resize_start = time.time()
            resized_frame = cv.resize(frame, (Config.MODEL_IMAGE_HEIGHT, Config.MODEL_IMAGE_WIDTH))

            # MobileNetSSD Expects 224x224 resized frames
            blob = cv.dnn.blobFromImage(resized_frame,
                                        Config.MODEL_SCALE,
                                        (Config.MODEL_IMAGE_HEIGHT, Config.MODEL_IMAGE_WIDTH),
                                        (Config.MODEL_MEANS[0], Config.MODEL_MEANS[1], Config.MODEL_MEANS[2]))

            resize_end = time.time()
            resize_time_duration += (resize_end - resize_start)

            infer_start = time.time()
            detections = opencv_inference(blob, network)
            infer_end = time.time()
            inference_time_duration += (infer_end - infer_start)
            inferred_frame_count += 1
        else:
            print('Framework Not Found, Exiting ...')
            sys.exit(2)

        # Post Process over Detections
        post_process_start = time.time()
        if detections is not None and not Config.YOLO_MODEL_DEFINED:
            post_process(frame, detections)

        if yolo_objects is not None and Config.YOLO_MODEL_DEFINED:
            post_process_yolo(frame, yolo_objects)

        # display text to let user know how to quit
        cv.rectangle(frame, (0, 0), (220, 60), (50, 50, 50, 100), -1)
        cv.putText(frame,
                   "Q to Quit",
                   (10, 12),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.4,
                   (255, 255, 255),
                   1)
        # CPU Load
        current_cpu_load = process.cpu_percent()
        cpu_load = current_cpu_load / cpu_count
        cv.putText(frame,
                   'CPU Load %: {} '.format(cpu_load),
                   (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.4,
                   (255, 255, 255),
                   1)
        current_end = time.time()
        current_fps = frame_count / (current_end - start_time)
        cv.putText(frame,
                   'FPS : {} '.format((round(current_fps, 3))),
                   (10, 38),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.4,
                   (255, 255, 255),
                   1)

        cv.imshow('Real Time Object Detection', frame)

        if Config.ASYNC:
            request_ids.append(cur_request_id)
            cur_request_id = next_request_id
            next_request_id = request_ids.pop(0)
            frame = next_frame

        if cv.waitKey(Config.FPS_DELAY) & 0xFF == ord('q'):
            break

        post_process_end = time.time()
        global post_process_durations
        post_process_durations += post_process_end - post_process_start
        frame_count += 1

    # Summarize Performance Metrics
    end_time = time.time()
    elapsed_time = end_time - start_time
    network_load_time = network_load_time_end - network_load_time_start

    print('Total Execution Time             :',
          elapsed_time, ' Seconds')
    print('Processed Frame Count            :',
          inferred_frame_count, ' Frames')
    print('Network Load Time: ' +
          str(network_load_time) + ' Seconds')
    print('Reading 1 Frame in               :' +
          str(round((frame_read_times / frame_count) * 1000, 3)) + ' Milliseconds')
    print('Frames Per Second                :' +
          str(round(frame_count / elapsed_time, 3)))
    print('Pre-process for 1 Frame          :' +
          str(round((resize_time_duration / inferred_frame_count) * 1000, 3)),
          ' milliseconds')

    global post_process_durations
    if not Config.ASYNC:
        print('Inference for 1 Frame        :' +
              str(round((inference_time_duration / inferred_frame_count) * 1000, 3)),
              ' milliseconds')
    else:
        print('Inference for 1 Frame        :',
              str(round(((elapsed_time - frame_read_times -
                          resize_time_duration - post_process_durations)
                         / frame_count) * 1000, 3)),
              ' milliseconds')

    print('Post-process for 1 Frame         :' +
          str(round((post_process_durations / inferred_frame_count) * 1000, 3)),
          ' milliseconds (including display, key wait time ...)')

    print('Final Time Table in Milliseconds')
    print('Elapsed Time - '
          'Frame Read Time - Pre Process Time - '
          'Infer Time - Post Process Time')

    print('{} - {} - {} - {} - {} \n'.format(elapsed_time * 1000.,
                                             frame_read_times * 1000,
                                             resize_time_duration * 1000,
                                             inference_time_duration * 1000,
                                             post_process_durations * 1000))

    # print('Total Elapsed Time: {} Milliseconds'.format(elapsed_time * 1000))

    # time_sums = frame_display_times + resize_time_duration + \
    #             inference_time_duration + post_process_durations

    # print('Sum of Measured Time: {} Milliseconds'.format(time_sums * 1000))

    # When frames finished
    if Config.INFERENCE_FRAMEWORK == 'openvino' and Config.OPENVINO_PERFORMANCE_COUNTER:
        print("No more frame from from video source, exiting ....")

        perf_counts = openvino_net.requests[0].get_perf_counts()
        print("Performance counters:")
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name',
                                                          'layer_type',
                                                          'exet_type',
                                                          'status',
                                                          'real_time, us'))
        for layer, stats in perf_counts.items():
            print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'],
                                                              stats['status'], stats['real_time']))

    # Release Resources
    cv.destroyAllWindows()
    cap.release()

    del openvino_net
    del network
    del openvino_plugin


# Application Entry Point
if __name__ == "__main__":
    main(sys.argv[1:])
