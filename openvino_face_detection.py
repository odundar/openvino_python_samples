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

import time
import cv2 as cv
import argparse
import numpy as np

# Import OpenVINO Inference Engine
from openvino.inference_engine import IECore, IENetwork

global arguments


def crop_frame(frame, coordinate, normalized=True):
    """
    Crop Frame as Given Coordinates
    :param frame:
    :param coordinate:
    :param normalized:
    :return:
    """

    x1 = coordinate[0]
    y1 = coordinate[1]
    x2 = coordinate[2]
    y2 = coordinate[3]

    if normalized:
        h = frame.shape[0]
        w = frame.shape[1]

        x1 = int(x1 * w)
        x2 = int(x2 * w)

        y1 = int(y1 * h)
        y2 = int(y2 * h)

    return frame[y1:y2, x1:x2]


def run_app():
    # Load IECore Object
    OpenVinoIE = IECore()
    print("Available Devices: ", OpenVinoIE.available_devices)

    # Load CPU Extensions if Necessary
    if "CPU" in arguments.face_target_device or "CPU" in arguments.ag_target_device:
        OpenVinoIE.add_extension('/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.so', "CPU")

    print('Loading Face Detection Model   ....')
    # Load Network
    FaceDetectionNetwork = IENetwork(model=arguments.face_model_xml, weights=arguments.face_model_bin)
    # Get Input Layer Information
    FaceDetectionInputLayer = next(iter(FaceDetectionNetwork.inputs))
    print("Face Detection Input Layer: ", FaceDetectionInputLayer)
    # Get Output Layer Information
    FaceDetectionOutputLayer = next(iter(FaceDetectionNetwork.outputs))
    print("Face Detection Output Layer: ", FaceDetectionOutputLayer)
    # Get Input Shape of Model
    FaceDetectionInputShape = FaceDetectionNetwork.inputs[FaceDetectionInputLayer].shape
    print("Face Detection Input Shape: ", FaceDetectionInputShape)
    # Get Output Shape of Model
    FaceDetectionOutputShape = FaceDetectionNetwork.outputs[FaceDetectionOutputLayer].shape
    print("Face Detection Output Shape: ", FaceDetectionOutputShape)

    # Load Executable Network
    FaceDetectionExecutable = OpenVinoIE.load_network(network=FaceDetectionNetwork, device_name=arguments.face_target_device)

    print('Loading Age - Gender Detection Model  .....')
    # Load Network
    AgeGenderDetectionNetwork = IENetwork(model=arguments.ag_model_xml, weights=arguments.ag_model_bin)
    # Get Input Layer Information
    AgeGenderDetectionInputLayer = next(iter(AgeGenderDetectionNetwork.inputs))
    print("Age Gender Detection Input Layer: ", AgeGenderDetectionInputLayer)
    # Get Output Layer Information
    AgeGenderDetectionOutputLayers = list(AgeGenderDetectionNetwork.outputs.keys())
    print("Age Gender Detection Output Layer: ", AgeGenderDetectionOutputLayers)
    # Get Input Shape of Model
    AgeGenderDetectionInputShape = AgeGenderDetectionNetwork.inputs[AgeGenderDetectionInputLayer].shape
    print("Age Gender Detection Input Shape: ", AgeGenderDetectionInputShape)
    # Get Output Shape of Model
    AgeOutputShape = AgeGenderDetectionNetwork.outputs[AgeGenderDetectionOutputLayers[0]].shape
    print("Age Gender Detection Output Layer {} Shape: ".format(AgeGenderDetectionOutputLayers[0]), AgeOutputShape)
    GenderOutputShape = AgeGenderDetectionNetwork.outputs[AgeGenderDetectionOutputLayers[0]].shape
    print("Age Gender Detection Output Layer {} Shape: ".format(AgeGenderDetectionOutputLayers[1]), GenderOutputShape)

    # Set Maximum Batch Size for Age-Gender Detection
    AgeGenderDetectionNetwork.batch_size = int(arguments.ag_max_batch_size)

    # Check if Dynamic Batching Enabled for Age Gender Detection
    config = {}

    # Get the Batch Size and Allocate Input for Dynamic Batch Process
    NAG, CAG, HAG, WAG = AgeGenderDetectionNetwork.inputs[AgeGenderDetectionInputLayer].shape

    if arguments.ag_dynamic_batch:
        config = {"DYN_BATCH_ENABLED": "YES"}
        print("Dynamic Batch Enabled")

    if NAG > 1:
        age_detection_input = np.zeros(shape=(NAG, CAG, HAG, WAG), dtype=float)

    # Load Executable Network
    AgeGenderDetectionExecutable = OpenVinoIE.load_network(network=AgeGenderDetectionNetwork, config=config, device_name=arguments.ag_target_device)

    # Get Shape Values for Face Detection Network
    N, C, H, W = FaceDetectionNetwork.inputs[FaceDetectionInputLayer].shape

    # Generate a Named Window
    cv.namedWindow('Window', cv.WINDOW_NORMAL)
    cv.resizeWindow('Window', 800, 600)

    # Enables Single Image Inference ...
    if arguments.input_type == 'image':
        # Read Image
        image = cv.imread(arguments.input)
        fh = image.shape[0]
        fw = image.shape[1]

        # Pre-process Image
        resized = cv.resize(image, (W, H))
        resized = resized.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        input_image = resized.reshape((N, C, H, W))

        # Start Inference
        fdetect_start = time.time()
        results = FaceDetectionExecutable.infer(inputs={FaceDetectionInputLayer: input_image})
        fdetect_end = time.time()

        inf_time = fdetect_end - fdetect_start
        fps = 1./inf_time
        # Write Information on Image
        text = 'Face Detection - FPS: {}, INF: {} Sec'.format(round(fps, 2), round(inf_time, 4))
        print(text)
        cv.putText(image, text, (0, 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 125, 255), 1)

        # Get Results
        detections = results[FaceDetectionOutputLayer][0][0]

        # Get Faces
        i = 0
        face_coordinates = list()
        face_frames = list()
        for detection in detections:
            if detection[2] > arguments.face_detection_threshold:
                # Get Coordinates
                xmin = int(detection[3] * fw)
                ymin = int(detection[4] * fh)
                xmax = int(detection[5] * fw)
                ymax = int(detection[6] * fh)
                coordinates = [xmin, ymin, xmax, ymax]
                face_coordinates.append(coordinates)

                # Crop Face
                face_image = crop_frame(frame=image, coordinate=coordinates, normalized=False)
                r_frame = cv.resize(face_image, (WAG, HAG))
                r_frame = cv.cvtColor(r_frame, cv.COLOR_BGR2RGB)
                r_frame = np.transpose(r_frame, (2, 0, 1))
                r_frame = np.expand_dims(r_frame, axis=0)

                if NAG > 1:
                    age_detection_input[i-1:i, ] = r_frame
                else:
                    face_frames.append(r_frame)

                i += 1
                cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 125, 255), 3)
        print('Detected {} Faces'.format(i))
        ag_total_infer_time = 0.0
        if NAG > 1 and arguments.ag_dynamic_batch:
            AgeGenderDetectionExecutable.requests[0].set_batch(i)
            ag_infer_start = time.time()
            AgeGenderDetectionExecutable.infer({AgeGenderDetectionInputLayer: age_detection_input})
            ag_infer_end = time.time()
            ag_total_infer_time = (ag_infer_end - ag_infer_start)
            for f in range(age_detection_input.shape[0]):
                age = int(
                    AgeGenderDetectionExecutable.requests[0].outputs[AgeGenderDetectionOutputLayers[0]][f][0][0][
                        0] * 100)
                gender = 'male'
                if AgeGenderDetectionExecutable.requests[0].outputs[AgeGenderDetectionOutputLayers[1]][f][0][0][0] > \
                        AgeGenderDetectionExecutable.requests[0].outputs[AgeGenderDetectionOutputLayers[1]][f][1][0][0]:
                    gender = 'female'

                text = "Face {} - A: {} - G: {}".format(f, age, gender)
                print(text)
                cv.putText(image, text, (face_coordinates[f][0], face_coordinates[f][1] - 7), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)

            fps = i / ag_total_infer_time
            inf_time = ag_total_infer_time / i
            text = 'AG Detection FPS: {}, INF: {} Sec Per Face'.format(round(fps, 2), round(inf_time, 4))
            print(text)
            cv.putText(image, text, (0, 40), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 125, 255), 1)
        else:
            f = 0
            for face in face_frames:
                ag_infer_start = time.time()
                AgeGenderDetectionExecutable.infer({AgeGenderDetectionInputLayer: face})
                ag_infer_end = time.time()
                ag_total_infer_time += (ag_infer_end - ag_infer_start)
                age = int(
                    AgeGenderDetectionExecutable.requests[0].outputs[AgeGenderDetectionOutputLayers[0]][0][0][0][
                        0] * 100)

                gender = 'male'
                if AgeGenderDetectionExecutable.requests[0].outputs[AgeGenderDetectionOutputLayers[1]][0][0][0][0] > \
                        AgeGenderDetectionExecutable.requests[0].outputs[AgeGenderDetectionOutputLayers[1]][0][1][
                            0][0]:
                    gender = 'female'

                text = "Face {} - A: {} - G: {}".format(f, age, gender)
                print(text)
                cv.putText(image, text, (face_coordinates[f][0], face_coordinates[f][1] - 7), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)
                f += 1

            fps = 1.0 / ag_total_infer_time
            inf_time = ag_total_infer_time
            text = 'Age Detection - FPS: {}, INF: {} Sec'.format(round(fps, 2), round(inf_time, 4))
            print(text)
            cv.putText(image, text, (0, 40), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 125, 255), 1)

        cv.imshow('Window', image)
        cv.waitKey(0)

    # Enables Video File or Web Cam
    else:
        # Implementation for CAM or Video File
        # Read Image
        capture = cv.VideoCapture(arguments.input)
        has_frame, frame = capture.read()

        fh = frame.shape[0]
        fw = frame.shape[1]
        print('Original Frame Shape: ', fw, fh)

        # Variables to Hold Inference Time Information
        total_ag_inference_time = 0
        inferred_face_count = 0

        while has_frame:
            # Pre-process Image
            resized = cv.resize(frame, (W, H))
            resized = resized.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            input_image = resized.reshape((N, C, H, W))

            # Start Inference
            fdetect_start = time.time()
            results = FaceDetectionExecutable.infer(inputs={FaceDetectionInputLayer: input_image})
            fdetect_end = time.time()
            inf_time = fdetect_end - fdetect_start
            fps = 1. / inf_time
            # Write Information on Image
            text = 'Face Detection - FPS: {}, INF: {}'.format(round(fps, 2), round(inf_time, 4))
            cv.putText(frame, text, (0, 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 125, 255), 1)

            # Print Bounding Boxes on Image
            detections = results[FaceDetectionOutputLayer][0][0]

            face_count = 0
            face_coordinates = list()
            face_frames = list()
            # Check All Detections
            for detection in detections:
                if detection[2] > arguments.face_detection_threshold:
                    # Crop Frame
                    xmin = int(detection[3] * fw)
                    ymin = int(detection[4] * fh)
                    xmax = int(detection[5] * fw)
                    ymax = int(detection[6] * fh)

                    coordinates = [xmin, ymin, xmax, ymax]
                    face_coordinates.append(coordinates)
                    face = crop_frame(frame=frame, coordinate=coordinates, normalized=False)
                    r_frame = cv.resize(face, (WAG, HAG))
                    r_frame = cv.cvtColor(r_frame, cv.COLOR_BGR2RGB)
                    r_frame = np.transpose(r_frame, (2, 0, 1))
                    r_frame = np.expand_dims(r_frame, axis=0)

                    if NAG > 1:
                        age_detection_input[face_count - 1:face_count, ] = r_frame
                    else:
                        face_frames.append(r_frame)

                    face_count += 1
                    cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 125, 255), 3)

            agdetect_start = time.time()
            inferred_face_count += face_count

            if face_count > 0 and NAG > 1:
                if arguments.ag_dynamic_batch:
                    AgeGenderDetectionExecutable.requests[0].set_batch(face_count)

                AgeGenderDetectionExecutable.infer({AgeGenderDetectionInputLayer: age_detection_input})
                for f in range(face_count):
                    age = int(
                        AgeGenderDetectionExecutable.requests[0].outputs[AgeGenderDetectionOutputLayers[0]][0][0][0][
                            0] * 100)

                    gender = 'male'
                    if AgeGenderDetectionExecutable.requests[0].outputs[AgeGenderDetectionOutputLayers[1]][0][0][0][0] > \
                            AgeGenderDetectionExecutable.requests[0].outputs[AgeGenderDetectionOutputLayers[1]][0][1][
                                0][0]:
                        gender = 'female'

                    text = "A: {} - G: {}".format(age, gender)
                    cv.putText(frame, text, (face_coordinates[f][0], face_coordinates[f][1] - 7), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)

                agdetect_end = time.time()
                # Write Information on Image
                inf_time = (agdetect_end - agdetect_start) / face_count
                fps = face_count / inf_time
            elif face_count > 0:
                f = 0
                for face in face_frames:
                    AgeGenderDetectionExecutable.infer({AgeGenderDetectionInputLayer: face})
                    age = int(AgeGenderDetectionExecutable.requests[0].outputs[AgeGenderDetectionOutputLayers[0]][0][0][0][0] * 100)

                    gender = 'male'
                    if AgeGenderDetectionExecutable.requests[0].outputs[AgeGenderDetectionOutputLayers[1]][0][0][0][0] > AgeGenderDetectionExecutable.requests[0].outputs[AgeGenderDetectionOutputLayers[1]][0][1][0][0]:
                        gender = 'female'

                    text = "A: {} - G: {}".format(age, gender)
                    cv.putText(frame, text, (face_coordinates[f][0], face_coordinates[f][1] - 7), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)

                    f += 1

                agdetect_end = time.time()
                # Write Information on Image
                inf_time = (agdetect_end - agdetect_start) / f
                fps = f / inf_time

            if face_count > 0:
                text = 'AG Detection - FPS: {}, INF Per Face: {}'.format(round(fps, 2), round(inf_time, 4))
                cv.putText(frame, text, (0, 40), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 125, 255), 1)
                total_ag_inference_time += inf_time

            cv.imshow('Window', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            has_frame, frame = capture.read()

        print('Total AG Inference Time : {}'.format(total_ag_inference_time))
        print('Number of Face Inferred : {}'.format(inferred_face_count))
        print('Average AG Inference Time : {}'.format(total_ag_inference_time/inferred_face_count))


"""
Entry Point of Application
"""
if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Basic OpenVINO Example to Face, Age and Gender Detection')
    parser.add_argument('--face-model-xml',
                        default='/home/intel/openvino_models/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.xml',
                        help='Face Detection Model XML File')
    parser.add_argument('--face-model-bin',
                        default='/home/intel/openvino_models/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.bin',
                        help='Face Detection Model BIN File')
    parser.add_argument('--face-detection-threshold', default=0.5, help='Face Detection Accuracy Threshold')

    parser.add_argument('--face-target-device', default='CPU',
                        help='Target Plugin: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA,CPU')

    parser.add_argument('--ag-model-xml',
                        default='/home/intel/openvino/Retail/object_attributes/age_gender/dldt/FP32/age-gender-recognition-retail-0013.xml',
                        help='Age-Gender Detection XML File')
    parser.add_argument('--ag-model-bin',
                        default='/home/intel/openvino/Retail/object_attributes/age_gender/dldt/FP32/age-gender-recognition-retail-0013.bin',
                        help='Age-Gender Detection BIN File')

    parser.add_argument('--ag-max-batch-size', default=1, help='Age Gender Detection Max Batch Size')
    parser.add_argument('--ag-dynamic-batch', action="store_true", default=False, help='Age Gender Detection Enable Dynamic Batching')
    parser.add_argument('--ag-target-device', default='CPU',
                        help='Target Plugin: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA,CPU')

    parser.add_argument('--input-type', default='image', help='Type of Input: image, video, cam')
    parser.add_argument('--input', default='/home/intel/Pictures/faces.jpg',
                        help='Path to Input: WebCam: 0, Video File or Image file')

    arguments = parser.parse_args()
    print('WARNING: No Argument Control Done, You Can GET Runtime Errors')
    run_app()
