## Introduction

This repo contains couple python sample applications to teach about Intel(R) Distribution of OpenVINO(TM).

## Object Detection Application

`openvino_basic_object_detection.py`  

This application intends to showcase how a model is being used with OpenVINO(TM) Toolkit. 

Aim is to show initial use case of Inference Engine API and Async Mode.

```bash
usage: openvino_basic_object_detection.py [-h] [--model-xml MODEL_XML]
                                          [--model-bin MODEL_BIN]
                                          [--target-device TARGET_DEVICE]
                                          [--input-type INPUT_TYPE]
                                          [--input INPUT]
                                          [--detection-threshold DETECTION_THRESHOLD]
                                          [--async]
                                          [--request-number REQUEST_NUMBER]

Basic OpenVINO Example with MobileNet-SSD

optional arguments:
  -h, --help            show this help message and exit
  --model-xml MODEL_XML
                        XML File
  --model-bin MODEL_BIN
                        BIN File
  --target-device TARGET_DEVICE
                        Target Plugin: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU,
                        HETERO:FPGA,CPU
  --input-type INPUT_TYPE
                        Type of Input: image, video, cam
  --input INPUT         Path to Input: WebCam: 0, Video File or Image file
  --detection-threshold DETECTION_THRESHOLD
                        Object Detection Accuracy Threshold
  --async               Run Async Mode
  --request-number REQUEST_NUMBER
                        Number of Requests
```

### mobilenet-ssd Model Download & Conversion

Downloaded model with OpenVINO(TM) Toolkit Model Downloader

```bash
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py \
--name mobilenet-ssd \
--output_dir ~/openvino_models/
```

Converted Model using OpenVINO(TM) Toolkit Model Optimizer

- FP32 Conversion

```bash
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
--input_model ~/openvino_models/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel \
--output_dir ~/openvino_models/object_detection/common/mobilenet-ssd/FP32 \
--data_type FP32 \
--scale 127.50223128904757 \
--input_shape [1,3,300,300] \
--mean_values [127.5,127.5,127.5] \
--framework caffe
``` 

- FP16 Conversion

```bash
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
--input_model ~/openvino_models/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel \
--output_dir ~/openvino_models/object_detection/common/mobilenet-ssd/FP16 \
--data_type FP16 \
--scale 127.50223128904757 \
--input_shape [1,3,300,300] \
--mean_values [127.5,127.5,127.5] \
--framework caffe
```

## Face Detection Application

`openvino_basic_object_detection.py`  

This application intends to showcase how two model from Open Model Zoo (https://github.com/opencv/open_model_zoo) is being used.

Aim is to show initial use case of Inference Engine API Dynamic Batching with using Face Detection and Gender-Age Detection.

```bash
usage: openvino_face_detection.py [-h] [--face-model-xml FACE_MODEL_XML]
                                  [--face-model-bin FACE_MODEL_BIN]
                                  [--face-detection-threshold FACE_DETECTION_THRESHOLD]
                                  [--face-target-device FACE_TARGET_DEVICE]
                                  [--ag-model-xml AG_MODEL_XML]
                                  [--ag-model-bin AG_MODEL_BIN]
                                  [--ag-max-batch-size AG_MAX_BATCH_SIZE]
                                  [--ag-dynamic-batch]
                                  [--ag-target-device AG_TARGET_DEVICE]
                                  [--input-type INPUT_TYPE] [--input INPUT]

Basic OpenVINO Example to Face, Age and Gender Detection

optional arguments:
  -h, --help            show this help message and exit
  --face-model-xml FACE_MODEL_XML
                        Face Detection Model XML File
  --face-model-bin FACE_MODEL_BIN
                        Face Detection Model BIN File
  --face-detection-threshold FACE_DETECTION_THRESHOLD
                        Face Detection Accuracy Threshold
  --face-target-device FACE_TARGET_DEVICE
                        Target Plugin: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU,
                        HETERO:FPGA,CPU
  --ag-model-xml AG_MODEL_XML
                        Age-Gender Detection XML File
  --ag-model-bin AG_MODEL_BIN
                        Age-Gender Detection BIN File
  --ag-max-batch-size AG_MAX_BATCH_SIZE
                        Age Gender Detection Max Batch Size
  --ag-dynamic-batch    Age Gender Detection Enable Dynamic Batching
  --ag-target-device AG_TARGET_DEVICE
                        Target Plugin: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU,
                        HETERO:FPGA,CPU
  --input-type INPUT_TYPE
                        Type of Input: image, video, cam
  --input INPUT         Path to Input: WebCam: 0, Video File or Image file
```

### Face Detection Model

Downloaded Face Detection model as below:

```bash
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py \
--name face-detection-retail* \
--output_dir ~/openvino_models/
```

Detailed information about the model:

- https://github.com/opencv/open_model_zoo/blob/master/intel_models/face-detection-retail-0004/description/face-detection-retail-0004.md

### Age Gender Detection Model

Downloaded Age Gender Detection Model as below:

```bash
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py \
--name age-gender-recognition-retail* \
--output_dir ~/openvino_models/
```

- https://github.com/opencv/open_model_zoo/blob/master/intel_models/age-gender-recognition-retail-0013/description/age-gender-recognition-retail-0013.md

## Object Detection Application with Configurations

`openvino_object_detection_configs.py`

This application is similar to Object Detection Application above, however some additional configurations options added to see the performance and understand their effect to overall performance.

```bash
usage: openvino_object_detection_configs.py [-h] [--model-xml MODEL_XML]
                                            [--model-bin MODEL_BIN]
                                            [--target-device TARGET_DEVICE]
                                            [--input-type INPUT_TYPE]
                                            [--input INPUT]
                                            [--detection-threshold DETECTION_THRESHOLD]
                                            [--async]
                                            [--request-number REQUEST_NUMBER]
                                            [--pc]
                                            [--cpu-num-threads CPU_NUM_THREADS]
                                            [--cpu-bind-thread]
                                            [--cpu-throughput-streams CPU_THROUGHPUT_STREAMS]
                                            [--gpu-throughput-streams GPU_THROUGHPUT_STREAMS]
                                            [--gpu-throttle]

Basic OpenVINO Example to Test Object Detection Model

optional arguments:
  -h, --help            show this help message and exit
  --model-xml MODEL_XML
                        XML File
  --model-bin MODEL_BIN
                        BIN File
  --target-device TARGET_DEVICE
                        Target Plugin: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU,
                        HETERO:FPGA,CPU
  --input-type INPUT_TYPE
                        Type of Input: image, video, cam
  --input INPUT         Path to Input: WebCam: 0, Video File or Image file
  --detection-threshold DETECTION_THRESHOLD
                        Object Detection Accuracy Threshold
  --async               Run Async Mode
  --request-number REQUEST_NUMBER
                        Number of Requests
  --pc                  Enable Performance Counters
  --cpu-num-threads CPU_NUM_THREADS
                        Limit CPU Threads
  --cpu-bind-thread     Bind Threads to CPU
  --cpu-throughput-streams CPU_THROUGHPUT_STREAMS
                        Int Values or CPU_THROUGHPUT_NUMA if not set
                        CPU_THROUGHPUT_AUTO
  --gpu-throughput-streams GPU_THROUGHPUT_STREAMS
                        Int Values if not set GPU_THROUGHPUT_AUTO
  --gpu-throttle        multi-device execution with the CPU+GPU performs best
                        with GPU trottling hint
```

## Object Detection Multi Framework Sample

This example designed to run with mobilenet-ssd model, it is being used to work together with multiple target devices.

Example application also determines how to use OpenCV DNN with mobilenet-ssd which has been downloaded with OpenVINO.

Object Detection Model:

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name mobilenet-ssd --output_dir /home/intel/openvino


Face Detection Model:

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face* --output_dir /home/intel/openvino


Age-Gender Detection Model:

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name age* --output_dir /home/intel/openvino


#### Running Object Detection Example

##### Step 1 

Install Intel Distribution of OpenVINO

I will not get into details because all the steps are already covered in Intel site with detail.

https://software.intel.com/en-us/articles/OpenVINO-Install-Linux 

**Note**: Please don't skip any dependency installation you can get errors at the following stages. 


##### Step 2 

Download `mobilenet-ssd` with OpenVINO Model Downloader

```bash
python3 /opt/intel/computer_vision_sdk/deployment_tools/model_downloader/downloader.py --name mobilenet-ssd --output_dir "<path>"
```

##### Step 3

Download & Convert MobileNet SSD With Model Downloader and Optimizer.

*Note*: Change `<path>` and output location according to your environment

- FP32 Conversion for CPU/GPU

```bash
python3 /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py --input_model "<path>/mobilenet-ssd.caffemodel" --output_dir /home/intel/openvino_models/ir/mobilenet-ssd/FP32 --data_type FP32 --scale 256 --mean_values [127, 127, 127]
```

- FP16 Conversion for CPU/GPU

```bash
python3 /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py --input_model "<path>/mobilenet-ssd.caffemodel" --output_dir /home/intel/openvino_models/ir/mobilenet-ssd/FP32 --data_type FP16 --scale 256 --mean_values [127, 127, 127]
```

##### Step 4

Initialize Environment Variables

```bash
source /opt/intel/computer_vision_sdk/bin/setupvar.sh
```

##### Step 5

Run the example using one of the scripts below:

- OpenCV

```bash
python3 object_detection_multi_platform.py --framework caffe --device CPU --backend OPENCV_DNN --mconfig <path>/mobilenet-ssd.prototxt --mweight <path>/mobilenet-ssd.caffemodel --mlabels models/caffe_ssd_labels.txt -i offline --source <video_path> --model_image_height 300 --model_image_width 300 --confidence 0.60
```

- CPU
  
```bash
python3 object_detection_multi_platform.py --framework openvino --device CPU --mconfig <path_to_FP32_IR>/mobilenet-ssd.xml --mweight <path_to_FP32_IR>/mobilenet-ssd.bin --mlabels <path_to_labels>/caffe_ssd_labels.txt -i offline --source <path_to_video> --model_image_height 300 --model_image_width 300 --confidence 0.60 --async True
```    
      
```bash
python3 object_detection_multi_platform.py --framework openvino --device CPU --mconfig <path_to_FP32_IR>/mobilenet-ssd.xml --mweight <path_to_FP32_IR>/mobilenet-ssd.bin --mlabels <path_to_labels>/caffe_ssd_labels.txt -i live --source <camera index 0/1...> --model_image_height 300 --model_image_width 300 --confidence 0.60 --async True
```    

- GPU

```bash
python3 object_detection_multi_platform.py --framework openvino --device GPU --mconfig <path_to_FP32_IR>/mobilenet-ssd.xml --mweight <path_to_FP32_IR>/mobilenet-ssd.bin --mlabels <path_to_labels>/caffe_ssd_labels.txt -i offline --source <path_to_video> --model_image_height 300 --model_image_width 300 --confidence 0.60 --async True
```
```bash
python3 object_detection_multi_platform.py --framework openvino --device GPU --mconfig <path_to_FP16_IR>/mobilenet-ssd.xml --mweight <path_to_FP16_IR>/mobilenet-ssd.bin --mlabels <path_to_labels>/caffe_ssd_labels.txt -i offline --source <path_to_video> --model_image_height 300 --model_image_width 300 --confidence 0.60 --async True
```

- MYRIAD

```bash
python3 object_detection_multi_platform.py --framework openvino --device MYRIAD --mconfig <path_to_FP16_IR>/mobilenet-ssd.xml --mweight <path_to_FP16_IR>/mobilenet-ssd.bin --mlabels <path_to_labels>/caffe_ssd_labels.txt -i offline --source <path_to_video> --model_image_height 300 --model_image_width 300 --confidence 0.60 --async True
```

- HETERO

```bash
python3 object_detection_multi_platform.py --framework openvino --device HETERO:MYRIAD,GPU --mconfig <path_to_FP16_IR>/mobilenet-ssd.xml --mweight <path_to_FP16_IR>/mobilenet-ssd.bin --mlabels <path_to_labels>/caffe_ssd_labels.txt -i offline --source <path_to_video> --model_image_height 300 --model_image_width 300 --confidence 0.60 --async True
```

```bash
python3 object_detection_multi_platform.py --framework openvino --device HETERO:GPU,CPU --mconfig <path_to_FP32_IR>/mobilenet-ssd.xml --mweight <path_to_FP32_IR>/mobilenet-ssd.bin --mlabels <path_to_labels>/caffe_ssd_labels.txt -i offline --source <path_to_video> --model_image_height 300 --model_image_width 300 --confidence 0.60 --async True
```

#### How to test multiple MYRIAD

**Note**: Below numeric results would change according to used video and platform. 

**Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz** 

OpenVINO supports multiple NCS (Myriad X) devices, each IEPlugin associates with one of the attached idle NCS respectively.

- First open two terminals 
- Set environment variables 
- Let's run 1 instance of application with GPU support at first.

```bash
python3 object_detection_multi_platform.py --framework openvino --device GPU --mconfig <path_to_FP16_IR>/mobilenet-ssd.xml --mweight <path_to_FP16_IR>/mobilenet-ssd.bin --mlabels <path_to_labels>/caffe_ssd_labels.txt -i offline --source <path_to_video> --model_image_height 300 --model_image_width 300 --confidence 0.60 --async True
```

We get 15ms for inferring a frame

- Then, run two instances with GPU support.

Here we get 19ms for inferring a frame

- Then, run three instances with GPU support.

Here we get 25ms for inferring a frame

- Now, continue running 1 instance with MYRIAD.

```bash
python3 object_detection_multi_platform.py --framework openvino --device GPU --mconfig <path_to_FP16_IR>/mobilenet-ssd.xml --mweight <path_to_FP16_IR>/mobilenet-ssd.bin --mlabels <path_to_labels>/caffe_ssd_labels.txt -i offline --source <path_to_video> --model_image_height 300 --model_image_width 300 --confidence 0.60 --async True
```

- Finally, we can run 2 instances with MYRIAD and see that, inference time didn't change if 2 MYRIAD attached.

**Note**: Each IEPlugin allocates one NCS so, two instances with 1 NCS can't run.

- When we run one instance we get

42ms for inferring a frame

- When we run two instances we get 

43ms for inferring a frame