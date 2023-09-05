# Ultra-Fast-Lane-Detection-V2

This README shows how to perform hardware-aware optimization of Ultra-Fast-Lane-Detection-V2 ResNet-18 model on
[TuSimple](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection) dataset.

## Setup the environment

The repository code was tested on Python 3.8.

To get started, install `torch==1.13.1` and `torchvision==0.14.1` compatible with your CUDA using
[the instruction](https://pytorch.org/get-started/previous-versions/#v1131) from the official site.

The repository is based on two main packages:

- [ENOT Framework](https://enot-autodl.rtd.enot.ai/en/v3.3.2/) — a flexible tool for Deep Learning developers which automates neural architecture optimization.
- [ENOT Latency Server](https://enot-autodl.rtd.enot.ai/en/latest/latency_server.html) — small open-source package that provides simple API for latency measurement on remote device.

Follow [the installation guide](https://enot-autodl.rtd.enot.ai/en/v3.3.2/installation_guide.html) to install `enot-autodl==3.3.2`.

To install `enot-latency-server` simply run:

```bash
pip install enot-latency-server==1.2.0
```

Install other requirements:

> **_NOTE:_** You must have the same CUDA version on your system as PyTorch's CUDA version.
> We built `my_interp` using CUDA 11.7.

```bash
pip install -r requirements.txt
# Install NVIDIA DALI - very fast data loading library:
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

cd my_interp
# If the following command fails, you might need to add path to your cuda to PATH:
# PATH=/usr/local/cuda-11/bin:$PATH bash build.sh
bash build.sh
```

> **_NOTE:_** All pruning/training procedures are performed on x86-64 computer,
> ONLY latency measurements are performed on a remote target device,
> so you do not need to install `enot-autodl` package on the target device,
> you only need to install `enot-latency-server` package for latency measurements.

## Prepare dataset

Download preprocessed TuSimple dataset from [Google Drive](https://drive.google.com/file/d/16Uk7_uRtue9OLaQCuMEfEbS5rhSxUHC1/view?usp=sharing) and unzip it to the repository root:

```bash
unzip dataset.zip
```

The dataset should have the following structure:

```text
└── ultra-fast-lane-detector-v2 (repository root)
    └──dataset
        ├── clips
            ├── 0313-1
            ├── 0313-2
            ├── 0530
            ├── 0531
            └── 0601
        ├── label_data_0313.json
        ├── label_data_0531.json
        ├── label_data_0601.json
        ├── test_label.json
        ├── test_tasks_0627.json
        ├── test.txt
        ├── train_gt.txt
        └── tusimple_anno_cache.json
```

If you want to use your own path for dataset, change `data_root` parameter in `configs/tusimple_res18.py`.

To train baseline model, run:

```bash
bash commands/baseline/train.sh
```

The result of this command is the `model_best.pth` checkpoint in the `runs/baseline` directory.

Use this command to verify baseline accuracy:

```bash
bash commands/baseline/test.sh
```

## Model optimization (Jetson)

To optimize a model by latency for Jetson, run our latency server on Jetson (see [instruction](https://github.com/ENOT-AutoDL/latency-server-nvidia-jetson-agx-orin-devkit)).

> **_NOTE:_** Substitute `--host` and `--port` in the commands and `.sh` scripts below with the host and port of your server on Jetson.

### Pruning

To optimize a model by latency for Jetson, run the corresponding script (x2/x3 means latency acceleration):

```bash
bash commands/x2_jetson/prune.sh
bash commands/x3_jetson/prune.sh
```

### Model tuning

After pruning, the model should be tuned with the following command:

```bash
bash commands/x2_jetson/tune.sh
bash commands/x3_jetson/tune.sh
```

### Quantization

To use INT8 data type for model inference, follow our quantization pipeline:

```bash
bash commands/x3_jetson/quant.sh
```

### Accuracy and latency verification

Use this command to verify the optimized model accuracy:

```bash
bash commands/x2_jetson/test.sh
bash commands/x3_jetson/test.sh
```

Use this command to verify the optimized model latency:

```bash
bash commands/x2_jetson/measure.sh
bash commands/x3_jetson/measure.sh
```

### Our optimization results

Download our checkpoints from [Google Drive](https://drive.google.com/file/d/1uDzWVkCwWnY5XZ8CH80b0vGVxRSmDU9G/view?usp=sharing).

To extract `checkpoints` use the following command:

```bash
unzip ufld_ckpt_with_onnx.zip
```

To check their metrics, run with the following commands:

```bash
python test.py configs/tusimple_res18.py --model_ckpt checkpoints/baseline/model_best.pth
python test.py configs/tusimple_res18.py --model_ckpt checkpoints/x2_jetson/model_best.pth
python test.py configs/tusimple_res18.py --model_ckpt checkpoints/x3_jetson/model_best.pth
```

To check metrics on ONNX run:

```bash
python test.py configs/tusimple_res18.py --onnx_path checkpoints/baseline/model_best.onnx --batch_size 1
python test.py configs/tusimple_res18.py --onnx_path checkpoints/x2_jetson/model_best.onnx --batch_size 1
python test.py configs/tusimple_res18.py --onnx_path checkpoints/x3_jetson/model_best.onnx --batch_size 1
```

> **_NOTE:_** We recommend to check metric for `quantized_model.onnx` on a target device (see our instruction in [Validation on Jetson AGX Orin device](#validation-on-jetson-agx-orin-device))

To check their latency, run the following commands:

```bash
python measure.py --model_ckpt checkpoints/baseline/model_best.pth --host <jetson-server-host> --port 15003
python measure.py --model_ckpt checkpoints/x2_jetson/model_best.pth --host <jetson-server-host> --port 15003
python measure.py --model_ckpt checkpoints/x3_jetson/model_best.pth --host <jetson-server-host> --port 15003
python measure.py --onnx checkpoints/x3_jetson/quantized_model.onnx --host <jetson-server-host> --port 15003
```

### Validation on Jetson AGX Orin device

To make sure that your model accuracy is not affected by computations in FP16 or INT8 on Jetson device, follow this validation pipeline:

⚠️ On PC where you run scripts from this repository:

1. Create a dataset in the pickle format:

   ```bash
   python pickle_dataset.py configs/tusimple_res18.py --pickle_data_path pickle_data
   ```

1. Send an ONNX model, `pickle_data`, and `inference_on_device.py` to the Jetson device using `scp`:

   ```bash
   scp -P <jetson-port> -r path/to/model.onnx pickle_data inference_on_device.py <user-name>@<jetson-host>:/your/location/
   ```

⚠️ On Jetson device:

1. Install OnnxRuntime package with TensorRT backend using the following commands:

   ```bash
   wget https://nvidia.box.com/shared/static/mvdcltm9ewdy2d5nurkiqorofz1s53ww.whl -O onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl
   pip3 install onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl
   ```

1. Run inference on the pickled dataset from the directory with previously copied model ONNX, `pickle_data`, and `inference_on_device.py`:

   ```bash
   python3 inference_on_device.py -m your/model.onnx -i pickle_data -o out_pickle --device jetson
   ```

1. Send the resulting `out_pickle` directory to your PC to the `ultra-fast-lane-detector-v2` repository root using `scp`:

   ```bash
   scp -P <pc-port> -r out_pickle <user-name>@<pc-host>:/path/to/ultra-fast-lane-detector-v2/
   ```

⚠️ On PC where you run scripts from this repository:

```bash
python test_on_pickles.py configs/tusimple_res18.py --batch_size 1 --pickled_inference_results out_pickle
```

## Model optimization (TI)

To optimize a model by latency for Texas Instruments (TI), you need to run a latency server on TI and a compile server on x86 PC (Linux OS).
The compile server creates binaries for a model and sends them to the latency server.
The latency server measures model latency using these binaries.
Use our [instruction](https://github.com/ENOT-AutoDL/latency-server-ti-tda4-j721exskg01evm) to run latency server and compile server.

> **_NOTE:_** Substitute `--host` and `--port` in the commands and `.sh` scripts below with the host and port of your compile server on x86 PC.

### Pruning

To optimize a model by latency for TI, run the corresponding script (x4 means latency acceleration):

```bash
bash commands/x4_ti/prune.sh
```

### Model tuning

After pruning, the model should be tuned with the following command:

```bash
bash commands/x4_ti/tune.sh
```

### Accuracy and latency verification

Use this command to verify the optimized model accuracy:

```bash
bash commands/x4_ti/test.sh
```

Use this command to verify the optimized model latency:

```bash
bash commands/x4_ti/measure.sh
```

### Our optimization results

Download our checkpoints from [Google Drive](https://drive.google.com/file/d/1uDzWVkCwWnY5XZ8CH80b0vGVxRSmDU9G/view?usp=sharing).

To extract `checkpoints` use the following command:

```bash
unzip ufld_ckpt_with_onnx.zip
```

To check their metrics, run with the following commands:

```bash
python test.py configs/tusimple_res18.py --model_ckpt checkpoints/baseline/model_best.pth
python test.py configs/tusimple_res18.py --model_ckpt checkpoints/x3_ti/model_best.pth
python test.py configs/tusimple_res18.py --model_ckpt checkpoints/x4_ti/model_best.pth
```

> **_NOTE:_** Model `checkpoints/x3_ti/model_best.pth` was obtained on Jetson (`checkpoints/x2_jetson/model_best.pth`) and has x3 acceleration on TI device.

To check metrics on ONNX run:

```bash
python test.py configs/tusimple_res18.py --onnx_path checkpoints/baseline/model_best.onnx --batch_size 1
python test.py configs/tusimple_res18.py --onnx_path checkpoints/x3_ti/model_best.onnx --batch_size 1
python test.py configs/tusimple_res18.py --onnx_path checkpoints/x4_ti/model_best.onnx --batch_size 1
```

To check their latency on TI, run the following commands:

```bash
python measure.py --model_ckpt checkpoints/baseline/model_best.pth --host <compile-server-host> --port 15003 --ti_server
python measure.py --model_ckpt checkpoints/x3_ti/model_best.pth --host <compile-server-host> --port 15003 --ti_server
python measure.py --model_ckpt checkpoints/x4_ti/model_best.pth --host <compile-server-host> --port 15003 --ti_server
```

### Validation on TI device

TI NPU performs computations in FX8 data type (8-bit fixed point numbers).
To make sure that your model accuracy is not affected by computations in FX8, follow this validation pipeline:

⚠️ On PC where you run scripts from this repository:

1. Create a dataset in the pickle format:

   ```bash
   python pickle_dataset.py configs/tusimple_res18.py --pickle_data_path pickle_data
   ```

1. Download calibration data from [Google Drive](https://drive.google.com/file/d/1M_aKQvDQ3NRnI7Sz5-8F3_EL7Q0cizT1/view?usp=drive_link) to the repository root.

1. Create model artifacts for TI NPU using these calibration data:

   ```bash
   python compile_model.py -m <your-checkpoint.pth> -c ufldv2_calibration.zip -o compiled_artifacts --host <compilation-server-host> --port <compilation-server-port>
   ```

   > **_NOTE:_** Make sure that your compilation server is up-to-date with the last version from the [repository](https://github.com/ENOT-AutoDL/latency-server-ti-tda4-j721exskg01evm).

   > **_NOTE:_** It takes more about 60 min to calibrate and compile the baseline model on our x86_PC.

1. Send `compiled_artifacts`, `pickle_data`, and `inference_on_device.py` to the TI device using `scp`:

   ```bash
   scp -P <ti-port> -r compiled_artifacts pickle_data inference_on_device.py <user-name>@<ti-host>:/your/location/
   ```

⚠️ On TI device:

1. Run inference on the pickled dataset from the directory with previously copied `compiled_artifacts`, `pickle_data`, and `inference_on_device.py`:

   ```bash
   TIDL_TOOLS_PATH=/opt/latency_server/tidl_tools python3 inference_on_device.py -m compiled_artifacts -i pickle_data -o out_pickle --device ti
   ```

1. Send the resulting `out_pickle` directory to your PC to the `ultra-fast-lane-detector-v2` repository root using `scp`:

   ```bash
   scp -P <pc-port> -r out_pickle <user-name>@<pc-host>:/path/to/ultra-fast-lane-detector-v2/
   ```

⚠️ On PC where you run scripts from this repository:

```bash
python test_on_pickles.py configs/tusimple_res18.py --batch_size 1 --pickled_inference_results out_pickle
```
