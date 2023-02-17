# onnx2mlir
System: Ubuntu 18.04
ONNX-OpenVino Intermediate Format-TF-Lite-Float32-MLIR
After convert to TF-Lite-Float32. Run:
```shell
WORKDIR="./tfl2mlir"
TFLITE_PATH='openvino2tensorflow/model_float32.tflite'

IMPORT_PATH=${WORKDIR}/tosa.mlir
MODULE_PATH=${WORKDIR}/module.vmfb

# Import the sample model to an IREE compatible form
iree-import-tflite --output-format=mlir-ir ${TFLITE_PATH} -o ${IMPORT_PATH}

```
