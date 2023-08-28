# ONNX Model Conversion and Inference

This subfolder provides instructions and relevant details about converting a given tabular data model along with its preprocessing into the ONNX (Open Neural Network Exchange) format. Furthermore, we will also showcase the steps to perform inference using the converted ONNX model.

## Table of Contents

1. [Pre-requisites](#pre-requisites)
2. [Converting Preprocessing to ONNX](#converting-preprocessing-to-onnx)
3. [Converting Model to ONNX](#converting-model-to-onnx)
4. [Inference with ONNX Runtime](#inference-with-onnx-runtime)
5. [Performance Metrics](#performance-metrics)

## Pre-requisites

1. Ensure you have the ONNX library installed:

   ```
   pip install onnx
   ```

2. Install the ONNX runtime:

   ```
   pip install onnxruntime
   ```

## Converting Preprocessing to ONNX

Before converting the model itself, the preprocessing steps (like normalization, encoding, etc.) need to be converted to the ONNX format.

1. Define your preprocessing steps using sklearn or any other supported framework.
2. Convert the preprocessing pipeline using an appropriate conversion utility, for instance, `skl2onnx`.

```python
import onnxmltools
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Define the input shape for the scaler
initial_type = [('float_input', FloatTensorType([None, train_x.shape[1]]))]

# Convert the sklearn scaler to ONNX
onnx_scaler = convert_sklearn(scaler, initial_types=initial_type)
onnxmltools.utils.save_model(onnx_scaler, 'scaler.onnx')
```

## Converting Model to ONNX

1. Train your model using any preferred framework that supports ONNX conversion (e.g., PyTorch, TensorFlow, sklearn).
2. Convert the trained model to ONNX format.

For instance, using PyTorch:

```python
import torch.onnx

torch.onnx.export(
    model,               # model being run
    dummy_input,                  # model input (or a tuple for multiple inputs)
    "model.onnx",       # where to save the model (can be a file or file-like object)
    export_params=True, # store the trained parameter weights inside the model file
    opset_version=11,   # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names = ['input'],   # the model's input names
    output_names = ['output'], # the model's output names
    dynamic_axes={
        'input' : {0 : 'batch_size'},  # variable length axes
        'output' : {0 : 'batch_size'}
    }
)
```

## Inference with ONNX Runtime

To run the model with ONNX Runtime:

```python
import onnxruntime as rt

# Load the model
sess = rt.InferenceSession("model.onnx")

# Assuming input_data is your data for inference
input_name = sess.get_inputs()[0].name
predicted_label = sess.run(None, {input_name: input_data})
```

## Performance Metrics

Here, we list down the results of the runtime in two distinct cases:

1. **Batched Inference:**
    - Torch: 0.00018 seconds, Stdev: 3e-05 seconds
    - ONNX: Mean Latency: 8.6e-05 seconds, Stdev: 2e-05 seconds

2. **Single Inference:**
    - Torch: Mean Latency: 0.00018 seconds, Stdev: 2e-05 seconds
    - ONNX: Mean Latency: 7.2e-05 seconds, Stdev: 4.5e-05 seconds

(Note: Replace X, Y, A, and B with the actual metrics from your experimentation.)

---

For further details or any troubleshooting, feel free to refer to the official [ONNX documentation](https://onnx.ai/documentation/).
