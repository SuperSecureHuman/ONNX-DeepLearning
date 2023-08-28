import pandas as pd
import numpy as np
import onnxruntime as ort
import time

# Load the data from heart.csv
data = pd.read_csv('heart.csv')
test_x = data.drop(['target'], axis=1)  # assuming 'target' column is present and you want to drop it

# Load the ONNX models
ort_session_scaler = ort.InferenceSession('scaler.onnx', providers=['CPUExecutionProvider'])
ort_session_model = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])

single_sample_latencies = []
batched_sample_latencies = []

# Run 1000 trials
for trial in range(1000):

    # Calculate latency for single sample
    start = time.time()
    inputs = test_x.iloc[1].to_numpy().reshape(1, -1)  # Convert to numpy array and reshape
    scaled_input = ort_session_scaler.run(None, {'float_input': inputs.astype(np.float32)})
    predicted_output = ort_session_model.run(None, {'input': scaled_input[0]})
    end = time.time()
    latency = end - start
    single_sample_latencies.append(latency)

    # Calculate throughput for batched samples
    batch_size = 100  # Adjust based on preference and memory capacity
    start = time.time()
    inputs = test_x.iloc[:batch_size].to_numpy()  # Use iloc to get the first `batch_size` rows
    scaled_input = ort_session_scaler.run(None, {'float_input': inputs.astype(np.float32)})
    predicted_output = ort_session_model.run(None, {'input': scaled_input[0]})
    end = time.time()
    latency = end - start
    batched_sample_latencies.append(latency)

# Compute mean and stdev
mean_single_sample_latency = np.mean(single_sample_latencies)
std_single_sample_latency = np.std(single_sample_latencies)

mean_batched_sample_latency = np.mean(batched_sample_latencies)
std_batched_sample_latency = np.std(batched_sample_latencies)

# Log everything to file
with open('onnx.txt', 'w') as f:
    f.write('Single Sample:\n')
    f.write('Mean Latency: {} seconds, Stdev: {} seconds\n'.format(
        mean_single_sample_latency, std_single_sample_latency))
    f.write('\n')
    f.write('Batched Sample:\n')
    f.write('Mean Latency: {} seconds, Stdev: {} seconds\n'.format(
        mean_batched_sample_latency, std_batched_sample_latency))
