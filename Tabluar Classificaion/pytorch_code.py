import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# Read the data
data = pd.read_csv('heart.csv')

# Split the data into train and test
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Separate the target and features
train_y = train['target']
test_y = test['target']
train_x = train.drop(['target'], axis=1)
test_x = test.drop(['target'], axis=1)

# Scale the data
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Convert the data into tensors
train_x = torch.from_numpy(train_x.astype(np.float32))
train_y = torch.from_numpy(train_y.values.astype(
    np.float32))  # Added .values to convert to numpy array
test_x = torch.from_numpy(test_x.astype(np.float32))
test_y = torch.from_numpy(test_y.values.astype(
    np.float32))  # Added .values to convert to numpy array

# Reshape the data
train_y = train_y.view(train_y.shape[0], 1)
test_y = test_y.view(test_y.shape[0], 1)


# Create the model
class Model(nn.Module):

    def __init__(self, input_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_features, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


# Initialize the model
model = Model(train_x.shape[1])

# Define the loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model with printing progress every 10 epochs
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(train_x)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {}, loss {}'.format(epoch, loss.item()))
        # Evaluate the model
        with torch.no_grad():
            outputs = model(test_x)
            predicted = (outputs > 0.5).float()
            acc = accuracy_score(test_y, predicted)
            print('Accuracy: {}'.format(acc))

# Prepare a list to record latencies and accuracies for single samples and batched samples
single_sample_latencies = []
batched_sample_latencies = []

test = test
test_y = test['target']
test_x = test.drop(['target'], axis=1)

# Run 1000 trials
for trial in range(1000):

    # Calculate latency for single sample
    start = time.time()
    with torch.no_grad():
        #inputs = test_x[0]
        inputs = test_x.iloc[1].to_numpy().reshape(
            1, -1)  # Convert to numpy array and reshape

        # Scale the input
        inputs = scaler.transform(inputs)
        inputs = torch.from_numpy(inputs.astype(np.float32))
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
    end = time.time()
    latency = end - start
    single_sample_latencies.append(latency)

    # Calculate throughput for batched samples
    batch_size = 100  # Adjust based on preference and memory capacity
    start = time.time()
    with torch.no_grad():
        #inputs = test_x[:batch_size]
        inputs = test_x.iloc[:batch_size].to_numpy(
        )  # Use iloc to get the first `batch_size` rows
        inputs = scaler.transform(inputs)
        inputs = torch.from_numpy(inputs.astype(np.float32))
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()

    end = time.time()
    latency = end - start
    batched_sample_latencies.append(latency)


# Compute mean and stdev
mean_single_sample_latency = np.mean(single_sample_latencies)
std_single_sample_latency = np.std(single_sample_latencies)

mean_batched_sample_latency = np.mean(batched_sample_latencies)
std_batched_sample_latency = np.std(batched_sample_latencies)

# Log everything to file
with open('pytorch.txt', 'w') as f:
    f.write('Single Sample:\n')
    f.write('Mean Latency: {} seconds, Stdev: {} seconds\n'.format(
        mean_single_sample_latency, std_single_sample_latency))

    f.write('\n')
    f.write('Batched Sample:\n')
    f.write('Mean Latency: {} seconds, Stdev: {} seconds\n'.format(
        mean_batched_sample_latency, std_batched_sample_latency))

## Save the preprocessing with ONNXml Tools
import onnxmltools
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Define the input shape for the scaler
initial_type = [('float_input', FloatTensorType([None, train_x.shape[1]]))]

# Convert the sklearn scaler to ONNX
onnx_scaler = convert_sklearn(scaler, initial_types=initial_type)
onnxmltools.utils.save_model(onnx_scaler, 'scaler.onnx')

## Save model to ONNX
# Create a dummy input that matches the shape and type of your model's input
dummy_input = torch.randn(1, train_x.shape[1])

# Export the PyTorch model to ONNX
# Export the model
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
