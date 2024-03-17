import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Data preprocessing
def preprocess_data(data):
    # Feature scaling
    scaled_data = data[['Rev1', 'Rev2', 'Rev3', 'Rev4', 'Rev5']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Data normalization
    normalized_data = (scaled_data - scaled_data.mean()) / scaled_data.std()

    # Convert categorical variables to one-hot encoding
    categorical_data = pd.get_dummies(data['Pris'])

    # Concatenate numerical and categorical features
    processed_data = pd.concat([normalized_data, categorical_data, data[['X', 'Y', 'Z']]], axis=1)

    return processed_data

# Build the neural network model
def build_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dense(1, activation='linear'))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    return model

# Load and preprocess the data
train_data_path = '/Users/tszsanwu/Desktop/Code/Project Mojave/Robot_movement_dataset/traindata1.xlsx'
train_data = pd.read_excel(train_data_path).astype(float)  # Convert to float

test_data_path = '/Users/tszsanwu/Desktop/Code/Project Mojave/Robot_movement_dataset/testdata1.xlsx'
test_data = pd.read_excel(test_data_path).astype(float)  # Convert to float

# Check for NaN values and fill them with valid values
train_data = train_data.fillna(0)  # Fill NaN values with 0 or other valid values

# Convert non-numeric columns to float
train_data[['Rev1', 'Rev2', 'Rev3', 'Rev4', 'Rev5']] = train_data[['Rev1', 'Rev2', 'Rev3', 'Rev4', 'Rev5']].astype(float)

train_processed_data = preprocess_data(train_data)
test_processed_data = preprocess_data(test_data)

train_data_input = train_processed_data.iloc[:, :-1]  # Input features
train_labels = train_processed_data.iloc[:, -1]  # Output labels

test_data_input = test_processed_data.iloc[:, :-1]  # Input features
test_labels = test_processed_data.iloc[:, -1]  # Output labels

train_data_input = train_data_input.astype(float)
train_labels = train_labels.astype(float)
test_data_input = test_data_input.astype(float)
test_labels = test_labels.astype(float)

# Build the neural network model
input_shape = (train_data_input.shape[1],)
model = build_model(input_shape)

# Train the model
model.fit(train_data_input, train_labels, epochs=10, batch_size=32, validation_data=(test_data_input, test_labels))

# Make predictions using the model
predicted_labels = model.predict(test_data_input)

# Normalize the predicted labels
normalized_predicted_labels = (predicted_labels - predicted_labels.min()) / (predicted_labels.max() - predicted_labels.min())

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the predicted labels
ax.plot(range(len(normalized_predicted_labels)), normalized_predicted_labels, lw=2)

# Set the axis limits
ax.set_xlim(0, len(normalized_predicted_labels))
ax.set_ylim(0, 1)

# Set the plot title and labels
ax.set_title("Predicted Labels")
ax.set_xlabel("Frame")
ax.set_ylabel("Normalized Value")

# Save the plot as a PNG image
plt.savefig("predicted_labels.png", dpi=300)
