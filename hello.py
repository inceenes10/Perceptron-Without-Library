import numpy as np


# isSmoke | isObesity | isDoExercise
# We predict whether the person is diabet or not in the future

feature_set = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]]) # 5 x 3
labels = np.array([[1, 0, 0, 1, 1]]).reshape(5, 1) # 1 x 5


weights = np.random.randn(3, 1) # weights
bias = np.random.randn(1) # bias


learning_rate = 0.5

def sigmoid(x):
    return (1 / (np.exp(-x) + 1))

def sigmoid_der(x):
    return (sigmoid(x) * (1 - sigmoid(x)))

for epoch in range(20000):
    input_set = feature_set.T

    # Feed forward

    predicted = sigmoid(np.dot(feature_set, weights) + bias)

    error = predicted - labels
    print(error.sum())

    # Backpropagation
    dcost_dpred = error
    dpred_dz = sigmoid_der(predicted)

    z_delta = dcost_dpred * dpred_dz

    weights -= learning_rate * np.dot(input_set, z_delta)
    for num in z_delta:
        bias -= learning_rate * num


# Example

example_data = np.array([0, 1, 1])

predicted_output = sigmoid(np.dot(example_data, weights) + bias)

print(predicted_output)
