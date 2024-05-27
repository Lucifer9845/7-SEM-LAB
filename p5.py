# 5.)Build an Artificial Neural Network by implementing the Backpropagation algorithm and test the same using appropriate data sets.

import numpy as np


x = np.array(([2, 9], [3, 4], [1, 6]), dtype=float)
y = np.array(([92], [86], [73]), dtype=float)
x = x / np.amax(x, axis=0)  # normalize
y = y / 100  # normalize


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivatives_sigmoid(x):
    return x * (1 - x)


epoch = 90  # total
lr = 0.2
inputlayer_neurons = 2
hiddenlayer_neurons = 3
output_neurons = 1

# randomise weights, bias
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

for i in range(epoch):
    # 1 layer output
    hinp1 = np.dot(x, wh)
    hinp = hinp1 + bh
    hlayer_act = sigmoid(hinp)

    # 2 layer output
    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)

    # 3 output error
    EO = y - output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad

    # 4 hidden error
    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hidden = EH * hiddengrad

    # 5 readjust weights
    wout += hlayer_act.T.dot(d_output) * lr
    wh += x.T.dot(d_hidden) * lr

print("Input :\n" + str(x))
print("Actual output :\n" + str(y))
print("Predicted output :\n", output)

# OUTPUT :
# Input :
# [[0.66666667 1.        ]
#  [0.33333333 0.55555556]
#  [1.         0.66666667]]
# Actual output :
# [[0.92]
#  [0.86]
#  [0.89]]
# Predicted output :
#  [[0.89611258]
#  [0.87930992]
#  [0.89433858]]
