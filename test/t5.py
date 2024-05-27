import numpy as np

x = np.array(([1, 4], [2, 9], [2, 7]), dtype=float)
y = np.array(([95], [87], [81]), dtype=float)
x = x / np.amax(x, axis=0)
y = y / 100

epoch = 990
lr = 0.2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def der_sigmoid(x):
    return x * (1 - x)


inp_layer = 2
hid_layer = 3
out_layer = 1

wh = np.random.uniform(size=(inp_layer, hid_layer))
wout = np.random.uniform(size=(hid_layer, out_layer))
bh = np.random.uniform(size=(1, hid_layer))
bout = np.random.uniform(size=(1, out_layer))


for i in range(epoch):
    hinp1 = np.dot(x, wh)
    hinp = hinp1 + bh
    hact_layer = sigmoid(hinp)

    outinp1 = np.dot(hact_layer, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)

    EO = y - output
    outgrad = der_sigmoid(output)
    d_output = EO * outgrad

    EH = d_output.dot(wout.T)
    hidgrad = der_sigmoid(hact_layer)
    d_hidden = EH * hidgrad

    wout += hact_layer.T.dot(d_output) * lr
    wh += x.T.dot(d_hidden) * lr

print("X: ", str(x))
print("Y: ", str(y))
print("output: ", output)
