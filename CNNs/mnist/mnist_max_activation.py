
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from mnist_help import model_from_file, deprocess_image, square_like
import theano
import theano.tensor as T

img_width = 28
img_height = 28
num_channels = 1

loaded_model = model_from_file('model5x5.json', 'trained_model5x5.h5')
input_img = loaded_model.input

layer_dict = dict([(layer.name, layer) for layer in loaded_model.layers])

layer_name = 'conv2d_2'

# build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_output = layer_dict[layer_name].output
num_filters = layer_dict[layer_name].weights[0].get_value().shape[3]

rows, cols = square_like(num_filters)

for filter_index in range(num_filters):

    print('optimizing filter %d' % filter_index)

    plt.subplot(rows,cols,filter_index+1)

    loss = K.mean(layer_output[:, :, :, filter_index])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]
    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we start from a gray image with some noise
    input_img_data = np.random.random((1, num_channels, img_width, img_height)) * 20 + 128.

    # run gradient ascent
    for i in range(5000):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * 1e-5

    plt.imshow(deprocess_image(input_img_data[0,0,:,:]), cmap='Greys')

plt.show()