
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from mnist_help import *

np.set_printoptions(formatter={'float_kind':lambda x: "%.2f" % x})

img_width = 28
img_height = 28
num_channels = 1

loaded_model = model_from_file('model.json', 'trained_model.h5')
loaded_model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

input_img = loaded_model.input

layer_dict = dict([(layer.name, layer) for layer in loaded_model.layers])

num_outputs = 10

rows, cols = square_like(num_outputs)

output_idx = 0
while output_idx < num_outputs:

    print('optimizing filter %d' % output_idx)

    plt.subplot(rows,cols,output_idx+1)

    loss = K.mean(loaded_model.output[:, output_idx])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]
    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img, K.learning_phase()], [loss, grads])

    # we start from a gray image with some noise
    input_img_data = (np.random.random((1, num_channels, img_width, img_height)) * 10.0 + 5.) / 255.

    failed = False
    # run gradient ascent
    for i in range(40000):

        processed = process(input_img_data)

        predictions = loaded_model.predict(processed)

        loss_value, grads_value = iterate([processed, 0])

        grads_value *= dprocessed(input_img_data[0])

        if i%2000 == 0:
            print('Iteration %d/%d, loss: %f' % (i, 40000, loss_value))
            print('Mean grad: %f' % np.mean(grads_value))
            if all(np.abs(grads_val) < 0.000001 for grads_val in grads_value.flatten()):
                failed = True
                print('Failed')
                break
            # print('Image:\n%s' % str(input_img_data[0,0,:,:]))
            if loss_value > 0.9999:
                break

        input_img_data += grads_value * 1e-5

    if failed:
        continue

    output_idx += 1
    plt.imshow((process(input_img_data[0,0,:,:])*255).astype('uint8'), cmap='Greys')

plt.show()