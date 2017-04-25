
from mnist_help import square_like, model_from_file
import matplotlib.pyplot as plt

loaded_model = model_from_file('model5x5.json', 'trained_model5x5.h5')

layer_dict = dict([(layer.name, layer) for layer in loaded_model.layers])

conv_1_kernel = layer_dict['conv2d_1'].weights[0].get_value()
print('Read first conv layer with shape: %s' % str(conv_1_kernel.shape))

rows, cols = square_like(conv_1_kernel.shape[3])

for idx in range(conv_1_kernel.shape[3]):
    plt.subplot(rows,cols,idx+1)
    plt.imshow(conv_1_kernel[:,:,0,idx],cmap='Greys')

plt.show()