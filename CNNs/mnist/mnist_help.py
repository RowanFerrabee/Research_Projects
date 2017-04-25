
import numpy as np
from keras.models import model_from_json

def square_like(n):
	assert n > 0
	rows = int(np.sqrt(n))
	while (n % rows != 0):
		rows -= 1
	return rows, n // rows

def model_from_file(json_file_name, weights_file_name):
	json_file = open(json_file_name,'r')
	loaded_model_json = json_file.read()
	json_file.close()

	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(weights_file_name)
	return loaded_model

def process(x):
	res = np.clip(x, 0, 1)
	return res

def dprocessed(x):
	res = np.zeros_like(x)
	res += 1
	res[x < 0] = 0
	res[x > 1] = 0
	return res

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def deprocess_image_binary(x):
	img = np.zeros_like(x)
	mean = x.mean()
	std = x.std()
	img[x < mean - std] = 255
	return img