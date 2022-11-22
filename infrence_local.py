import requests
import json
import numpy as np
#import tensorflow as tf
import time
from PIL import Image

count = 0

url = "http://localhost:8501/v1/models/model:predict"
headers = {"content-type":"application/json"}

class_names = ['one_column', 'two_column']

for count in range(20):
	count += 1
	#path = keras.utils.get_file(origin=url_x)
	#img = keras.preprocessing.image.load_img(str(count) + '.jpg', target_size=(300, 200))
	img = Image.open(str(count) + '.jpg')
	img = img.resize((200,300))
	img = np.expand_dims(img, axis = 0)

	data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})

	start_time = time.time()
	reponse = requests.post(url, data=data, headers=headers)
	prediction = json.loads(reponse.text)["predictions"]

	print(prediction)
	print(class_names[np.argmax(prediction)])
	
url = 'http://localhost:8501/v1/models/model'
print(requests.get(url))

