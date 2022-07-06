from os import listdir
from shutil import copy
from time import time
from keras.models import load_model
from PIL.Image import open
import numpy as np
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
from keras.preprocessing import image

def get_frozen_graph(model_path):
	with tf.gfile.FastGFile(model_path, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		
	return graph_def



input_names = ['input_1']
output_names = ['output_layer/Softmax']


CHOOSE_MODEL = ["VGG19", "MobileNetV2", "Xception"]
MODEL_NAME = CHOOSE_MODEL[0]

if MODEL_NAME == "VGG19":
	from keras.applications.vgg19 import preprocess_input
	#model_path = './Models/VGG19_tensorrt_FP16_100/trt_graph.pb'
	model_path = './Results_Jetson/VGG19/VGG19_tensorrt_FP16/trt_graph.pb'
	target_size = (224,224)
elif MODEL_NAME == "MobileNetV2":
	from keras.applications.mobilenet_v2 import preprocess_input
	model_path = './Results_Jetson/MobileNetV2/MobileNetV2_tensorrt_FP16/trt_graph.pb'
	target_size = (224,224)
	
elif MODEL_NAME == "Xception":
	from keras.applications.xception import preprocess_input
	model_path = './Results_Jetson/Xception/Xception_tensorrt_FP16/trt_graph.pb'
	target_size = (299,299)

trt_graph = get_frozen_graph(model_path)

# Create a session and load the graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')



input_tensor_name = input_names[0]+':0'
output_tensor_name = output_names[0]+':0'

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)

#Corresponding classes 0,1,2
classes = ["Hintergrund", 'KeineVereisung', 'Vereisung']

monitor_folder = './data/'

#for testing
dest = './predictions/'

processed_files = []

print("Starting...")
while True:
	# Check each 100ms if there is a new image
	check_files = listdir(monitor_folder)
	
	new_files = list(set(check_files) - set(processed_files))
	if len(new_files)>0:
		
		# Wait until the image is properly saved
		i = 0
		counter = 0
		for f in new_files:
			img = open(monitor_folder+f)
			img = img.resize(target_size)
			img_prepro = preprocess_input(np.asarray(img))
			img_prepro = img_prepro[np.newaxis, ...]
			
			feed_dict = {
					input_tensor_name: img_prepro
					}
			y_pred = tf_sess.run(output_tensor, feed_dict)
			class_pred = classes[np.argmax(y_pred)]
			copy(monitor_folder+f,dest+class_pred)
			i += 1
			counter += 1
			# warmup
			warmup = 20
			if i==warmup:
				start = time()
				counter -= warmup
			if counter%200==0:
				end = time()-start
				print(f"Processed {counter} files in {end:.2f} sec")
			
		processed_files = check_files
		
end = time()-start
print(f"Processed {len(processed_files)} files in {end:2f} sec")
