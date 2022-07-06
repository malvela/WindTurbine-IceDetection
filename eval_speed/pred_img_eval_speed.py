from os import listdir
from shutil import copy
from time import time
from keras.models import load_model
from PIL.Image import open
import numpy as np


CHOOSE_MODEL = ["VGG19", "MobileNetV2", "Xception"]
MODEL_NAME = CHOOSE_MODEL[2]

if MODEL_NAME == "VGG19":
	from keras.applications.vgg19 import preprocess_input
	model = load_model('./Results_Jetson/VGG19/VGG19.h5')
	target_size = (224,224)
elif MODEL_NAME == "MobileNetV2":
	from keras.applications.mobilenet_v2 import preprocess_input
	model = load_model('./Results_Jetson/MobileNetV2/MobileNetV2.h5')
	target_size = (224,224)
	
elif MODEL_NAME == "Xception":
	from keras.applications.xception import preprocess_input
	model = load_model('./Results_Jetson/Xception/Xception.h5')
	target_size = (299,299)
	
#Corresponding classes 0,1,2
classes = ["Hintergrund", 'KeineVereisung', 'Vereisung']


monitor_folder = './data/'

#for testing
dest = './predictions/'

processed_files = []


print("Starting...")
while True:
	# Check each 100ms if there is a new image
	#sleep(0.01)
	check_files = listdir(monitor_folder)
	
	new_files = list(set(check_files) - set(processed_files))
	if len(new_files)>0:
		
		# Wait until the image is properly saved
		#sleep(0.05)
		i = 0
		counter = 0
		for f in new_files:
			img = open(monitor_folder+f)
			img = img.resize(target_size)
			img_prepro = preprocess_input(np.asarray(img))
			img_prepro = img_prepro[np.newaxis, ...]
			y_pred = model.predict(img_prepro)
			class_pred = classes[np.argmax(y_pred)]
			copy(monitor_folder+f,dest+class_pred)
			i +=1
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
