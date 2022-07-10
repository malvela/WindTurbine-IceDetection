"""
Created on Tue Jun 12 2022

@author: GEL, ALV


License
------------------------------

Copyright 2022 University of Bremen

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

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
