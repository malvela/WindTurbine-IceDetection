"""
Created on Tue Mar 22 2022

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

import pathlib
import os
import time

import keras
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.xception import Xception

from keras.models import Model
from keras.layers import Dense
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import LearningRateScheduler, CSVLogger, EarlyStopping

from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
np.random.seed(42)

# Matplotlib settings
import matplotlib
#Backend that does not display to user
matplotlib.use('Agg')
import matplotlib.pyplot as plt
font = {'size':16}
matplotlib.rc('font', **font)



#Function definition
def scheduler(epoch, lr):
	""" Reduce the learning rate each second epoch by 0.06 percent
	"""
	if epoch%3==0:
		return lr*0.94
	else:
		return lr

def plot_confm(ax, confm, classes, title="", colorbar=True):
	im = ax.imshow(confm,
			interpolation='nearest',
			cmap=plt.cm.Blues,
			)
	ax.figure.colorbar(im, ax=ax)
	
	ax.set(xticks=np.arange(confm.shape[1]),
		yticks=np.arange(confm.shape[0]),
		xticklabels=classes,
		yticklabels=classes,
		ylabel='True label',
		xlabel='Predicted label',
		title=title
		)
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		rotation_mode="anchor")
	ax = put_text(confm, ax)
	
	return ax

def put_text(confm, ax):
	for i in range(confm.shape[0]):
		for j in range(confm.shape[1]):
			ax.text(j, i, format(confm[i, j], '.2f'),
				ha="center", va="center", fontsize=24,
				color="orange"
				)
	return ax
	
def run_repititions(root_path, random_state=42, save_models=False):
	# Unit test for setup
	SAVE_PATH = './Results/'+data_path[-12:]
	CHOOSE_MODEL = ['VGG19', 'Xception', 'MobilenetV2']
	MODEL_NAME = CHOOSE_MODEL[2]
	TEST_SETUP = False
	
	# Create save folder
	pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
	if not os.path.exists(SAVE_PATH):
		os.makedirs(SAVE_PATH)
	
	# Run test
	if TEST_SETUP:
		DATA_PATH  = "../../../data/test_setup_anatoli/" + data_path[-12:]
		TRAIN_PATH = DATA_PATH + "train/"
		TEST_PATH  = DATA_PATH + "test/"
		VAL_PATH   = DATA_PATH + "test/"
		VEBOSE = 0
		BATCH_SIZE = 1
		epoch = 1
		epoch_fine = 1
		
    # Define training parameters
	else:
		DATA_PATH  = root_path
		TRAIN_PATH = DATA_PATH + "train/"
		TEST_PATH  = DATA_PATH + "test/"
		VAL_PATH   = DATA_PATH + "valid/"
		VEBOSE = 1
		BATCH_SIZE = 16
		epoch = 10
		epoch_fine = 5
	
	# Get the corresponding Model and choose the correct preprocessing function
	if MODEL_NAME == CHOOSE_MODEL[0]:
		base_model = VGG19(include_top=False, weights='imagenet', pooling='avg')
		from keras.applications.vgg19 import preprocess_input
		image_size=224
		
	elif MODEL_NAME == CHOOSE_MODEL[1]:
		base_model =Xception(weights='imagenet',include_top=False, pooling='avg')
		from keras.applications.xception import preprocess_input
		image_size=299
		
	elif MODEL_NAME == CHOOSE_MODEL[2]:
		base_model = MobileNetV2(weights='imagenet',include_top=False, pooling='avg')
		from keras.applications.mobilenet_v2 import preprocess_input
		image_size=224
	
	# Create image data generator
	train_gen = ImageDataGenerator(
					width_shift_range=0.1,
					height_shift_range=0.1,
					preprocessing_function=preprocess_input                                  
					)
	test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
	
	train_set = train_gen.flow_from_directory(
			directory=TRAIN_PATH,
			target_size=(image_size, image_size),
			color_mode="rgb",
			batch_size=BATCH_SIZE,
			class_mode="categorical",
			shuffle=True,
			seed=random_state)
			
	val_set = test_gen.flow_from_directory(
			directory=VAL_PATH,
			target_size=(image_size, image_size),
			color_mode="rgb",
			batch_size=BATCH_SIZE,
			class_mode="categorical",
			shuffle=True,
			seed=random_state
			)
    	
	val_set_2 = test_gen.flow_from_directory(
			directory=VAL_PATH,
			target_size=(image_size, image_size),
			color_mode="rgb",
			batch_size=1,
			class_mode=None,
			shuffle=False,
			seed=random_state
			)
			
	test_set = test_gen.flow_from_directory(
			directory=TEST_PATH,
			target_size=(image_size, image_size),
			color_mode="rgb",
			batch_size=1,
			class_mode=None,
			shuffle=False,
			seed=random_state
			)
	#Get the classes
	classes = (train_set.class_indices)
	
	# Construct final model
	last_layer = base_model.output
	out = Dense(3, activation='softmax', name='output_layer')(last_layer)
	model = Model(inputs=base_model.input, outputs=out)
	
	# Print model infos
	for layer in model.layers:
		print(layer.name ," = ", layer.trainable)
		
	print(model.summary())
	
	# Start training
	start = time.time()
	STEP_SIZE_TRAIN=train_set.n//train_set.batch_size
	STEP_SIZE_VALID=test_set.n//test_set.batch_size
	
	model.compile(
				optimizer=SGD(lr=0.0015, momentum=0.9),
				loss='categorical_crossentropy',
				metrics=['accuracy']
				)
				
	# define callbacks
	callbacks = []
	callbacks.append(LearningRateScheduler(scheduler))
	callbacks.append(EarlyStopping(
				monitor='val_loss',
				mode='auto',
				verbose=1,
				patience=15,
				restore_best_weights=True)
				)
	callbacks.append(CSVLogger(
					SAVE_PATH+'transfer_learning.log',
					separator=";",
					append=False)
					)
	hist = model.fit_generator(
					generator=train_set,
					steps_per_epoch=STEP_SIZE_TRAIN,
					epochs=epoch,
					verbose=1,
					validation_data=val_set,
					validation_steps=STEP_SIZE_VALID,
					callbacks=callbacks,
					)
	if save_models:
		model.save(SAVE_PATH+f'./{MODEL_NAME}_epochs_{epoch}.h5')
	
	# Evaluate the model on unseen data
	print('Predicting')
	test_set.reset()
	prediction = model.predict_generator(
						test_set,
						steps=test_set.n,
						verbose=1
						)
	#Save predictions and corresponding files
	np.save(SAVE_PATH+'filename_test.npy', test_set.filenames)
	np.save(SAVE_PATH+'predictions_test.npy', prediction)
	
	y_true = test_set.classes
	y_pred = np.argmax(prediction, axis=1)
	
	# Evaluation metrics
	confmat = confusion_matrix(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average='macro')
	acc = accuracy_score(y_true, y_pred)
	
	# Save confusion matrix
	fig, ax = plt.subplots(figsize=(14,14))
	ax_1 = plot_confm(ax, confmat, classes=classes)
	plt.tight_layout()
	plt.savefig(SAVE_PATH+f'cm_test_f1{f1:.3f}_acc{acc:.3f}.png')
	
	#Validation
	val_set_2.reset()
	prediction = model.predict_generator(val_set_2, steps=val_set_2.n,verbose=1)
	
	#Save predictions and corresponding files
	np.save(SAVE_PATH+'filename_validation.npy', val_set_2.filenames)
	np.save(SAVE_PATH+'predictions_validation_n{len(prediction)}_t{end:.4f}.npy', prediction)
	
	y_true = val_set_2.classes
	y_pred = np.argmax(prediction, axis=1)
	
	# Evaluation metrics
	confmat = confusion_matrix(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average='macro')
	acc = accuracy_score(y_true, y_pred)
	
	# Save confusion matrix
	fig, ax = plt.subplots(figsize=(14,14))
	ax_1 = plot_confm(ax, confmat, classes=classes)
	plt.tight_layout()
	plt.savefig(SAVE_PATH+f'cm_val_f1{f1:.3f}_acc{acc:.3f}.png')
	
	
	# Fine tuning the model
	
	# Reset the dataset
	train_set.reset()
	test_set.reset()
	val_set.reset()
	val_set_2.reset()
	
	base_model.trainable = True
	
	# Override callbacks (new csv path)
	callbacks = []
	callbacks.append(LearningRateScheduler(scheduler))
	callbacks.append(EarlyStopping(
				monitor='val_loss',
				mode='auto',
				verbose=1,
				patience=15,
				restore_best_weights=True)
				)
	callbacks.append(CSVLogger(
				SAVE_PATH+'finetune.log',
				separator=";",
				append=False)
				)
	# Newly compile the model with a lower learning rate
	model.compile(
			optimizer=SGD(lr=0.0001, momentum=0.9),
	 		loss='categorical_crossentropy',
	 		metrics=['accuracy']
	 		)
	# Again Fit the model to the training data
	hist = model.fit_generator(
				generator=train_set,
				steps_per_epoch=STEP_SIZE_TRAIN,
				epochs=epoch_fine,
				verbose=1,
				validation_data=val_set,
				validation_steps=STEP_SIZE_VALID,
				callbacks=callbacks,
				)
	if save_models:
		model.save(SAVE_PATH+f'{MODEL_NAME}__fintuned_epochs_{epoch_fine}.h5')
	print('Predicting')
		
	# Evaluate the model on unseen data (Testset)
	test_set.reset()
	start = time.time()
	prediction = model.predict_generator(test_set, steps=test_set.n,verbose=1)
	end = time.time()-start
	
	# Save predictions
	np.save(SAVE_PATH+'predictions_test_finetuned_n{len(prediction)}_t{end:.4f}.npy', prediction)
	y_true = test_set.classes
	y_pred = np.argmax(prediction, axis=1)
	
	# Calcualate metrics
	confmat = confusion_matrix(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average='macro')
	acc = accuracy_score(y_true, y_pred)
	
	# Save confusion matrix
	fig, ax = plt.subplots(figsize=(14,14))
	ax_1 = plot_confm(ax, confmat, classes=classes)
	plt.tight_layout()
	plt.savefig(SAVE_PATH+f'cm_fine_test_f1{f1:.3f}_acc{acc:.3f}.png')
	
	#Evaluate the model on the validation data
	val_set_2.reset()
	prediction = model.predict_generator(val_set_2,steps=val_set_2.n,verbose=1)
	np.save(SAVE_PATH+'predictions_validation_finetuned.npy', prediction)
		
	y_true = val_set_2.classes
	y_pred = np.argmax(prediction, axis=1)
	
	# Calculate metrics
	confmat = confusion_matrix(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average='macro')
	acc = accuracy_score(y_true, y_pred)
	
	# Save confusion matrix
	fig, ax = plt.subplots(figsize=(14,14))
	ax_1 = plot_confm(ax, confmat, classes=classes)
	plt.tight_layout()
	plt.savefig(SAVE_PATH+f'cm_fine_val_f1{f1:.3f}_acc{acc:.3f}.png')


# Main calls
print('Main')
ROOT = '../../../data/SPITZ_REP/'
start = time.time()
for rep in range(1,4):
	for k in range(1,6):
		data_path = ROOT+f'Rep{rep}/Fold {k}/'
		random_state = int(str(rep)+str(k))
		print(data_path[-12:])
		run_repititions(data_path, random_state=random_state)
end = time.time()-start
print(f'Total time: {end:.2f}')

