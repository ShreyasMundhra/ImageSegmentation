import skimage.io as io

from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Conv2DTranspose, Add, Reshape
from keras.layers.core import Activation

from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.applications.vgg19 import VGG19

from pycocotools.coco import COCO
from pycocotools.cocostuffeval import *
from pycocotools.cocostuffhelper import *

# get ground truth for an image in the output format of the model
def get_ground_truth(coco, imgId, num_classes=92, class_start_id=92):
	label_map = cocoSegmentationToSegmentationMap(coco, imgId, checkUniquePixelLabel=True)
	ground_truth = np.zeros((label_map.shape[0], label_map.shape[1], num_classes))

	for i in range(num_classes):
		ground_truth[:, :, i][label_map == class_start_id + i] = 1
	return ground_truth

# batch generator
def batch(coco, input_shape, n=32):
	imgIds = coco.getImgIds()

	l = len(imgIds)
	for ndx in range(0, l, n):
		batch_ids = imgIds[ndx:min(ndx + n, l)]
		imgs_dict = coco.loadImgs(batch_ids)

		imgs = []
		y = []

		for i in range(len(batch_ids)):
			img = imgs_dict[i]
			imgId = batch_ids[i]

			image = io.imread(img['coco_url'])
			if len(image.shape) != 3:
				image = np.asarray(np.dstack((image, image, image)), dtype=np.uint8)

			reshaped = np.zeros((input_shape[0], input_shape[1], 3))

			cropped_h = min(image.shape[0], input_shape[0])
			cropped_w = min(image.shape[1], input_shape[1])
			reshaped[:cropped_h, :cropped_w] = image[:cropped_h, :cropped_w]
			imgs.append(reshaped.tolist())

			gt = get_ground_truth(coco, imgId, num_classes=92, class_start_id=92)
			reshaped_gt = np.zeros((input_shape[0], input_shape[1], gt.shape[2]))
			reshaped_gt[:cropped_h, :cropped_w] = gt[:cropped_h, :cropped_w]

			y.append(reshaped_gt.tolist())
		yield np.array(imgs), np.array(y)

# generate neural network model
def gen_model():
	model = VGG19(include_top=False, input_shape=(256, 256, 3))
	for layer in model.layers:
		layer.trainable = False

	for i in range(5):
		model.layers.pop()

	out = Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same')(model.layers[-1].output)
	out = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(out)
	out = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(out)

	skip_connection2 = model.layers[5].output
	out = Add()([skip_connection2, out])

	out = Conv2DTranspose(92, (3, 3), strides=(2, 2), activation='relu', padding='same')(out)

	# input shape: (256, 256, 3), output shape: (256, 256, 92)
	seg_model = Model(inputs=model.inputs, outputs=[out])
	return seg_model

# get an estimated memory usage for the model in GB
def get_model_memory_usage(batch_size, model):
	shapes_mem_count = 0
	for l in model.layers:
		single_layer_mem = 1
		for s in l.output_shape:
			if s is None:
				continue
			single_layer_mem *= s
		shapes_mem_count += single_layer_mem

	trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
	non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

	number_size = 4.0
	if K.floatx() == 'float16':
		number_size = 2.0
	if K.floatx() == 'float64':
		number_size = 8.0

	total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
	gbytes = np.round(total_memory / (1024.0 ** 3), 3)
	return gbytes

# custom loss function for pixelwise crossentropy
def pixelwise_crossentropy(y_true, y_pred):
	reshaped_output = Reshape((-1, 92))(y_pred)
	new_output = Activation('softmax')(reshaped_output)
	return categorical_crossentropy(Reshape((-1, 92))(y_true), new_output)

# train model
def train_model():
	# Comment this if loading the model from model file
	model = gen_model()
	# Uncomment this if loading model from model file
	# model = load_model('./model.h5', custom_objects={'pixelwise_crossentropy': pixelwise_crossentropy})

	model.compile(loss=pixelwise_crossentropy, optimizer='adam')

	model_path = './model.h5'
	callbacks = [
	EarlyStopping(
		monitor='loss',
		patience=10,
		mode='min',
		verbose=1),
	ModelCheckpoint(model_path,
		monitor='loss',
		save_best_only=True,
		mode='min',
		verbose=0)
	]

	train_file = 'annotations/stuff_train2017.json'
	coco_train = COCO(train_file)

	val_file = 'annotations/stuff_val2017.json'
	coco_val = COCO(val_file)

	training_generator = batch(coco_train, (256, 256, 3), 10)
	validation_generator = batch(coco_val, (256, 256, 3), 10)
	history = model.fit_generator(generator=training_generator,
					validation_data=validation_generator,
					epochs=15,
					use_multiprocessing=False,
					workers=0,
					shuffle=True,
					steps_per_epoch=20,
					validation_steps=20,
					callbacks=callbacks)

	return model, history

# convert output format of model to label map
# each item in the label map is the label of its corresponding pixel in the image
def convert_pred_to_label_map(pred, class_start_id=92):
	return class_start_id + np.argmax(pred, axis=2)

# evaluate model
def evaluate(val_file):
	print("Evaluating...")

	cocoGt = COCO(val_file)
	num_correct = 0

	model = gen_model()

	batch_num = 1
	validation_generator = batch(cocoGt, (256, 256, 3), 10)
	for images, gt in validation_generator:
		start = time.time()
		print("Batch number: {}".format(batch_num))
		y = model.predict(images)

		for i in range(y.shape[0]):
			labelMap_pred = convert_pred_to_label_map(y[i])
			labelMap_gt = convert_pred_to_label_map(gt[i])

			num_correct = num_correct + len(labelMap_pred == labelMap_gt)

		end = time.time()
		print("Time taken: {}".format(end - start))
		batch_num = batch_num + 1

	mean_pix_level_accuracy = num_correct / float(256 * 256 * (batch_num - 1) * 10)
	return mean_pix_level_accuracy

if __name__ == "__main__":
	print("Memory required for model: {} GB".format(get_model_memory_usage(10, gen_model())))
	train_model()
	val_file = 'annotations/stuff_val2017.json'
	print("Mean pixel-level accuracy: {}".format(evaluate(val_file)))