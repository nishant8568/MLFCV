import scipy
import numpy as np
import matplotlib.pyplot as plt
import sys
from os import listdir
import os.path

CAFFE_ROOT = "../"
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe

MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
MYPATH = 'images'
SYNSET_PATH = '../data/ilsvrc12/synset_words.txt'

def load_test_images():
	imgList = []
	images = listdir(MYPATH)
	for image in images:
		img = caffe.io.load_image(os.path.join(MYPATH, image))
		imgList.append((img, image))
	return imgList

def get_image_details():
	details = []
	with open(SYNSET_PATH) as f:
		for line in f:
			line = line[10:]
			components = line.strip().split(',')
			details.append(components)
	return details

if __name__ == '__main__':
	caffe.set_mode_cpu()
	net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load(CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), channel_swap=(2,1,0), raw_scale=255, image_dims=(256,256))

	#input_image = caffe.io.load_image(IMAGE_FILE)
	image_details = get_image_details()
	classification_details = []
	for input_image, orig_image in load_test_images():
		#plt.imshow(input_image)
		#plt.show()
		prediction = net.predict([input_image])
		prediction_shape = prediction[0].shape
		#print 'Prediction Shape:', prediction_shape
		plt.plot(prediction[0])
		class_predicted = prediction[0].argmax()
		#print 'Predicted Class: ', class_predicted
		label = image_details[class_predicted]
		#print label
		probability = prediction[0][class_predicted]
		#print 'Probability: ', prediction[0][class_predicted]
		entropy = scipy.stats.entropy(prediction[0])
		#print 'Entropy: ', entropy
		classification_details.append((orig_image, class_predicted, label, probability, entropy))
	
	print 'Input Image, Predicted Class, Label, Probability, Entropy'
	for detail in classification_details:	
		print detail
