import scipy
import numpy as np
import matplotlib.pyplot as plt
import sys
from os import listdir
import os.path
import PIL
from PIL import Image

CAFFE_ROOT = "../"
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe

MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
MYPATH = 'images'
PREPROCESSED_IMAGES = 'preprocessed'
SYNSET_PATH = '../data/ilsvrc12/synset_words.txt'

# this function loads images to be classified (test set)
def load_raw_images():
	imgList = []
	images = os.listdir(MYPATH)
	for image in images:
		imgList.append(image)
	return imgList

# this function loads the preprocessed input images
def load_test_images():
	imgList = []
	images = listdir(PREPROCESSED_IMAGES)
	for image in images:
		img = caffe.io.load_image(os.path.join(PREPROCESSED_IMAGES, image))
		imgList.append((img, image))
	return imgList

# this function reads, manipulate the Synset text file and returns the required details of image
def get_image_details():
	details = []
	with open(SYNSET_PATH) as f:
		for line in f:
			line = line[10:]
			components = line.strip().split(',')
			details.append(components)
	return details

# this function resizes image to the dimensions 256x256
def preprocess(image):
	img = Image.open(os.path.join(MYPATH, image))
        hsize = img.size[1]
        wsize = img.size[0]
        if(hsize<wsize):
                img = img.resize((img.size[0], 256), PIL.Image.ANTIALIAS)
                print 'height resized to 256'
                print img.size[1]
                #plt.imshow(img)
                #plt.show()
                #img.save(os.path.join(PREPROCESSED_IMAGES,(image+'_resized_image_height.jpg')))

		to_be_cropped = wsize-256
                box = (to_be_cropped/2, 0, wsize-(to_be_cropped/2), img.size[1])
                img = img.crop(box)
                img.save(os.path.join(PREPROCESSED_IMAGES, image))
		plt.imshow(img)
		plt.show()
	else:
                img = img.resize((256, img.size[1]), PIL.Image.ANTIALIAS)
                print 'width resized to 256'
                print img.size[0]
                #plt.imshow(img)
                #plt.show()
                #img.save(os.path.join(PREPROCESSED_IMAGES,(image+'_resized_image_width.jpg')))

                to_be_cropped = hsize-256
                box = (0, to_be_cropped/2, img.size[0], hsize-(to_be_cropped/2))
                img = img.crop(box)
                img.save(os.path.join(PREPROCESSED_IMAGES, image))
		plt.imshow(img)
		plt.show()

if __name__ == '__main__':
	caffe.set_mode_cpu()
	net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load(CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), channel_swap=(2,1,0), raw_scale=255, image_dims=(256,256))

	#input_image = caffe.io.load_image(IMAGE_FILE)
	image_details = get_image_details()

	imgList = load_raw_images()
	print imgList
	for image in imgList:
		preprocess(image)

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
