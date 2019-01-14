# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import tensorflow as tf
import os
import factory
import dataset
import argparse
import csv
import utils

def write_csv(path, index,label):
	label = np.int32(label)
	file_path = os.path.join(path,'%d.csv'%index)
	fileobj=open(file_path,'w')
	writer = csv.writer(fileobj)
	row = label.shape[0]
	col = label.shape[1]
	ret=[]
	for c in range(0,col):
		for r in range(0,row):
			ret.append(label[r,c])
	writer.writerow(ret)

def save_y(path, index, label):
	row = label.shape[0]
	col = label.shape[1]
	file_path = os.path.join(path,'%s_result.png'%index)
	image = np.zeros((row,col,3))
	# image[label==1] = [0,255,0]
	# image[label==2] = [0,0,255]
	# image[label==3] = [0,255,255]
	# image[label==4] = [255,0,0]
	# image[label==0] = [0,0,0]

	image[label==1] = [0,255,0]
	image[label==2] = [0,255,255]
	image[label==3] = [255,0,0]
	image[label==4] = [0,0,255]
	image[label==0] = [0,0,0]
	image = np.uint8(image)
	cv2.imwrite(file_path,image)

def main(args):
	dataset_path = './BDCI2017-jiage-Semi'
	model_path='./model'
	model_name='UNet_ResNet_itr100000'
	model_file = os.path.join(model_path,'%s.ckpt'%model_name)
	period = 'test'
	csv_path = './CSV'
	class_num = 5
	sample_num = 3
	patch_size=256
	sample_size = 1024
	rate = sample_size/patch_size
	batch_size=1
	accuracy = 0
	radius = 8
	eps = 0.2*0.2
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	file_path = os.path.join(dataset_path,'test')
	print('NO.1 test sample is loading...')
	image = dataset.load_image(file_path,index=1,load_label=False)
	ori_row = image.shape[0]
	ori_col = image.shape[1]
	image = cv2.resize(image,(np.int32(image.shape[1]/rate),np.int32(image.shape[0]/rate)))
	print('loading finished')
	row = image.shape[0]
	col = image.shape[1]
	
	x = tf.placeholder(tf.float32,[batch_size,patch_size,patch_size,3])
	net = factory.UNet_ResNet(x,class_num)

	net_sub = tf.slice(net,[0,0,0,1],[1,256,256,4])
	#CRF
	net_softmax = tf.nn.softmax(net_sub)#########attention net not net_sub
	x_int = (1+x)*128
	x_int = tf.cast(x_int,dtype=tf.uint8)
	result = tf.py_func(utils.dense_crf, [net_softmax, x_int], tf.float32)
	# result = tf.argmax(result,axis=-1)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	print('start restore parameter...')
	saver.restore(sess,model_file)
	print('parameter restore finished!')
	offs=int(patch_size/4)
	offe=int(3*patch_size/4)
	for n in range(1,sample_num+1):
		if n!=1:
			print('NO.%d test sample is loading...'%n)
			image = dataset.load_image(file_path,index=n,load_label=False)
			ori_row = image.shape[0]
			ori_col = image.shape[1]
			image = cv2.resize(image,(np.int32(image.shape[1]/rate),np.int32(image.shape[0]/rate)))
			row = image.shape[0]
			col = image.shape[1]
			print('loading finished')
		print('float transforming...')
		image = np.float32(image)/128.0-1.0
		vote = np.zeros((row,col,class_num-1))#original class_num-1
		
		sub_image = image[0:patch_size,0:patch_size,:]
		sub_image = np.reshape(sub_image,[1,patch_size,patch_size,3])
		cls_result = sess.run(result,feed_dict={x:sub_image})
		vote[0:offe,0:offe]\
			 = cls_result[0,0:offe,0:offe]
		for c in range(0,col-patch_size,int(patch_size/2)):
			sub_image = image[0:patch_size,c:c+patch_size,:]
			sub_image = np.reshape(sub_image,[1,patch_size,patch_size,3])
			cls_result = sess.run(result,feed_dict={x:sub_image})
			vote[0:offe,c+offs:c+offe] \
				= cls_result[0,0:offe,offs:offe]
		sub_image = image[0:patch_size,col-patch_size:col,:]
		sub_image = np.reshape(sub_image,[1,patch_size,patch_size,3])
		cls_result = sess.run(result,feed_dict={x:sub_image})
		vote[0:offe,col-patch_size+offs:col]\
			 = cls_result[0,0:offe,offs:]
		for r in range(0,row-patch_size,int(patch_size/2)):
			print('sample%d,row:%d patch is processing'%(n,r))
			sub_image = image[r:r+patch_size,0:patch_size,:]
			sub_image = np.reshape(sub_image,[1,patch_size,patch_size,3])
			cls_result = sess.run(result,feed_dict={x:sub_image})
			vote[r+offs:r+offe,0:offe] \
				= cls_result[0,offs:offe,0:offe]
			for c in range(0,col-patch_size,int(patch_size/2)):
				sub_image = image[r:r+patch_size,c:c+patch_size,:]
				sub_image = np.reshape(sub_image,[1,patch_size,patch_size,3])
				cls_result = sess.run(result,feed_dict={x:sub_image})
				vote[r+offs:r+offe,c+offs:c+offe]\
					 = cls_result[0,offs:offe,offs:offe]
			sub_image = image[r:r+patch_size,col-patch_size:col,:]
			sub_image = np.reshape(sub_image,[1,patch_size,patch_size,3])
			cls_result = sess.run(result,feed_dict={x:sub_image})
			vote[r+offs:r+offe,col-patch_size+offs:col] \
				= cls_result[0,offs:offe,offs:patch_size]
			
		sub_image = image[row-patch_size:row,0:patch_size,:]
		sub_image = np.reshape(sub_image,[1,patch_size,patch_size,3])
		cls_result = sess.run(result,feed_dict={x:sub_image})
		vote[row-patch_size+offs:row,0:offe]\
			 = cls_result[0,offs:,0:offe]
		for c in range(0,col-patch_size,int(patch_size/2)):
			sub_image = image[row-patch_size:row,c:c+patch_size,:]
			sub_image = np.reshape(sub_image,[1,patch_size,patch_size,3])
			cls_result = sess.run(result,feed_dict={x:sub_image})
			vote[row-patch_size+offs:row,c+offs:c+offe] \
				= cls_result[0,offs:patch_size,offs:offe]
		sub_image = image[row-patch_size:row,col-patch_size:col,:]
		sub_image = np.reshape(sub_image,[1,patch_size,patch_size,3])
		cls_result = sess.run(result,feed_dict={x:sub_image})
		vote[row-patch_size+offs:row,col-patch_size+offs:col]\
			 = cls_result[0,offs:,offs:]

		# gray_img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)		
		# for channel in range(0,class_num-1):
		# 	temp_vote = vote[:,:,channel]
		# 	vote[:,:,channel] = guidedfilter.guidedfilter(temp_vote,gray_img,radius,eps)
		# vote_softmax = vote.copy()
		# vote_softmax = vote_softmax[:,:,[1,3]]
		# vote_pb = np.argmax(vote_softmax,axis=-1)
		# vote_pb[vote_pb==1]=3
		# vote_pb[vote_pb==0]=1
		vote = np.argmax(vote,axis=-1)
		vote = np.uint8(vote)
		# vote[vote==0]=vote_pb[vote==0]
		# vote = cv2.medianBlur(vote,7)
		vote = vote+1
		copy_vote = vote.copy()
		copy_vote[vote==2] = 4
		copy_vote[vote==3] = 2
		copy_vote[vote==4] = 3
		vote = copy_vote.copy()
		vote = cv2.resize(vote,(ori_col,ori_row),cv2.INTER_NEAREST)
		
		print('%d test sample is writing into csv...'%n)
		write_csv(csv_path,index=n,label=vote)
		print('%d test result is writing into png...'%n)
		save_y(file_path,n,vote)
		print('writing finished')
	print('accuracy: %f'%accuracy)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
	parser.add_argument("--gpu", choices=['0','1','2','3'], default='0', help="gpu_id")
	args = parser.parse_args()
	main(args)
