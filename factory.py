# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np

def weight_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial_value=initial,name=name)


def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial_value=initial,name=name)

def conv2d(x, w, s=1):
	return tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding='SAME')

def deconv2d(x,w):
	return tf.nn.conv2d_transpose(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1], padding='SAME')

def batchnorm(x):
	mean,variance=tf.nn.moments(x,[0,1,2,3])
	return tf.nn.batch_normalization(x,
									mean=mean,
									variance=variance,
									offset=0,
									scale=1,
									variance_epsilon=1e-6)

def conv_layer(x, input_channel, output_channel,k_size=3, relu=True,stride=1,bn=True,name='conv_layer'):
	with tf.name_scope(name):
		w = weight_variable([k_size,k_size,input_channel,output_channel],'weight')
		b = bias_variable([output_channel],'bias')
		answer = conv2d(x,w,s=stride)+b
		if bn:
			answer = batchnorm(answer)
		if relu:
			answer = tf.nn.relu(answer)
		return answer

def res_conv_layer(x, input_channel, output_channel,relu=True,stride=1,name='res_conv_layer'):
	with tf.name_scope(name):
		if input_channel == output_channel and stride == 1:
			conv1 = conv_layer(x,input_channel,output_channel,name='conv1')
			conv2 = conv_layer(conv1,output_channel,output_channel,name='conv2')
			conv3 = conv_layer(conv2,output_channel,output_channel,name='conv3')
			answer = conv3+x
			if relu:
				return tf.nn.relu(answer)
			else:
				return answer
		else:
			conv1 = conv_layer(x,input_channel,output_channel,name='conv1',stride=stride)
			conv2 = conv_layer(conv1,output_channel,output_channel,name='conv2')
			conv3 = conv_layer(conv2,output_channel,output_channel,name='conv3',relu=False)
			conv1_ = conv_layer(x,input_channel,output_channel,name='conv1_',relu=False,stride=stride)
			answer = conv1_+conv3
			if relu:
				return tf.nn.relu(answer)
			else:
				return answer

def Fully_ResNet(x, class_num):
	repeat = 16
	layer_num = 128
	input_row = x.shape[1]
	input_col = x.shape[2]
	input_channel = x.shape[3]
	if [input_row,input_col,input_channel] != [256,256,3]:
		print('U_Net input error: the size of input not matched\n')
		return
	net=batchnorm(x)
	net= res_conv_layer(net,3,64,name='res1')
	for i in range(0,repeat):
		name = 'res1_'+str(i)
		net = res_conv_layer(net,64,64,name=name)
	net = res_conv_layer(net,64,class_num,name='res2',relu=False)
	return net

def UNet_ResNet(x, class_num):
	input_row = x.shape[1]
	input_col = x.shape[2]
	input_channel = x.shape[3]
	if [input_row,input_col,input_channel] != [256,256,3]:
		print('U_Net input error: the size of input not matched\n')
		return
	#norm=batchnorm(x)
	net_res_conv1 = res_conv_layer(x,3,64,name='res_conv1',relu=True,stride=1)	#256x256x64
	net_res_conv2 = res_conv_layer(net_res_conv1,64, 128,name='res_conv2',relu=True,stride=2)	#128x128x128
	net_res_conv3 = res_conv_layer(net_res_conv2,128,256,name='res_conv3',relu=True,stride=2)	#64x64x256
	net_res_conv4 = res_conv_layer(net_res_conv3,256,512,name='res_conv4',relu=True,stride=2)	#32x32x512
	net_res_conv5 = res_conv_layer(net_res_conv4,512,512,name='res_conv5',relu=True,stride=1)	#32x32x512

	net_up6 = tf.image.resize_bilinear(net_res_conv5,[64,64],name='upsample1')	#64x64x512
	net_res_conv3_cut = res_conv_layer(net_res_conv3, 256,512,name='res_conv3_cut',relu=True,stride=1) #64x64x512
	net_fp6 = net_up6 + net_res_conv3_cut	#64x64x512
	net_res_conv6 = res_conv_layer(net_fp6,512,512,name='res_conv6',relu=True,stride=1)	#64x64x512

	net_up7 = tf.image.resize_bilinear(net_res_conv6,[128,128],name='upsample2')	#128x128x512
	net_res_conv2_cut = res_conv_layer(net_res_conv2, 128,512,name='res_conv2_cut',relu=True,stride=1)	#128x128x512
	net_fp7 = net_up7 + net_res_conv2_cut	#128x128x512
	net_res_conv7 = res_conv_layer(net_fp7,512,512,name='res_conv7',relu=True,stride=1)	#128x128x512

	net_up8 = tf.image.resize_bilinear(net_res_conv7,[256,256],name='upsample3')	#256x256x512
	net_res_conv1_cut = res_conv_layer(net_res_conv1, 64, 512,name='res_conv1_cut',relu=True,stride=1)	#256x256x512
	net_fp8 = net_up8 + net_res_conv1_cut	#256x256x512
	net_res_conv8 = res_conv_layer(net_fp8,512,512,name='res_conv8',relu=True,stride=1)	#256x256x512
	net_fc = conv_layer(net_res_conv8,512,class_num,k_size=1,name='fc',relu=False,bn=False,stride=1)

	return net_fc


def U_Net(x, class_num):
	input_row = x.shape[1]
	input_col = x.shape[2]
	input_channel = x.shape[3]
	if [input_row,input_col,input_channel] != [256,256,3]:
		print('U_Net input error: the size of input not matched\n')
		return
	norm=batchnorm(x)
	net_conv1 = conv_layer(norm,3,64,name='conv1')	#256x256
	net_conv2 = conv_layer(net_conv1,64,64,name='conv2')
	net_pool1 = max_pool_2x2(net_conv2)

	net_conv3 = conv_layer(net_pool1,64,128,name='conv3')	#128x128
	net_conv4 = conv_layer(net_conv3,128,128,name='conv4')
	net_pool2 = max_pool_2x2(net_conv4)

	net_conv5 = conv_layer(net_pool2,128,256,name='conv5')	#64x64
	net_conv6 = conv_layer(net_conv5,256,256,name='conv6')
	net_pool3 = max_pool_2x2(net_conv6)

	net_conv7 = conv_layer(net_pool3,256,512,name='conv7')	#32x32
	net_conv8 = conv_layer(net_conv7,512,512,name='conv8')

	net_conv9 = conv_layer(net_conv8,512,256,name='conv9')	
	net_up1 = tf.image.resize_bilinear(net_conv9, [64,64],name='upsample1')	#64x64
	net_concat1 = tf.concat([net_up1,net_conv6],axis=-1,name='concat1')
	net_conv10 = conv_layer(net_concat1,512,256,name='conv10')
	net_conv11 = conv_layer(net_conv10,256,256,name='conv11')

	net_conv12 = conv_layer(net_conv11,256,128,name='conv12')
	net_up2 = tf.image.resize_bilinear(net_conv12,[128,128],name='upsample2')	#128x128
	net_concat2 = tf.concat([net_up2,net_conv4],axis=-1,name='concat2')
	net_conv13 = conv_layer(net_concat2,256,128,name='conv13')
	net_conv14 = conv_layer(net_conv13,128,128,name='conv14')

	net_conv15 = conv_layer(net_conv14,128,64,name='conv15')
	net_up3 = tf.image.resize_bilinear(net_conv15,[256,256],name='upsample3')	#256x256
	net_concat3 = tf.concat([net_up3,net_conv2],axis=-1,name='concat3')
	net_conv16 = conv_layer(net_concat3,128,64,name='conv16')
	net_conv17 = conv_layer(net_conv16,64,64,name='conv17')

	net_conv18 = conv_layer(net_conv17,64,class_num,k_size=1,name='conv18',relu=False)

	return net_conv18
