# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import tensorflow as tf
import os
import factory
import argparse


def load_data(path,patch_size=256):
	filename_queue = tf.train.string_input_producer([path])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,
										features={
											'row': tf.FixedLenFeature([],tf.int64),
											'col': tf.FixedLenFeature([],tf.int64),
											'image_raw': tf.FixedLenFeature([],tf.string),
											'label_raw': tf.FixedLenFeature([],tf.string),
										})
	image = tf.decode_raw(features['image_raw'], tf.uint8)
	image = tf.reshape(image,[patch_size,patch_size,3])
	label = tf.decode_raw(features['label_raw'], tf.uint8)
	label = tf.reshape(label,[patch_size,patch_size])
	return image, label

def main(args):
	dataset_path = './TFRecord'
	log_path='./log'
	model_path='./model'
	#model_name='UNet_ResNet_pure_itr50000'
	#model_file = os.path.join(model_path,'%s.ckpt'%model_name)
	period = 'train'
	patch_size=256
	batch_size = 1
	max_iteration = 120000
	class_num = 5
	learning_rate=1e-5
	momentum = 0.99
	os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
	
	file_path = os.path.join(dataset_path,'%s.tfrecord'%period)
	image, label = load_data(file_path,patch_size)
	image_batch, label_batch = tf.train.shuffle_batch([image,label], 
										batch_size = batch_size,
										capacity=5000,
										min_after_dequeue=100
										)
	label_one_hot = tf.one_hot(indices = label_batch,
								depth = class_num,
								on_value=1,
								off_value=0,
								)
	mini_batch = tf.cast(image_batch,dtype=tf.float32)

	x = tf.placeholder(tf.float32,[batch_size,patch_size,patch_size,3])
	y = tf.placeholder(tf.float32,[batch_size,patch_size,patch_size,class_num])
	net = factory.UNet_ResNet(x,class_num)
	loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(labels = y,
													logits = net,
													)))
	correct_prediction = tf.equal(tf.argmax(net,axis=-1),tf.argmax(y,axis=-1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('entropy_loss', loss)
	tf.summary.scalar('accuracy', accuracy)
	train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
	saver=tf.train.Saver()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	#saver.restore(sess,model_file)
	print('variables initialized.')
	threads = tf.train.start_queue_runners(sess=sess)
	merged = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(log_path,sess.graph)
	print('training start...')
	for itr in range(1,max_iteration+1):
		step_batch, step_label = sess.run([mini_batch,label_one_hot])
		step_batch = np.float32(step_batch)/128.0 - 1.0
		feed_dict={x:step_batch, y:step_label}
		sess.run(train_step,feed_dict=feed_dict)
		if itr%10==0:
			summary,train_loss,train_accuracy = sess.run([merged,loss,accuracy],feed_dict=feed_dict)
			print('iteration %d, loss:%f, acc:%f'%(itr,train_loss,train_accuracy))
			summary_writer.add_summary(summary, itr)
		if itr%30000==0:
			save_path = os.path.join(model_path,'UNet_ResNet_itr%d.ckpt'%(itr))
			saver.save(sess,save_path)
			print('model parameter has been saved in %s.'%save_path)

			

if __name__ == '__main__':
	parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
	parser.add_argument("--gpu", choices=['0','1','2','3'], default='0', help="gpu_id")
	args = parser.parse_args()
	main(args)
