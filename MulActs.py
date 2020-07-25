import os
import sys
import shutil
import imp
import time
import h5py

import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from openpyxl import Workbook

os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
# 模型/程序的全局参数
flags = tf.app.flags
FLAGS = flags.FLAGS

# 模型选择
flags.DEFINE_string('flag_net_module', './ModuleUCF11LSTM_5fold.py', 'Module selection with specific dataset.')

# 全局参数，tf的log文件夹位置
flags.DEFINE_string('flag_log_dir', './log', 'Directory to put log files.')

# 最大epoch次数
# CIFAR-10(ResNet): 120
flags.DEFINE_integer('flag_max_epochs', 100, 'Maximum number of epochs to train.')

# batch size大小
flags.DEFINE_integer('flag_batch_size', 20, 'Batch size which must be divided extractly by the size of dataset.')

# 标记，网络选择
flags.DEFINE_boolean('flag_compressed_network', True, 'Network to be chosen. True means compressed network.')

# 学习率
flags.DEFINE_float('flag_learning_rate', 0.001, 'Learning rate to define the momentum optimizer.')

# 学习率衰减
flags.DEFINE_float('flag_lr_decay', 30.0, 'Epochs to decay learning rate.')


# 求平均梯度
def average_gradients(tower_grads):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		grads = []
		for g, _ in grad_and_vars:
			expanded_g = tf.expand_dims(g, 0)
			grads.append(expanded_g)
		grad = tf.concat(grads, 0)
		grad = tf.reduce_mean(grad, 0)
		grad_and_var = (grad, grad_and_vars[0][1])
		average_grads.append(grad_and_var)
	return average_grads


def run_training(b_gpu_enabled = False, str_restore_ckpt = None):
	network = imp.load_source('network', FLAGS.flag_net_module)

	with tf.Graph().as_default(), tf.device('/cpu:0'):
		print('Begin to get dataset.')
		dict_dataset, dict_mean_std = network.get_dataset()
		print('Get dataset has done.')

		# gpu数量
		n_num_gpus =2
		# if b_gpu_enabled == True:
		# 	l_devices = device_lib.list_local_devices()
		# 	for i in range(len(l_devices)):
		# 		if l_devices[i].device_type == 'GPU':
		# 			n_num_gpus += 1
		# 	n_num_gpus -= 1

		# 迭代步数设置，初始化为0
		tfv_global_step = tf.get_variable('var_global_step', [], tf.int32, tf.constant_initializer(0, tf.int32), trainable = False)

		# 训练标记变量，为True则执行训练过程，为False则执行估计过程
		tfv_train_phase = tf.Variable(True, trainable = False, name = 'var_train_phase', dtype = tf.bool, collections = [])

		# 滑动平均模型对象，用于所有训练变量
		tfob_variable_averages = tf.train.ExponentialMovingAverage(0.9, name = 'avg_variable')

		# 动量优化器
		n_decay_steps = int(FLAGS.flag_lr_decay * np.array(dict_dataset['train']['train_labels']).shape[0]/ FLAGS.flag_batch_size)
		f_learning_rate = tf.train.exponential_decay(FLAGS.flag_learning_rate, tfv_global_step, n_decay_steps, 0.1, staircase = True)
		optim = tf.train.MomentumOptimizer(f_learning_rate, 0.9)

		dict_inputs_batches = network.construct_batch_part(dict_mean_std, FLAGS.flag_batch_size * n_num_gpus)
		dict_phs = dict_inputs_batches['input_placeholders']
		t_labels = dict_inputs_batches['batches']['batch_train_labels']
		v_labels = dict_inputs_batches['batches']['batch_validation_labels']
		t_data = dict_inputs_batches['batches']['batch_train_data']
		v_data = dict_inputs_batches['batches']['batch_validation_data']
		t_data_split = tf.split(t_data, n_num_gpus)
		t_labels_split = tf.split(t_labels, n_num_gpus)
		v_data_split = tf.split(v_data, n_num_gpus)
		v_labels_split = tf.split(v_labels, n_num_gpus)

		tower_losses_t = []
		tower_evals_t = []
		tower_losses_v = []
		tower_evals_v = []
		tower_grads = []		
		for i in range(n_num_gpus):
			with tf.device('/gpu:%d' % i):
				loss_t, eval_t, loss_v, eval_v = network.get_network_output(i, t_data_split[i], t_labels_split[i], v_data_split[i], v_labels_split[i], FLAGS.flag_batch_size,FLAGS.flag_compressed_network,tfv_train_phase)

				tower_losses_t.append(loss_t)
				tower_evals_t.append(eval_t)
				tower_losses_v.append(loss_v)
				tower_evals_v.append(eval_v)

				# 使用动量优化器optim计算梯度
				grads = optim.compute_gradients(loss_t)

				# 梯度裁剪
				grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads if grad is not None]
				tower_grads.append(grads)

		# 平均多个gpu上的梯度
		grads = average_gradients(tower_grads)

		# 更新梯度操作，使用动量优化器optim
		tfop_apply_gradients = optim.apply_gradients(grads, tfv_global_step)
		with tf.control_dependencies([tfop_apply_gradients]):
			# 多gpu的global_step修正
			tfop_normalize_gs = tfv_global_step.assign_add(n_num_gpus - 1)

		# 变量平均操作，所有训练变量应用滑动平均模型
		tfop_variable_averages_apply = tfob_variable_averages.apply(tf.trainable_variables())

		# 训练集的误差和准确率变量
		tfv_train_loss = tf.Variable(5.0, trainable = False, name = 'var_train_loss', dtype = tf.float32)
		tfv_train_precision = tf.Variable(0.0, trainable = False, name = 'var_train_precision', dtype = tf.float32)

		# 训练更新操作，训练集的误差和准确率变化操作
		l_ops_train_lp_update = []
		for i in range(n_num_gpus):
			l_ops_train_lp_update.append(tfv_train_loss.assign_sub(0.1 * (tfv_train_loss - tower_losses_t[i])))
			new_precision = tf.reduce_mean(tf.cast(tower_evals_t[i], tf.float32))
			l_ops_train_lp_update.append(tfv_train_precision.assign_sub(0.1 * (tfv_train_precision - new_precision)))
		tfop_train_lp_update = tf.group(*l_ops_train_lp_update)

		# 训练操作打包，将更新梯度操作、变量平均操作、训练更新操作打包为一个训练操作
		tfop_train = tf.group(tfop_apply_gradients, tfop_normalize_gs, tfop_variable_averages_apply, tfop_train_lp_update)

		# 变量保存器，用于写入checkpoint
		tfob_saver = tf.train.Saver(tf.global_variables())
		tfob_saver_ema = tf.train.Saver(tfob_variable_averages.variables_to_restore())

		# 程序运行的Session
		if b_gpu_enabled == True:
			tfob_sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, gpu_options = tf.GPUOptions(allow_growth = True, per_process_gpu_memory_fraction = 0.95)))
		else:
			tfob_sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, device_count = {'GPU': 0}))

		# 执行全部变量初始化
		tfob_sess.run(tf.global_variables_initializer())

		# 线程协调器及多线程启动数据队列填充
		tfob_coord = tf.train.Coordinator()
		th_threads = tf.train.start_queue_runners(tfob_sess, tfob_coord)

		# 每个epoch中step的数量及开始时的epoch
		n_epoch_steps = int(np.array(dict_dataset['train']['train_labels'] ).shape[0]/ FLAGS.flag_batch_size + 0.5)
		n_start_epoch = 0
		if str_restore_ckpt is not None:
			tfob_saver.restore(tfob_sess, str_restore_ckpt)
			sys.stdout.write('Previously started training session restored from "%s".\n' % str_restore_ckpt)
			n_start_epoch = int(tfob_sess.run(tfv_global_step)) // n_epoch_steps
		sys.stdout.write('Starting with epoch #%d.\n' % (n_start_epoch + 1))

		# 记录每epoch训练效果
		l_rc_loss_pre = []
		if os.path.exists(FLAGS.flag_log_dir + '/learning_curve.h5'):
			with h5py.File(FLAGS.flag_log_dir + '/learning_curve.h5', 'r') as file:
				arr_rc_loss_pre = file.get('curve').value
			l_rc_loss_pre = arr_rc_loss_pre.tolist()

		# 训练循环开始，从n_start_epoch到flag_max_epochs
		for n_epoch in range(n_start_epoch, FLAGS.flag_max_epochs):
			sys.stdout.write('\n')
			cur_loss_pre = []

			# -------------------------------------------------------------------------------------------------
			# Training begin! 训练标记变量设置为True，表示当前epoch训练阶段开始
			tfob_sess.run(tfv_train_phase.assign(True))
			sys.stdout.write('Epoch #%d. [Train]\n' % (n_epoch + 1))
			sys.stdout.flush()

			# epoch内步数和每次在数据集中取batch_size数据的头索引
			n_step = 0
			n_index = 0

			# 当前epoch训练过程
			while n_step < n_epoch_steps:
				# 当前step喂一个batch_size大小的数据
				dict_input_feed = network.get_batch_part_train(dict_dataset, dict_mean_std, dict_phs, n_index, FLAGS.flag_batch_size * n_num_gpus)

				# 执行训练
				_, loss_train, eval_train = tfob_sess.run([tfop_train, loss_t, eval_t], dict_input_feed)
				assert not np.isnan(loss_train), 'Model diverged with loss = NaN.'				
				n_step += n_num_gpus
				n_index += (FLAGS.flag_batch_size * n_num_gpus)

				sys.stdout.write('Epoch #%d [Train]. Step %d/%d. Batch loss = %.2f. Batch precision = %.2f.' % 
					 (n_epoch + 1, n_step, n_epoch_steps, loss_train, np.mean(eval_train) * 100.0))
				sys.stdout.write('\n')
				sys.stdout.flush()

			# Training end! 当前训练集结果估计，并记录checkpoint
			train_loss_value, train_precision_value = tfob_sess.run([tfv_train_loss, tfv_train_precision])
			sys.stdout.write('Epoch #%d. Train loss = %.2f. Train precision = %.2f.\n' % 
					(n_epoch + 1, train_loss_value, train_precision_value * 100.0))
			cur_loss_pre += [train_loss_value, train_precision_value * 100.0]
			str_checkpoint_path = os.path.join(FLAGS.flag_log_dir, 'model.ckpt')
			str_ckpt = tfob_saver.save(tfob_sess, str_checkpoint_path, tfv_global_step)
			sys.stdout.write('Checkpoint "%s" is saved.\n\n' % str_ckpt)
			# -------------------------------------------------------------------------------------------------

			# -------------------------------------------------------------------------------------------------
			# Evaluate begin! 训练标记变量设置为False，表示当前epoch验证阶段开始，使用滑动平均模型读取
			tfob_sess.run(tfv_train_phase.assign(False))
			sys.stdout.write('Epoch #%d. [Evaluation]\n' % (n_epoch + 1))
			tfob_saver_ema.restore(tfob_sess, str_ckpt)
			sys.stdout.write('EMA variables restored.\n')

			# 验证集容量，并根据容量计算步数
			n_val_count = np.array(dict_dataset['validation']['validation_labels']).shape[0]
			n_val_steps = (n_val_count + FLAGS.flag_batch_size - 1) // FLAGS.flag_batch_size

			# 每次在数据集中取batch_size数据的头索引
			n_index = 0

			# 正例数量和误差
			n_val_corrects = 0
			n_val_losses = 0.0

			# 当前epoch验证过程
			while n_val_count > 0:
				# 当前step喂一个batch_size大小的数据
				dict_input_feed = network.get_batch_part_validation(dict_dataset, dict_mean_std, dict_phs, n_index, FLAGS.flag_batch_size * n_num_gpus)

				# 执行验证
				eval_validation_and_loss_validation = tfob_sess.run(tower_evals_v + tower_losses_v, dict_input_feed)
				eval_validation = np.concatenate(eval_validation_and_loss_validation[:n_num_gpus], axis = 0)
				loss_validation = eval_validation_and_loss_validation[-n_num_gpus:]
				n_cnt = min(eval_validation.shape[0], n_val_count)
				n_val_count -= n_cnt
				n_cur_step = n_val_steps - (n_val_count + FLAGS.flag_batch_size - 1) // FLAGS.flag_batch_size
				n_index += (FLAGS.flag_batch_size * n_num_gpus)

				# 正例数累加
				n_val_corrects += np.sum(eval_validation[:n_cnt])

				# 平均误差累加
				n_val_losses += np.sum(loss_validation) * FLAGS.flag_batch_size

				sys.stdout.write('Epoch #%d [Evaluation]. Step %d/%d. Batch loss = %.2f. Batch precision = %.2f.' % 
					 (n_epoch + 1, n_cur_step, n_val_steps, np.mean(loss_validation), np.mean(eval_validation) * 100.0))
				sys.stdout.write('\n')
				sys.stdout.flush()

			# Evaluate end! 当前验证集结果估计，并重新以非滑动平均模型读取checkpoint进行下一次epoch
			validation_precision_value = n_val_corrects / np.array(dict_dataset['validation']['validation_labels']).shape[0]
			validation_loss_value = n_val_losses /np.array( dict_dataset['validation']['validation_labels']).shape[0]
			sys.stdout.write('Epoch #%d. Validation loss = %.2f. Validation precision = %.2f.\n' % 
					(n_epoch + 1, validation_loss_value, validation_precision_value * 100.0))
			cur_loss_pre += [validation_loss_value, validation_precision_value * 100.0]
			tfob_saver.restore(tfob_sess, str_ckpt)
			sys.stdout.write('Variables restored.\n\n')
			# -------------------------------------------------------------------------------------------------

			l_rc_loss_pre.append(cur_loss_pre)
			with h5py.File(FLAGS.flag_log_dir + '/learning_curve.h5', 'w') as file:
				file.create_dataset('curve', data = np.array(l_rc_loss_pre, dtype = np.float32))

		# 记录loss和precision表格
		wb = Workbook()
		ws = wb.create_sheet()
		for line in l_rc_loss_pre:
			ws.append(line)
		wb.save('learning_curve.xlsx')
		wb.close()

		tfob_coord.request_stop()
		tfob_coord.join(th_threads)


def main(_):
	b_gpu_enabled = True
	l_devices = device_lib.list_local_devices()
	for i in range(len(l_devices)):
		if l_devices[i].device_type == 'GPU':
			if l_devices[i].memory_limit > 2 * 1024 * 1024 * 1024 :
				b_gpu_enabled = True
				break

	str_last_ckpt = tf.train.latest_checkpoint(FLAGS.flag_log_dir)
	if str_last_ckpt is not None:
		while True:
			sys.stdout.write('Checkpoint "%s" found. Continue last training session?\n' % str_last_ckpt)
			sys.stdout.write('Continue - [c/C]. Restart (all content of log dir will be removed) - [r/R]. Abort - [a/A].\n')
			ans = input().lower()
			if len(ans) == 0:
				continue
			if ans[0] == 'c':
				break
			elif ans[0] == 'r':
				str_last_ckpt = None
				shutil.rmtree(FLAGS.flag_log_dir)
				time.sleep(1)
				break
			elif ans[0] == 'a':
				return

	if os.path.exists(FLAGS.flag_log_dir) == False:
		os.mkdir(FLAGS.flag_log_dir)

	run_training(b_gpu_enabled, str_last_ckpt)
	print('Program is finished.')


if __name__ == '__main__':
    tf.app.run()
