import tensorflow as tf
import numpy as np
import Layerslstm


dataset_path = '处理好的帧数据地址'
NUM_CLASSES = 11
NORM_FRAMES = 6

# 获取数据集
def get_dataset():
	train_data=[]
	train_labels=[]
	validation_data=[]
	validation_labels=[]
	#训练集数据后续按batch读取
	for n in range(320,1600):
		dataX=1
		dataY = 1
		train_data.append(dataX)
		train_labels.append(dataY)
	#读取验证集数据（随机选取6帧）
	for n in range(0,320):
		name=str(n)
		path=dataset_path+name+'.npz'
		ds = np.load(path)
		dataX=ds['images'].astype('float32')
		dataY = ds['labels']
		dataXX=[]
		n_anchor = np.random.randint(0, dataX.shape[1], size=NORM_FRAMES)
		n_anchor.sort()
		dataX=dataX.T
		n=0
		for f in range(NORM_FRAMES):
			dataXX.append(dataX[n_anchor[n]])
			n = n + 1
		arr_frames_rgb = np.array(dataXX)
		arr_frames_rgb=arr_frames_rgb.T
		validation_data.append(arr_frames_rgb)
		validation_labels.append(dataY)

	ds_mean =0
	ds_std =0

	dict_dataset = {}
	dict_dataset['train'] = {
		'train_labels' : train_labels,
		'train_data' : train_data
		}
	dict_dataset['validation'] = {
		'validation_labels' : validation_labels,
		'validation_data' : validation_data
		}
	dict_mean_std = {}
	dict_mean_std['mean'] = {
		'mean' : ds_mean
		}
	dict_mean_std['std'] = {
		'std' : ds_std
		}

	return dict_dataset, dict_mean_std


# 组织每次喂给神经网络的训练和验证数据，一次只喂flag_batch_size大小的数据
def construct_batch_part(dict_mean_std, flag_batch_size):
	shape_data = [57600,NORM_FRAMES]

	# 训练集的placeholder
	tfph_train_data = tf.placeholder(dtype = tf.float32, shape = [flag_batch_size] + shape_data, name = 'ph_train_data')

	# 验证集的placeholder
	tfph_validation_data = tf.placeholder(dtype = tf.float32, shape = [flag_batch_size] + shape_data, name = 'ph_validation_data')

	# mean和std的placeholder
	tfph_mean = tf.placeholder(dtype = tf.float32, shape = shape_data, name = 'ph_mean')
	tfph_std = tf.placeholder(dtype = tf.float32, shape = shape_data, name = 'ph_std')

	# labels的placeholder
	tfph_train_labels = tf.placeholder(dtype = tf.int32, shape = [flag_batch_size], name = 'ph_train_labels')
	tfph_validation_labels = tf.placeholder(dtype = tf.int32, shape = [flag_batch_size], name = 'ph_validation_labels')

	# 数据分割为LSTM的队列
	batch_train_data = tf.reshape(tfph_train_data, [-1,57600,NORM_FRAMES])
	batch_validation_data = tf.reshape(tfph_validation_data, [-1,57600 ,NORM_FRAMES])

	result = {}
	result['batches'] = {
		'batch_train_labels' : tfph_train_labels,
		'batch_validation_labels' : tfph_validation_labels,
		'batch_train_data' : batch_train_data,
		'batch_validation_data': batch_validation_data
		}
	result['input_placeholders'] = {
		'tfph_train_labels' : tfph_train_labels,
		'tfph_validation_labels' : tfph_validation_labels,
		'tfph_train_data' : tfph_train_data,
		'tfph_validation_data': tfph_validation_data,
		'tfph_mean' : tfph_mean,
		'tfph_std' : tfph_std
		}
	return result


# 组织一个flag_batch_size大小的训练数据，与construct_batch_part配套使用
def get_batch_part_train(dict_dataset, dict_mean_std, dict_placeholders, n_index_head, flag_batch_size):
	n_size = np.array(dict_dataset['train']['train_labels']).shape[0]
	assert n_size % flag_batch_size == 0, 'Batch size must be divided extractly.'
	#根据选择折数修改
	if n_index_head>=0:
		n_index_head=n_index_head+320
	n_index_end = n_index_head + flag_batch_size

	# 获取训练集
	lst_volume_rgb = []
	lst_label_rgb = []
	for data_str in range(n_index_head,n_index_end):
	# random clip的读取原始视频固定帧数
		name = str(data_str)
		path = dataset_path + name + '.npz'
		ds = np.load(path)
		dataX = ds['images'].astype('float32')
		dataY = ds['labels']
		arr_ori_frames_rgb = dataX

		# 生成随机帧编号，长度为NORM_FRAMES，表示从arr_ori_frames_rgb和arr_ori_frames_optic中取对应编号帧
		n_frame_num = arr_ori_frames_rgb.shape[1]
		arr_ori_frames_rgb=arr_ori_frames_rgb.T
		n_anchor = np.random.randint(0, n_frame_num,size=NORM_FRAMES)
		n_anchor.sort()
		lst_frames_rgb = []
		n=0
		for f in range(NORM_FRAMES):
			lst_frames_rgb.append(arr_ori_frames_rgb[n_anchor[n]])
			n = n + 1
		arr_frames_rgb = np.array(lst_frames_rgb)
		arr_frames_rgb=arr_frames_rgb.T

		lst_volume_rgb.append(arr_frames_rgb)
		lst_label_rgb.append(dataY)
	arr_volume_rgb = np.array(lst_volume_rgb)
	arr_label_rgb = np.array(lst_label_rgb)


	# 实际只用到train的placeholders
	dict_feeder = {
		dict_placeholders['tfph_train_data'] : arr_volume_rgb,
		dict_placeholders['tfph_train_labels'] : arr_label_rgb
		}

	return dict_feeder


# 组织一个flag_batch_size大小的验证数据， 与construct_batch_part配套使用
def get_batch_part_validation(dict_dataset, dict_mean_std, dict_placeholders, n_index_head, flag_batch_size):
	n_size = np.array(dict_dataset['validation']['validation_labels']).shape[0]
	assert n_size % flag_batch_size == 0, 'Batch size must be divided extractly.'

	n_index_end = n_index_head + flag_batch_size
	if n_index_end > n_size:
		n_index_end = n_size
		n_index_head = n_index_end - flag_batch_size

	# 实际只用到validation的placeholders
	dict_feeder = {
		dict_placeholders['tfph_validation_labels'] : dict_dataset['validation']['validation_labels'][n_index_head:n_index_end],
		dict_placeholders['tfph_validation_data'] : dict_dataset['validation']['validation_data'][n_index_head:n_index_end],
		}

	return dict_feeder


# 普通网络结构
def network(data, labels, tfv_train_phase = None, name = None):
	if name is None:
		name = 'network_normal'
	else:
		name = 'network_normal' + '_' + name

	with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
		l_layers = []
		l_layers.append(data)
		l_layers.append(Layerslstm.lstm_layer(l_layers[-1], 2304, tfv_train_phase, 0.5, name_scope = 'lstm_1'))
		l_layers.append(Layerslstm.fc(tf.squeeze(l_layers[-1][:,:,-1]), NUM_CLASSES, act_last = False, name_scope = 'fc_out'))

	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = l_layers[-1], name = 'softmax_xentropy' + name)
	losses = tf.reduce_mean(xentropy, name = 'losses' + name)
	total_loss = tf.add_n([losses], name = 'total_loss' + name)
	loss_averages = tf.train.ExponentialMovingAverage(0.99, name = 'avg_loss' + name)
	tfop_loss_averages = loss_averages.apply([losses] + [total_loss])
	with tf.control_dependencies([tfop_loss_averages]):
		total_loss = tf.identity(total_loss)
	correct_flags = tf.nn.in_top_k(l_layers[-1], labels, 1, name = 'eval' + name)
	evaluation = tf.cast(correct_flags, tf.int32)

	return total_loss, evaluation


# 压缩网络结构
def network_compressed(data, labels, tfv_train_phase = None, name = None):
	if name is None:
		name = 'network_compressed'
	else:
		name = 'network_compressed' + '_' + name

	with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
		l_layers = []
		l_layers.append(data)
		l_layers.append(Layerslstm.lstm_layer(l_layers[-1],2304, tfv_train_phase, 0.5, True, [15,16,16,15], [8,6,6,8], [96,96,96], [64,64,64], name_scope = 'lstm_1'))
		l_layers.append(Layerslstm.fc(tf.squeeze(l_layers[-1][:, :, -1]), NUM_CLASSES, act_last=False, name_scope='fc_out'))

	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = l_layers[-1], name = 'softmax_xentropy' + name)
	losses = tf.reduce_mean(xentropy, name = 'losses' + name)
	total_loss = tf.add_n([losses], name = 'total_loss' + name)
	loss_averages = tf.train.ExponentialMovingAverage(0.99, name = 'avg_loss' + name)
	tfop_loss_averages = loss_averages.apply([losses] + [total_loss])
	with tf.control_dependencies([tfop_loss_averages]):
		total_loss = tf.identity(total_loss)
	correct_flags = tf.nn.in_top_k(l_layers[-1], labels, 1, name = 'eval' + name)
	evaluation = tf.cast(correct_flags, tf.int32)

	return total_loss, evaluation


# 获取网络结构及输出，flag_tt_network为False或True分别表示：原始或多非线性TT网络
def get_network_output(i, t_data, t_labels, v_data, v_labels, flag_batch_size, flag_compressed_network, tfv_train_phase):
	b_reuse = i > 0

	if flag_compressed_network is False:
		loss_train, eval_train = network(t_data, t_labels, tfv_train_phase, 'ucf11')
		loss_validation, eval_validation = network(v_data, v_labels, tfv_train_phase, 'ucf11')
	else:
		loss_train, eval_train = network_compressed(t_data, t_labels, tfv_train_phase, 'ucf11_com')
		loss_validation, eval_validation = network_compressed(v_data, v_labels, tfv_train_phase, 'ucf11_com')

	return  loss_train, eval_train, loss_validation, eval_validation

