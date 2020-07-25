import numpy as np
import tensorflow as tf


# 全连接层
def fc(input,
	   output_size,
	   weights_initializer = tf.contrib.layers.xavier_initializer(), #tf.initializers.he_normal(),
	   weights_regularizer = None,
	   biases_initializer = tf.zeros_initializer,
	   biases_regularizer = None,
	   tfv_train_phase = None,
	   keep_prob = 0.9,
	   act_last = True,
	   name_scope = None):
	""" 全连接层，带dropout，以relu为激活函数，最后一层输出至softmax者无激活函数
	参数：
		input: 输入张量，2阶 - [batch_size, input_size]
		output_size: 输出维数
		weights_initializer: 权重初始化器
		weights_regularizer: 权重正则化器
		biases_initializer: 偏置项初始化器
		biases_regularizer: 偏置项正则化器
		tfv_train_phase: 是否训练标记
		keep_prob: dropout的保持概率
		act_last: 最后一层是否有激活函数
		name_scope: 本层名称
	"""
	# dropout定义
	if tfv_train_phase is not None:
		dropout_rate = lambda p: (p - 1.0) * tf.to_float(tfv_train_phase) + 1.0
	else:
		dropout_rate = lambda p: p * 0.0 + 1.0

	with tf.variable_scope(name_scope):
		# weights和biases，定义
		input_size = input.get_shape()[-1].value
		tfv_weights = tf.get_variable('var_weights', [input_size, output_size], initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
		output = tf.matmul(input, tfv_weights, name = 'output_mal')
		if biases_initializer is not None:
			tfv_biases = tf.get_variable('var_biases', [output_size], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
			output = tf.add(output, tfv_biases, name = 'output_add')
		
		# 加激活函数和dropout
		if act_last is True:
			if tfv_train_phase is not None:
				output = batch_normalization(output, tfv_train_phase, name_scope = 'batch_norm')
			output = tf.nn.relu(output, name = 'relu')
			output = tf.nn.dropout(output, keep_prob = dropout_rate(keep_prob), name = 'dropout')

	return output


# TT全连接层
def tt_fc(input,
		  output_size,
		  input_modes,
		  output_modes,
		  tt_ranks,
		  weights_initializer = tf.contrib.layers.xavier_initializer(),
		  weights_regularizer = None,
		  biases_initializer = tf.zeros_initializer,
		  biases_regularizer = None,
		  tfv_train_phase = None,
		  keep_prob = 0.9,
		  name_scope = None):
	""" TT全连接层，带dropout，如果是多非线性设计，则中间需要bn、selu和dropout
	参数：
		input: 输入张量，2阶 - [batch_size, input_size]
		output_size: 输出维数
		input_modes: 输入张量维数分解的modes，其积必须等于输入张量的input_size
		output_modes: 输出张量维数分解的modes，其积必须等于输出张量的output_size
		tt_ranks: 预设的TT秩，长度+1后必须等于input_modes或output_modes的长度
		weights_initializer: 权重初始化器
		weights_regularizer: 权重正则化器
		biases_initializer: 偏置项初始化器
		biases_regularizer: 偏置项正则化器
		tfv_train_phase: 是否训练标记
		keep_prob: dropout的保持概率
		name_scope: 本层名称
	"""
	assert input.get_shape()[-1].value == np.prod(input_modes), 'Input modes must be the factors of input tensor.'
	assert output_size == np.prod(output_modes), 'Output modes must be the factors of output tensor.'
	assert len(input_modes) == len(output_modes), 'Modes of input and output must be equal.'
	assert len(tt_ranks) == len(input_modes) - 1, 'The number of TT ranks must be matching to the tensor modes.'

	# dropout定义
	if tfv_train_phase is not None:
		dropout_rate = lambda p: (p - 1.0) * tf.to_float(tfv_train_phase) + 1.0
	else:
		dropout_rate = lambda p: p * 0.0 + 1.0

	with tf.variable_scope(name_scope):
		d = len(input_modes)
		batch_size = input.shape[0].value
		l_tt_ranks = [1] + tt_ranks + [1]

		# 先将input重构为:(batch_size*m_{2}*m_{3}*...*m_{d}, m_{1}*r_{0})，注意r_{0}=1
		cur_inp = tf.reshape(input, [batch_size, input_modes[0], -1])
		cur_inp = tf.transpose(cur_inp, [0, 2, 1])
		cur_inp = tf.reshape(cur_inp, [-1, cur_inp.shape[-1].value])

		# TT weights，定义及缩并
		input_modes.append(1)
		for i in range(d):
			# core的shape为:(r_{k-1}*m_{k}, n_{k}*r_{k})
			var_shape = [l_tt_ranks[i] * input_modes[i], output_modes[i] * l_tt_ranks[i + 1]]
			tfv_weight_core = tf.get_variable('var_weight_core_%d' % (i + 1), var_shape, initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)

			# 计算输入和当前core的乘积，链式，结果shape为:(batch_size*n_{1}*...*n_{k-1}m_{k+1}...*m_{d}, n_{k}*r_{k})
			output = tf.matmul(cur_inp, tfv_weight_core, name = 'output_mal_core_%d' % (i + 1))
			
			# 输出先重构shape为:(batch_size*n_{1}*...*n_{k-1}, m_{k+1}, m_{k+2}...*m_{d}, n_{k}, r_{k})
			output = tf.reshape(output, [batch_size * np.prod(np.array(output_modes[0:i]), dtype = np.int32), input_modes[i + 1], -1, output_modes[i], l_tt_ranks[i + 1]])

			# 调换m_{k+1}和n_{k}，重构输出shape为: (batch_size*n_{1}*...*n_{k-1}n_{k}m_{k+2}...*m_{d}, m_{k+1}*r_{k})
			output = tf.transpose(output, [0, 3, 2, 1, 4])
			output = tf.reshape(output, [-1, output.shape[-2].value * output.shape[-1].value])
			if i != d - 1:
				cur_inp = tf.identity(output)
		output = tf.reshape(tf.squeeze(output), [batch_size, -1])

		# biases，定义
		if biases_initializer is not None:
			tfv_biases = tf.get_variable('var_biases', [output_size], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
			output = tf.add(output, tfv_biases, name = 'output_add')

		# 加激活函数和dropout
		if tfv_train_phase is not None:
			output = batch_normalization(output, tfv_train_phase, name_scope = 'batch_norm')
		output = tf.nn.relu(output, name = 'relu')
		output = tf.nn.dropout(output, keep_prob = dropout_rate(keep_prob), name = 'dropout')
		
	return output


# 卷积层
def conv_2d(input,
			output_chs,
			filter_shape,
			strides = [1, 1],
			filter_initializer = tf.contrib.layers.xavier_initializer(),
			filter_regularizer = None,
			biases_initializer = tf.zeros_initializer,
			biases_regularizer = None,
			tfv_train_phase = None,
			name_scope = None):
	""" 卷积层，普通2D卷积，以relu为激活函数
	参数：
		input: 输入张量，4阶 - [batch_size, input_height, input_width, input_chs]，其中，input_height, input_width分别是输入图的高、宽
		output_chs: 输出张量的通道数
		filter_shape: 过滤器的尺寸，[h,w]([高、宽])
		strides: 过滤器扫描步长
		filter_initializer: 过滤器初始化器
		filter_regularizer: 过滤器正则化器
		biases_initializer: 偏置项初始化器
		biases_regularizer: 偏置项正则化器
		tfv_train_phase: 是否训练标记
		name_scope: 本层名称
	"""
	with tf.variable_scope(name_scope):
		# filters，定义
		input_chs = input.get_shape()[-1].value
		tfv_filter = tf.get_variable('var_filter', filter_shape + [input_chs, output_chs], initializer = filter_initializer, regularizer = filter_regularizer, trainable = True)

		# 卷积及偏置
		output = tf.nn.conv2d(input, tfv_filter, [1] + strides + [1], 'SAME', name = 'output_conv')
		if biases_initializer is not None:
			tfv_biases = tf.get_variable('var_biases', [output_chs], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
			output = tf.add(output, tfv_biases, name = 'output')

		# 加激活函数
		if tfv_train_phase is not None:
			output = batch_normalization(output, tfv_train_phase, name_scope = 'batch_norm')
		output = tf.nn.relu(output, name = 'relu')
	
	return output


# TT卷积层
def tt_conv_2d(input,
			   output_chs,
			   filter_shape,
			   input_ch_modes,
			   output_ch_modes,
			   tt_ranks,
			   strides = [1, 1],
			   filter_initializer = tf.contrib.layers.xavier_initializer(),
			   filter_regularizer = None,
			   biases_initializer = tf.zeros_initializer,
			   biases_regularizer = None,
			   tfv_train_phase = None,
			   name_scope = None):
	""" TT卷积层，以relu为激活函数，如果是多非线性设计，则中间需要bn和selu
	参数：
		input: 输入张量，4阶 - [batch_size, input_height, input_width, input_chs]，其中，input_height, input_width分别是输入图的高、宽
		output_chs: 输出张量的通道数
		filter_shape: 过滤器的尺寸，[h,w]([高、宽])
		input_ch_modes: 输入张量通道数分解的modes，其积必须等于输入张量的通道数input_chs
		output_ch_modes: 输出张量通道数分解的modes，其积必须等于输出张量的通道数output_chs
		tt_ranks: 预设的TT秩，长度+1后等于input_ch_modes或output_ch_modes的长度
		strides: 过滤器扫描步长
		filter_initializer: 过滤器初始化器
		filter_regularizer: 过滤器正则化器
		biases_initializer: 偏置项初始化器
		biases_regularizer: 偏置项正则化器
		tfv_train_phase: 是否训练标记
		name_scope: 本层名称
	"""
	assert input.get_shape()[-1].value == np.prod(input_ch_modes), 'Input modes must be the factors of the value of input channels.'
	assert output_chs == np.prod(output_ch_modes), 'Output modes must be the factors of the value of output channels.'
	assert len(input_ch_modes) == len(output_ch_modes), 'Modes of input and output channels must be equal.'
	assert len(tt_ranks) == len(input_ch_modes) - 1 , 'The number of TT ranks must be matching to the tensor modes.'

	with tf.variable_scope(name_scope):
		# -----recover模式TT卷积，开始线-----

		# 1x1卷积的阶数比正常卷积少1
		if np.prod(filter_shape) == 1:
			d = len(input_ch_modes)
			input_modes = input_ch_modes
			output_modes = output_ch_modes
			l_tt_ranks = [1] + tt_ranks + [1]
		else:
			d = len(input_ch_modes) + 1
			input_modes = [filter_shape[0]] + input_ch_modes
			output_modes = [filter_shape[-1]] + output_ch_modes
			l_tt_ranks = [1] + tt_ranks + [np.min(tt_ranks)] + [1]

		# TT filters，定义
		l_tt_kernels = []
		for i in range(d):
			var_shape = [l_tt_ranks[i] * input_modes[i], output_modes[i] * l_tt_ranks[i + 1]]
			tfv_conv_core = tf.get_variable('var_conv_core_%d' % (i + 1), var_shape, initializer = filter_initializer, regularizer = filter_regularizer, trainable = True)
			l_tt_kernels.append(tfv_conv_core)

		# TT filters recover，缩并，shape为(l*l*c_1*s_1*c_2*s_2*...*c_d*s_d)
		filter = tf.reshape(l_tt_kernels[0], [-1, l_tt_ranks[1]])
		recover_shape = []
		for i in range(d - 1):
			next_core = tf.reshape(l_tt_kernels[i + 1], [l_tt_ranks[i + 1], -1])
			filter = tf.matmul(filter, next_core)
			recover_shape += [input_modes[i], output_modes[i]]
			if i != d - 2:
				filter = tf.reshape(filter, [-1, l_tt_ranks[i + 2]])
		recover_shape += [input_modes[-1], output_modes[-1]]
		if np.prod(filter_shape) == 1:
			recover_shape = [1, 1] + recover_shape
			d += 1
		filter = tf.reshape(filter, recover_shape)

		# TT filters recover，还原，shape为(l^2*c_1*c_2*...*c_d*s_1*s_2*...*s_d)
		inch_orders = []
		outch_orders = []
		for i in range(1, d):
			inch_orders.append(2 * i)
			outch_orders.append(2 * i + 1)
		transpose_perm = [0, 1] + inch_orders + outch_orders
		filter = tf.transpose(filter, transpose_perm)
		filter = tf.reshape(filter, [filter_shape[0], filter_shape[-1], np.prod(input_ch_modes), np.prod(output_ch_modes)])

		# 卷积
		output = tf.nn.conv2d(input, filter, [1] + strides + [1], 'SAME', name = 'output_conv')
		# -----recover模式TT卷积，结束线-----


		# biases，定义
		if biases_initializer is not None:
			tfv_biases = tf.get_variable('var_biases', [output_chs], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
			output = tf.add(output, tfv_biases, name = 'output')

		# 加激活函数
		if tfv_train_phase is not None:
			output = batch_normalization(output, tfv_train_phase, name_scope = 'batch_norm')
		output = tf.nn.relu(output, name = 'relu')

	return output


# 2D最大池化层
def maxpool_2d(input,
			   ksize,
			   stride,
			   name_scope = None):
	""" 2D最大池化层，stride一般取2，ksize尺寸在各个方向一般比stride大1
	参数：
		input: 输入张量，4维 - [batch_size, input_height, input_width, input_chs]
		ksize: 池化的扫描窗口尺寸，2维 - [高、宽]
		stride: 池化扫描窗口移动步长，2维，分别代表池化扫描窗口在输入image的高、宽两个方向上的移动步长
		name_scope: 本层名称
	"""
	with tf.variable_scope(name_scope):
		output = tf.nn.max_pool(input, [1] + ksize + [1], [1] + stride + [1], 'SAME', name = 'max_pool_output')
	return output


# 2D平均池化层
def avgpool_2d(input,
			   ksize,
			   stride,
			   name_scope = None):
	""" 2D平均池化层，将若干通道的特征图化为向量，ksize和stride的尺寸一般情况下相等，大小取决于当前特征图尺寸
	参数：
		input: 输入张量，4维 - [batch_size, input_height, input_width, input_chs]
		ksize: 池化的扫描窗口尺寸，2维 - [高、宽]
		stride: 池化扫描窗口移动步长，2维，分别代表池化扫描窗口在输入image的高、宽两个方向上的移动步长
		name_scope: 本层名称
	"""
	with tf.variable_scope(name_scope):
		output = tf.nn.avg_pool(input, [1] + ksize + [1], [1] + stride + [1], 'SAME', name = 'avg_pool_output')
		sz = np.prod(output.get_shape().as_list()[1:])
		output = tf.reshape(output, [-1, sz])
	return output


# 局部响应标准化
def local_response_normalization(input,
								 n,
								 k,
								 alpha,
								 beta,
								 name_scope = None):
	""" 局部响应标准化，一般用于AlexNet
	参数：
		input: 输入数据
		n: 局部范围的半径，Alex论文中是大小为5的直径，这里取2
		k: 偏置，Alex论文中取2
		alpha: 和的系数，Alex论文中取10^-4
		beta: 和的次方，Alex论文中取0.75
		name_scope: 本层名称
	"""
	with tf.variable_scope(name_scope):
		output = tf.nn.lrn(input, n, k, alpha, beta, name = 'lrn_output')
	return output


# 批标准化
def batch_normalization(input,
						tfv_train_phase,
						ema_decay = 0.99,
                        eps = 1e-3,
                        use_scale = True,
                        use_shift = True,
                        name_scope = None):
	""" 批标准化，一般用于ReLU之前，不再使用SeLU
	参数：
		input: 输入数据
		tfv_train_phase: 是否训练标记
		ema_decay: 滑动平均模型衰减参数
		eps: 标准差分母的防除0偏差
		use_scale: 是否使用scale参数(gamma)
		use_shift: 是否使用shift参数(beta)
		name_scope: 本层名称
	"""
	reuse = tf.get_variable_scope().reuse
	with tf.variable_scope(name_scope):
		shape = input.get_shape().as_list()
		assert len(shape) in [2, 4]
		n_out = shape[-1]

		# 求均值和方差
		if len(shape) == 2:
			batch_mean, batch_variance = tf.nn.moments(input, [0], name = 'moments')
		else:
			batch_mean, batch_variance = tf.nn.moments(input, [0, 1, 2], name = 'moments')
		ema = tf.train.ExponentialMovingAverage(decay = ema_decay, zero_debias = True)
		if not reuse or reuse == tf.AUTO_REUSE:
			def mean_variance_with_update():
				with tf.control_dependencies([ema.apply([batch_mean, batch_variance])]):
					return (tf.identity(batch_mean), tf.identity(batch_variance))
			mean, variance = tf.cond(tfv_train_phase, mean_variance_with_update, lambda: (ema.average(batch_mean), ema.average(batch_variance)))
		else:
			vars = tf.get_variable_scope().global_variables()
			transform = lambda s: '/'.join(s.split('/')[-5:])
			mean_name = transform(ema.average_name(batch_mean))
			variance_name = transform(ema.average_name(batch_variance))
			existed = {}
			for v in vars:
				if (transform(v.op.name) == mean_name):
					existed['mean'] = v
				if (transform(v.op.name) == variance_name):
					existed['variance'] = v
			mean, variance = tf.cond(tfv_train_phase, lambda: (batch_mean, batch_variance), lambda: (existed['mean'], existed['variance']))

		# 归一化
		std = tf.sqrt(variance + eps, name = 'std')
		output = (input - mean) / std

		# 乘以gamma
		if use_scale:
			weights = tf.get_variable('weights', [n_out], initializer = tf.ones_initializer, trainable = True)
			output = tf.multiply(output, weights)

		# 加上beta
		if use_shift:
			biases = tf.get_variable('biases', [n_out], initializer = tf.zeros_initializer, trainable = True)
			output = tf.add(output, biases)

	return output


# LSTM单元
def lstm_cell(input_x,
			  input_y,
			  input_c,
			  output_dim,
			  weights_initializer = tf.glorot_uniform_initializer,
			  weights_regularizer = None,
			  biases_initializer = tf.ones_initializer,
			  biases_regularizer = None,
			  tfv_train_phase = None,
			  keep_prob = 0.9,
			  name_scope = None):
	""" LSTM单元，以"LSTM: A Search Space Odyssey"为准
	参数：
		input_x: 前一层输入向量，或2阶张量 - [batch_size, input_dim]
		input_y: 上一时刻输入向量，或2阶张量 - [batch_size, output_dim]，若为初始状态则输入全0
		input_c: 上一时刻输入状态向量，或2阶张量 - [batch_size, output_dim]，若为初始状态则输入全0
		output_dim: 输出向量维度
		weights_initializer: 权重初始化器
		weights_regularizer: 权重正则化器
		biases_initializer: 偏置项初始化器
		biases_regularizer: 偏置项正则化器
		tfv_train_phase: 是否训练标记
		keep_prob: dropout的保持概率
		name_scope: 本层名称
	"""
	# dropout定义
	if tfv_train_phase is not None:
		dropout_rate = lambda p: (p - 1.0) * tf.to_float(tfv_train_phase) + 1.0
	else:
		dropout_rate = lambda p: p * 0.0 + 1.0

	with tf.variable_scope(name_scope):
		input_dim = input_x.shape[-1].value

		# 定义遗忘门权重及偏置
		W_f = tf.get_variable('var_weight_forget', [input_dim, output_dim], initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
		R_f = tf.get_variable('var_recurrent_forget', [output_dim, output_dim], initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
		p_f = tf.get_variable('var_peephole_forget', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
		b_f = tf.get_variable('var_bias_forget', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)

		# 定义输入门权重及偏置
		W_i = tf.get_variable('var_weight_input', [input_dim, output_dim], initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
		R_i = tf.get_variable('var_recurrent_input', [output_dim, output_dim], initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
		p_i = tf.get_variable('var_peephole_input', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
		b_i = tf.get_variable('var_bias_input', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)

		# 定义状态门权重及偏置
		W_z = tf.get_variable('var_weight_state', [input_dim, output_dim], initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
		R_z = tf.get_variable('var_recurrent_state', [output_dim, output_dim], initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
		p_z = tf.get_variable('var_peephole_state', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
		b_z = tf.get_variable('var_bias_state', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
		
		# 定义输出门权重及偏置
		W_o = tf.get_variable('var_weight_output', [input_dim, output_dim], initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
		R_o = tf.get_variable('var_recurrent_output', [output_dim, output_dim], initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
		p_o = tf.get_variable('var_peephole_output', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
		b_o = tf.get_variable('var_bias_output', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)

		# 输入dropout
		input_x_drop = tf.nn.dropout(input_x, keep_prob = dropout_rate(keep_prob), name = 'dropout')

		# 遗忘门计算
		output_f = tf.nn.sigmoid(tf.matmul(input_x_drop, W_f) + tf.matmul(input_y, R_f) + input_c * p_f + b_f)
		
		# 输入门计算
		output_i = tf.nn.sigmoid(tf.matmul(input_x_drop, W_i) + tf.matmul(input_y, R_i) + input_c * p_i + b_i)

		# 状态门计算
		output_z = tf.nn.tanh(tf.matmul(input_x_drop, W_z) + tf.matmul(input_y, R_z) + b_z)

		# 当前状态
		output_c = output_f * input_c + output_i * output_z

		# 输出门计算
		output_o = tf.nn.sigmoid(tf.matmul(input_x_drop, W_o) + tf.matmul(input_y, R_o) + output_c * p_o + b_o)

		# 当前输出
		output_y = output_o * tf.nn.tanh(output_c)

	return output_y, output_c


# TT LSTM单元
def tt_lstm_cell(input_x,
				 input_y,
				 input_c,
				 output_dim,
				 input_modes,
				 output_modes,
				 tt_ranks_W,
				 tt_ranks_R,
				 weights_initializer = tf.glorot_uniform_initializer,
				 weights_regularizer = None,
				 biases_initializer = tf.ones_initializer,
				 biases_regularizer = None,
				 tfv_train_phase = None,
				 keep_prob = 0.9,
				 name_scope = None):
	""" TT LSTM单元，将lstm_cell中的权重矩阵(W和R)改为TT形式
	参数：
		input_x: 前一层输入向量，或2阶张量 - [batch_size, input_dim]
		input_y: 上一时刻输入向量，或2阶张量 - [batch_size, output_dim]，若为初始状态则输入全0
		input_c: 上一时刻输入状态向量，或2阶张量 - [batch_size, output_dim]，若为初始状态则输入全0
		output_dim: 输出向量维度
		input_modes: 前一层输入向量维数分解的modes，其积必须等于input_dim
		output_modes: 输出向量维数分解的modes，其积必须等于output_dim
		tt_ranks_W: W权重的预设TT秩，长度+1后必须等于input_modes或output_modes的长度
		tt_ranks_R: R权重的预设TT秩，长度+1后必须等于output_modes的长度
		weights_initializer: 权重初始化器
		weights_regularizer: 权重正则化器
		biases_initializer: 偏置项初始化器
		biases_regularizer: 偏置项正则化器
		tfv_train_phase: 是否训练标记
		keep_prob: dropout的保持概率
		name_scope: 本层名称
	"""
	assert input_x.get_shape()[-1].value == np.prod(input_modes), 'Input modes must be the factors of input tensor.'
	assert output_dim == np.prod(output_modes), 'Output modes must be the factors of output tensor.'
	assert len(input_modes) == len(output_modes), 'Modes of input and output must be equal.'
	assert len(tt_ranks_W) == len(input_modes) - 1 and len(tt_ranks_R) == len(input_modes) - 1, 'The number of TT ranks must be matching to the tensor modes.'

	# dropout定义
	if tfv_train_phase is not None:
		dropout_rate = lambda p: (p - 1.0) * tf.to_float(tfv_train_phase) + 1.0
	else:
		dropout_rate = lambda p: p * 0.0 + 1.0

	with tf.variable_scope(name_scope):
		d = len(input_modes)
		batch_size = input_x.shape[0].value
		l_tt_ranks_W = [1] + tt_ranks_W + [1]
		l_tt_ranks_R = [1] + tt_ranks_R + [1]
		
		# 定义遗忘门权重及偏置
		l_W_f_cores = []
		for i in range(d):
			# core的shape为:(r_{k-1}*m_{k}, n_{k}*r_{k})
			var_shape = [l_tt_ranks_W[i] * input_modes[i], output_modes[i] * l_tt_ranks_W[i + 1]]
			W_f_core = tf.get_variable('var_W_forget_core_%d' % (i + 1), var_shape, initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
			l_W_f_cores.append(W_f_core)
		l_R_f_cores = []
		for i in range(d):
			# core的shape为:(r_{k-1}*m_{k}, n_{k}*r_{k})
			var_shape = [l_tt_ranks_R[i] * output_modes[i], output_modes[i] * l_tt_ranks_R[i + 1]]
			R_f_core = tf.get_variable('var_R_forget_core_%d' % (i + 1), var_shape, initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
			l_R_f_cores.append(R_f_core)
		p_f = tf.get_variable('var_peephole_forget', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
		b_f = tf.get_variable('var_bias_forget', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)

		# 定义输入门权重及偏置
		l_W_i_cores = []
		for i in range(d):
			# core的shape为:(r_{k-1}*m_{k}, n_{k}*r_{k})
			var_shape = [l_tt_ranks_W[i] * input_modes[i], output_modes[i] * l_tt_ranks_W[i + 1]]
			W_i_core = tf.get_variable('var_W_input_core_%d' % (i + 1), var_shape, initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
			l_W_i_cores.append(W_i_core)
		l_R_i_cores = []
		for i in range(d):
			# core的shape为:(r_{k-1}*m_{k}, n_{k}*r_{k})
			var_shape = [l_tt_ranks_R[i] * output_modes[i], output_modes[i] * l_tt_ranks_R[i + 1]]
			R_i_core = tf.get_variable('var_R_input_core_%d' % (i + 1), var_shape, initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
			l_R_i_cores.append(R_i_core)
		p_i = tf.get_variable('var_peephole_input', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
		b_i = tf.get_variable('var_bias_input', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)

		# 定义状态门权重及偏置
		l_W_z_cores = []
		for i in range(d):
			# core的shape为:(r_{k-1}*m_{k}, n_{k}*r_{k})
			var_shape = [l_tt_ranks_W[i] * input_modes[i], output_modes[i] * l_tt_ranks_W[i + 1]]
			W_z_core = tf.get_variable('var_W_state_core_%d' % (i + 1), var_shape, initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
			l_W_z_cores.append(W_z_core)
		l_R_z_cores = []
		for i in range(d):
			# core的shape为:(r_{k-1}*m_{k}, n_{k}*r_{k})
			var_shape = [l_tt_ranks_R[i] * output_modes[i], output_modes[i] * l_tt_ranks_R[i + 1]]
			R_z_core = tf.get_variable('var_R_state_core_%d' % (i + 1), var_shape, initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
			l_R_z_cores.append(R_z_core)
		p_z = tf.get_variable('var_peephole_state', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
		b_z = tf.get_variable('var_bias_state', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)

		# 定义输出门权重及偏置
		l_W_o_cores = []
		for i in range(d):
			# core的shape为:(r_{k-1}*m_{k}, n_{k}*r_{k})
			var_shape = [l_tt_ranks_W[i] * input_modes[i], output_modes[i] * l_tt_ranks_W[i + 1]]
			W_o_core = tf.get_variable('var_W_output_core_%d' % (i + 1), var_shape, initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
			l_W_o_cores.append(W_o_core)
		l_R_o_cores = []
		for i in range(d):
			# core的shape为:(r_{k-1}*m_{k}, n_{k}*r_{k})
			var_shape = [l_tt_ranks_R[i] * output_modes[i], output_modes[i] * l_tt_ranks_R[i + 1]]
			R_o_core = tf.get_variable('var_R_output_core_%d' % (i + 1), var_shape, initializer = weights_initializer, regularizer = weights_regularizer, trainable = True)
			l_R_o_cores.append(R_o_core)
		p_o = tf.get_variable('var_peephole_output', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)
		b_o = tf.get_variable('var_bias_output', [output_dim], initializer = biases_initializer, regularizer = biases_regularizer, trainable = True)

		# 输入dropout
		input_x_drop = tf.nn.dropout(input_x, keep_prob = dropout_rate(keep_prob), name = 'dropout')

		# 门计算定义
		def tt_gated_matmul(input, cores, input_modes, output_modes, tt_ranks, d, batch_size, name):
			cur_inp = tf.reshape(input, [batch_size, input_modes[0], -1])
			cur_inp = tf.transpose(cur_inp, [0, 2, 1])
			cur_inp = tf.reshape(cur_inp, [-1, cur_inp.shape[-1].value])
			for i in range(d):
				output = tf.matmul(cur_inp, cores[i], name = name + '_mal_core_%d' % (i + 1))
				if i == d - 1:
					output = tf.reshape(output, [batch_size * np.prod(np.array(output_modes[0:i]), dtype = np.int32), 1, -1, output_modes[i], tt_ranks[i + 1]])
				else:
					output = tf.reshape(output, [batch_size * np.prod(np.array(output_modes[0:i]), dtype = np.int32), input_modes[i + 1], -1, output_modes[i], tt_ranks[i + 1]])
				output = tf.transpose(output, [0, 3, 2, 1, 4])
				output = tf.reshape(output, [-1, output.shape[-2].value * output.shape[-1].value])
				if i != d - 1:
					cur_inp = tf.identity(output)
			output = tf.reshape(tf.squeeze(output), [batch_size, -1])
			return output

		
		# 遗忘门计算
		output_W_f = tt_gated_matmul(input_x_drop, l_W_f_cores, input_modes, output_modes, l_tt_ranks_W, d, batch_size, 'W_f')
		output_R_f = tt_gated_matmul(input_y, l_R_f_cores, output_modes, output_modes, l_tt_ranks_R, d, batch_size, 'R_f')
		output_f = tf.nn.sigmoid(output_W_f + output_R_f + input_c * p_f + b_f)

		# 输入门计算
		output_W_i = tt_gated_matmul(input_x_drop, l_W_i_cores, input_modes, output_modes, l_tt_ranks_W, d, batch_size, 'W_i')
		output_R_i = tt_gated_matmul(input_y, l_R_i_cores, output_modes, output_modes, l_tt_ranks_R, d, batch_size, 'R_i')
		output_i = tf.nn.sigmoid(output_W_i + output_R_i + input_c * p_i + b_i)

		# 状态门计算
		output_W_z = tt_gated_matmul(input_x_drop, l_W_z_cores, input_modes, output_modes, l_tt_ranks_W, d, batch_size,'W_z')
		output_R_z = tt_gated_matmul(input_y, l_R_z_cores, output_modes, output_modes, l_tt_ranks_R, d, batch_size,  'R_z')
		output_z = tf.nn.tanh(output_W_z + output_R_z + b_z)

		# 当前状态
		output_c = output_f * input_c + output_i * output_z

		# 输出门计算
		output_W_o = tt_gated_matmul(input_x_drop, l_W_o_cores, input_modes, output_modes, l_tt_ranks_W, d, batch_size,  'W_o')
		output_R_o = tt_gated_matmul(input_y, l_R_o_cores, output_modes, output_modes, l_tt_ranks_R, d, batch_size,  'R_o')
		output_o = tf.nn.sigmoid(output_W_o + output_R_o + output_c * p_o + b_o)

		# 当前输出
		output_y = output_o * tf.nn.tanh(output_c)

	return output_y, output_c


def ht_lstm_cell(input_x,
				 input_y,
				 input_c,
				 output_dim,
				 input_modes,
				 output_modes,
				 ht_ranks_W,
				 ht_ranks_R,
				 weights_initializer=tf.glorot_uniform_initializer,
				 weights_regularizer=None,
				 biases_initializer=tf.ones_initializer,
				 biases_regularizer=None,
				 tfv_train_phase=None,
				 keep_prob=0.9,
				 name_scope=None):
	""" HT LSTM单元，将lstm_cell中的权重矩阵(W和R)改为HT形式
	参数：
		input_x: 前一层输入向量，或2阶张量 - [batch_size, input_dim]
		input_y: 上一时刻输入向量，或2阶张量 - [batch_size, output_dim]，若为初始状态则输入全0
		input_c: 上一时刻输入状态向量，或2阶张量 - [batch_size, output_dim]，若为初始状态则输入全0
		output_dim: 输出向量维度
		input_modes: 前一层输入向量维数分解的modes，其积必须等于input_dim
		output_modes: 输出向量维数分解的modes，其积必须等于output_dim
		ht_ranks_W: W权重的预设hT秩
		ht_ranks_R: R权重的预设hT秩
		weights_initializer: 权重初始化器
		weights_regularizer: 权重正则化器
		biases_initializer: 偏置项初始化器
		biases_regularizer: 偏置项正则化器
		tfv_train_phase: 是否训练标记
		keep_prob: dropout的保持概率
		name_scope: 本层名称
	"""
	assert input_x.get_shape()[-1].value == np.prod(input_modes), 'Input modes must be the factors of input tensor.'
	assert output_dim == np.prod(output_modes), 'Output modes must be the factors of output tensor.'
	assert len(input_modes) == len(output_modes), 'Modes of input and output must be equal.'
	assert len(ht_ranks_W) == len(input_modes) - 1 and len(ht_ranks_R) == len(
		input_modes) - 1, 'The number of TT ranks must be matching to the tensor modes.'

	# dropout定义
	if tfv_train_phase is not None:
		dropout_rate = lambda p: (p - 1.0) * tf.to_float(tfv_train_phase) + 1.0
	else:
		dropout_rate = lambda p: p * 0.0 + 1.0

	# 输入dropout
	input_x_drop = tf.nn.dropout(input_x, keep_prob=dropout_rate(keep_prob), name='dropout')

	# 克罗内克积
	def kron(a, b):

		[am, an] = a.shape.as_list()
		[bm, bn] = b.shape.as_list()
		a = tf.reshape(a, [am, 1, an, 1])
		b = tf.reshape(b, [1, bm, 1, bn])
		K = tf.reshape(tf.multiply(a, b), [am * bm, an * bn])

		return K

	# define var
	def get_var_wrap(name,
					 shape,
					 initializer,
					 regularizer,
					 trainable):

		return tf.get_variable(name,
							   shape=shape,
							   initializer=initializer,
							   regularizer=regularizer,
							   trainable=trainable)

	def ht_gated_matmul(inp,
			  inp_modes,  # n
			  out_modes,
			  matin_ranks,  # [r1,r2,r12]
			  matout_ranks,  # [r3,r4,r34]
			  cores_initializer=weights_initializer,
			  cores_regularizer=weights_regularizer,
			  trainable=True,
			  name_scope=None):

		with tf.variable_scope(name_scope):
			dimin = len(inp_modes)
			dimout = len(out_modes)
			matin_cores = []
			matout_cores = []
			batch_size = inp.shape[0].value
			inp = tf.reshape(inp, [batch_size, -1])

			cinit = cores_initializer
			creg = cores_regularizer
			# U1与U2克罗内克积
			for i in range(dimin // 2):
				matin_cores.append(get_var_wrap('mat_corein_%d' % (i + 1),
												shape=[inp_modes[i] * out_modes[i], matin_ranks[i]],
												initializer=cinit,
												regularizer=creg,
												trainable=trainable))

			matin_kron = kron(matin_cores[0], matin_cores[1])
			# U3与U4的克罗内克积
			for i in range(2, 2 + dimout // 2):
				matout_cores.append(get_var_wrap('mat_coreout_%d' % (i + 1),
												 shape=[inp_modes[i] * out_modes[i], matout_ranks[i - 2]],
												 initializer=cinit,
												 regularizer=creg,
												 trainable=trainable))

			matout_kron = kron(matout_cores[0], matout_cores[1])
			# [r12,r34]
			blast = get_var_wrap('mat_coreinlast',
								 shape=[matin_ranks[-1], matout_ranks[-1]],
								 initializer=cinit,
								 regularizer=creg,
								 trainable=trainable)

			# [r1r2,r12]
			tb12 = get_var_wrap('tb12',
								shape=[matin_ranks[0] * matin_ranks[1], matin_ranks[2]],
								initializer=cinit,
								regularizer=creg,
								trainable=trainable)
			# [r3r4,r34]
			tb34 = get_var_wrap('tb34',
								shape=[matout_ranks[0] * matout_ranks[1], matout_ranks[2]],
								initializer=cinit,
								regularizer=creg,
								trainable=trainable)
			# 合并过程
			matin_kron = tf.matmul(matin_kron, tb12)
			matin_kron = tf.matmul(matin_kron, blast)

			matout_kron = tf.matmul(matout_kron, tb34)
			matout_kron = tf.reshape(matout_kron,
									 [inp_modes[2], out_modes[2], inp_modes[3], out_modes[3], matout_ranks[2]])
			matout_kron = tf.transpose(matout_kron, [0, 2, 1, 3, 4])
			matout_kron = tf.reshape(matout_kron,
									 [inp_modes[2] * inp_modes[3] * out_modes[2] * out_modes[3], matout_ranks[2]])
			matout_kron = tf.transpose(matout_kron, [1, 0])

			# [n1n2m1m2*1]
			mat_cores1 = tf.reshape(matin_kron,
									[inp_modes[0], out_modes[0], inp_modes[1], out_modes[1], matout_ranks[-1]])
			mat_cores1 = tf.transpose(mat_cores1, [0, 2, 1, 3, 4])
			mat_cores1 = tf.reshape(mat_cores1,
									[inp_modes[0] * inp_modes[1], out_modes[0] * out_modes[1], matout_ranks[-1]])
			mat_cores1 = tf.reshape(mat_cores1, [-1, matout_ranks[-1]])

			mat_cores = tf.matmul(mat_cores1, matout_kron)
			mat_cores = tf.reshape(mat_cores,
								   [inp_modes[0] * inp_modes[1], out_modes[0] * out_modes[1],
									inp_modes[2] * inp_modes[3],
									out_modes[2] * out_modes[3]])
			mat_cores = tf.transpose(mat_cores, [0, 2, 1, 3])
			mat_cores = tf.reshape(mat_cores, [inp_modes[0] * inp_modes[1] * inp_modes[2] * inp_modes[3],
											   out_modes[0] * out_modes[1] * out_modes[2] * out_modes[3]])

			out = tf.matmul(inp, mat_cores)

		return out

	p_f = tf.get_variable('var_peephole_forget', [output_dim], initializer=biases_initializer,regularizer=biases_regularizer, trainable=True)
	b_f = tf.get_variable('var_bias_forget', [output_dim], initializer=biases_initializer,regularizer=biases_regularizer, trainable=True)

	p_i = tf.get_variable('var_peephole_input', [output_dim], initializer=biases_initializer,regularizer=biases_regularizer, trainable=True)
	b_i = tf.get_variable('var_bias_input', [output_dim], initializer=biases_initializer,regularizer=biases_regularizer, trainable=True)

	p_z = tf.get_variable('var_peephole_state', [output_dim], initializer=biases_initializer,regularizer=biases_regularizer, trainable=True)
	b_z = tf.get_variable('var_bias_state', [output_dim], initializer=biases_initializer,regularizer=biases_regularizer, trainable=True)

	p_o = tf.get_variable('var_peephole_output', [output_dim], initializer=biases_initializer,regularizer=biases_regularizer, trainable=True)
	b_o = tf.get_variable('var_bias_output', [output_dim], initializer=biases_initializer,regularizer=biases_regularizer, trainable=True)

	# 遗忘门计算
	output_W_f = ht_gated_matmul(input_x_drop, input_modes, output_modes, ht_ranks_W, ht_ranks_W, name_scope='W_f')
	output_R_f = ht_gated_matmul(input_y, output_modes, output_modes, ht_ranks_R, ht_ranks_R, name_scope= 'R_f')
	output_f = tf.nn.sigmoid(output_W_f + output_R_f + input_c * p_f + b_f)

	# 输入门计算
	output_W_i = ht_gated_matmul(input_x_drop,input_modes, output_modes, ht_ranks_W,ht_ranks_W, name_scope='W_i')
	output_R_i = ht_gated_matmul(input_y, output_modes, output_modes, ht_ranks_R, ht_ranks_R, name_scope='R_i')
	output_i = tf.nn.sigmoid(output_W_i + output_R_i + input_c * p_i + b_i)

	# 状态门计算
	output_W_z = ht_gated_matmul(input_x_drop, input_modes, output_modes, ht_ranks_W,ht_ranks_W, name_scope='W_z')
	output_R_z = ht_gated_matmul(input_y, output_modes, output_modes, ht_ranks_R, ht_ranks_R,  name_scope='R_z')
	output_z = tf.nn.tanh(output_W_z + output_R_z + b_z)

	# 当前状态
	output_c = output_f * input_c + output_i * output_z

	# 输出门计算
	output_W_o = ht_gated_matmul(input_x_drop, input_modes, output_modes, ht_ranks_W, ht_ranks_W,name_scope='W_o')
	output_R_o = ht_gated_matmul(input_y, output_modes, output_modes, ht_ranks_R, ht_ranks_R, name_scope='R_o')
	output_o = tf.nn.sigmoid(output_W_o + output_R_o + output_c * p_o + b_o)

	# 当前输出
	output_y = output_o * tf.nn.tanh(output_c)

	return output_y, output_c


# LSTM层
def lstm_layer(input_seq,
			   hidden_dim,
			   tfv_train_phase = None,
			   keep_prob = 0.9,
			   com_network = False,
			   input_modes = None,
			   output_modes = None,
			   com_ranks_W = None,
			   com_ranks_R = None,
			   name_scope = None):
	""" LSTM层，由hidden_dim个LSTM单元构建一个LSTM层
	参数：
		input_seq: 输入序列，一般为3阶张量 - [batch_size, input_dim, num_seq]，num_seq是本层LSTM单元数量
		hidden_dim: 隐藏层维度，即每个LSTM单元中权重矩阵的输出维度(输入维度为input_dim)
		tfv_train_phase: 是否训练标记
		keep_prob: dropout的保持概率
		com_network: 是否选择TT形式的LSTM单元，为真表示选择tt_lstm_cell
		input_modes: 输入向量维数分解的modes，其积必须等于输入向量的input_dim，com_network为真则起效
		output_modes: 输出向量维数分解的modes，其积必须等于hidden_dim，com_network为真则起效
		com_ranks_W: TT形式的W权重矩阵的秩
		com_ranks_R: TT形式的R权重矩阵的秩
		name_scope: 本层名称
	"""
	with tf.variable_scope(name_scope):
		batch_size = input_seq.shape[0].value
		input_dim = input_seq.shape[1].value
		num_seq = input_seq.shape[-1].value

		# 初始状态，包括c和y
		init_c = tf.zeros([batch_size, hidden_dim])
		init_y = tf.zeros([batch_size, hidden_dim])

		# 令数据通过num_seq个LSTM单元，每个单元中的权重共享
		l_outputs = []
		cur_c = init_c
		cur_y = init_y
		for i in range(num_seq):
			cur_x = tf.gather(input_seq, i, axis = -1)
			if com_network is True:
				cur_y, cur_c = tt_lstm_cell(cur_x, cur_y, cur_c, hidden_dim, input_modes, output_modes, com_ranks_W, com_ranks_R, tfv_train_phase = tfv_train_phase, keep_prob = keep_prob, name_scope = 'tt_lstm_cell')
				#cur_y, cur_c = ht_lstm_cell(cur_x, cur_y, cur_c, hidden_dim, input_modes, output_modes, com_ranks_W, com_ranks_R, tfv_train_phase = tfv_train_phase, keep_prob = keep_prob,name_scope = 'ht_lstm_cell')
			else:
				cur_y, cur_c = lstm_cell(cur_x, cur_y, cur_c, hidden_dim, tfv_train_phase = tfv_train_phase, keep_prob = keep_prob, name_scope = 'lstm_cell')
			l_outputs.append(tf.expand_dims(cur_y, -1))

	return tf.concat(l_outputs, axis = -1)
