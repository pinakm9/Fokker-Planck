# create and execute layers for nnsolvers
import tensorflow as tf

def make_admissible_params(params, param_details):
    for i in range(len(params)):
        if 'positive' in param_details[i]:
            if param_details[i]['positive']:
                params[i] = tf.keras.activations.exponential(params[i])
        if 'in_0_1' in param_details[i]:
            if param_details[i]['in_0_1']:
                params[i] = tf.keras.activations.sigmoid(params[i])
    return params

class ParamInitUniform(tf.keras.initializers.Initializer):
    def __init__(self, domain=[0., 1.], param_details={}):
        super(ParamInitUniform, self).__init__()
        self.domain = domain
        self.param_details = param_details

    def __call__(self, shape, dtype=tf.float32):
        param = tf.random.uniform(shape, self.domain[0], self.domain[1], dtype=dtype)
        return make_admissible_params([param], [self.param_details])[0]

















































def build_LSTM_layer(obj, input_dim, num_nodes, layer_name='', layer_index='l', initializer='random_normal', regularizer=None):
    name = layer_name + '_U_z_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(input_dim, num_nodes), initializer=initializer, regularizer=regularizer, trainable=True))
    name = layer_name + '_W_z_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, num_nodes), initializer=initializer, regularizer=regularizer, trainable=True))
    name = layer_name + '_b_z_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, ), initializer=initializer, regularizer=regularizer, trainable=True))
    name = layer_name + '_U_g_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(input_dim, num_nodes), initializer=initializer, regularizer=regularizer, trainable=True))
    name = layer_name + '_W_g_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, num_nodes), initializer=initializer, regularizer=regularizer, trainable=True))
    name = layer_name + '_b_g_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, ), initializer=initializer, regularizer=regularizer, trainable=True))
    name = layer_name + '_U_r_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(input_dim, num_nodes), initializer=initializer, regularizer=regularizer, trainable=True))
    name = layer_name + '_W_r_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, num_nodes), initializer=initializer, regularizer=regularizer, trainable=True))
    name = layer_name + '_b_r_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, ), initializer=initializer, regularizer=regularizer, trainable=True))
    name = layer_name + '_U_h_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(input_dim, num_nodes), initializer=initializer, regularizer=regularizer, trainable=True))
    name = layer_name + '_W_h_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, num_nodes), initializer=initializer, regularizer=regularizer, trainable=True))
    name = layer_name + '_b_h_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, ), initializer=initializer, regularizer=regularizer, trainable=True))

def call_LSTM_layer(obj, S_l, input, layer_name='', layer_index='l', activation=tf.keras.activations.tanh):
    d = input.shape[-1]
    Z_l = activation(tf.matmul(input, getattr(obj, layer_name + '_U_z_' + str(layer_index))[: d]) + \
                     tf.matmul(S_l, getattr(obj, layer_name + '_W_z_' + str(layer_index))) + getattr(obj, layer_name + '_b_z_' + str(layer_index)))
    G_l = activation(tf.matmul(input, getattr(obj, layer_name + '_U_g_' + str(layer_index))[: d]) + \
                     tf.matmul(S_l, getattr(obj, layer_name + '_W_g_' + str(layer_index))) + getattr(obj, layer_name + '_b_g_' + str(layer_index)))
    R_l = activation(tf.matmul(input, getattr(obj, layer_name + '_U_r_' + str(layer_index))[: d]) + \
                     tf.matmul(S_l, getattr(obj, layer_name + '_W_r_' + str(layer_index))) + getattr(obj, layer_name + '_b_r_' + str(layer_index)))
    H_l = activation(tf.matmul(input, getattr(obj, layer_name + '_U_h_' + str(layer_index))[: d]) + \
                     tf.matmul(tf.multiply(S_l, R_l), getattr(obj, layer_name + '_W_h_' + str(layer_index))) + \
                     getattr(obj, layer_name + '_b_h_' + str(layer_index)))
    return tf.multiply(1. - G_l, H_l) + tf.multiply(Z_l, S_l)

def build_LSTM(obj, nn_name, input_dim, output_dim, num_nodes, num_layers, initializer='random_normal', b_f_initializer=None, regularizer=None, b_f_regularizer=None):
    if input_dim > 0:
        name = nn_name + '_W_0'
        setattr(obj, name, obj.add_weight(name=name, shape=(input_dim, num_nodes), initializer=initializer, trainable=True, regularizer=regularizer))
        name = nn_name + '_b_0'
        setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, ), initializer=initializer, trainable=True, regularizer=regularizer))
        for l in range(obj.num_layers):
            build_LSTM_layer(obj, input_dim=input_dim, num_nodes=num_nodes, layer_name=nn_name, layer_index=l, initializer=initializer)
        name = nn_name + '_W_f'
        setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, output_dim), initializer=initializer, trainable=True, regularizer=regularizer))
        name = nn_name + '_b_f'
        setattr(obj, name, obj.add_weight(name=name, shape=(output_dim, ), initializer=initializer if b_f_initializer is None else b_f_initializer,\
                                          trainable=True, regularizer=b_f_regularizer))
    else:
        name = nn_name + '_b_f'
        setattr(obj, name, obj.add_weight(name=name, shape=(output_dim, ), initializer=initializer if b_f_initializer is None else b_f_initializer,\
                                          trainable=True, regularizer=b_f_regularizer))

def call_LSTM(obj, nn_name, input, num_layers, activation=tf.keras.activations.tanh, final_activation=tf.keras.activations.linear, input_dim=1):
    if input_dim > 0:
        d = input.shape[-1]
        #input = tf.keras.layers.BatchNormalization()(input)
        output = activation(tf.matmul(input, getattr(obj, nn_name + '_W_0')[: d]) + getattr(obj, nn_name + '_b_0'))
        for l in range(num_layers):
            output = call_LSTM_layer(obj, S_l=output, input=input, layer_name=nn_name, layer_index=l, activation=activation)
        output = tf.matmul(output, getattr(obj, nn_name + '_W_f')) + getattr(obj, nn_name + '_b_f')
        return final_activation(output)
    else:
        return final_activation(0.0*input +  getattr(obj, nn_name + '_b_f'))


























def build_type_1_layer(obj, input_dim, num_nodes, layer_name='', layer_index='l', initializer='random_normal', regularizer=None):
    name = layer_name + '_U_z_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(input_dim, num_nodes), initializer=initializer, regularizer=regularizer, trainable=True))
    name = layer_name + '_W_z_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, num_nodes), initializer=initializer, regularizer=regularizer, trainable=True))
    name = layer_name + '_b_z_' + str(layer_index)
    setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, ), initializer=initializer, regularizer=regularizer, trainable=True))

def call_type_1_layer(obj, S_l, input, layer_name='', layer_index='l', activation=tf.keras.activations.tanh):
    d = input.shape[-1]
    Z_l = activation(tf.matmul(input, getattr(obj, layer_name + '_U_z_' + str(layer_index))[: d]) + \
                     tf.matmul(S_l, getattr(obj, layer_name + '_W_z_' + str(layer_index))) + getattr(obj, layer_name + '_b_z_' + str(layer_index)))
    return tf.multiply(Z_l, S_l)

def build_type_1(obj, nn_name, input_dim, output_dim, num_nodes, num_layers, initializer='zeros', regularizer=None, b_f_initializer=None):
    name = nn_name + '_W_0'
    setattr(obj, name, obj.add_weight(name=name, shape=(input_dim, num_nodes), initializer=initializer, regularizer=regularizer, trainable=True))
    name = nn_name + '_b_0'
    setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, ), initializer=initializer, regularizer=regularizer, trainable=True))
    for l in range(obj.num_layers):
        build_type_1_layer(obj, input_dim=input_dim, num_nodes=num_nodes, layer_name=nn_name, layer_index=l, initializer=initializer,
                           regularizer=regularizer)
    name = nn_name + '_W_f'
    setattr(obj, name, obj.add_weight(name=name, shape=(num_nodes, output_dim), initializer=initializer, regularizer=regularizer, trainable=True))
    name = nn_name + '_b_f'
    setattr(obj, name, obj.add_weight(name=name, shape=(output_dim, ),\
                                      initializer=initializer if b_f_initializer is None else b_f_initializer, trainable=True))

def call_type_1(obj, nn_name, input, num_layers, activation=tf.keras.activations.tanh, final_activation=tf.keras.activations.linear):
    d = input.shape[-1]
    output = activation(tf.matmul(input, getattr(obj, nn_name + '_W_0')[: d]) + getattr(obj, nn_name + '_b_0'))
    for l in range(num_layers):
        output = call_type_1_layer(obj, S_l=output, input=input, layer_name=nn_name, layer_index=l, activation=activation)
    output = tf.matmul(output, getattr(obj, nn_name + '_W_f')) + getattr(obj, nn_name + '_b_f')
    return final_activation(output)
