from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.regularizers import l2



def make_layer_list(arch, network_type=None, reg=None, dropout=0):
    '''
    Generates the list of layers specified by arch, to be stacked
    by stack_layers (defined in src/core/layer.py)
    arch:           list of dicts, where each dict contains the arguments
                    to the corresponding layer function in stack_layers
    network_type:   siamese or spectral net. used only to name layers
    reg:            L2 regularization (if any)
    dropout:        dropout (if any)
    returns:        appropriately formatted stack_layers dictionary
    '''
    layers = []
    for i, a in enumerate(arch):

        layer = {'l2_reg': reg}
        layer.update(a)
        if network_type:
            layer['name'] = '{}_{}'.format(network_type, i)
        layers.append(layer)
        if a['type'] != 'Flatten' and dropout != 0:
            dropout_layer = {
                'type': 'Dropout',
                'rate': dropout,
                }
            if network_type:
                dropout_layer['name'] = '{}_dropout_{}'.format(network_type, i)
            layers.append(dropout_layer)
    return layers

def stack_layers(inputs,layers, kernel_initializer='glorot_uniform'):
    '''
    Builds the architecture of the network by applying each layer specified in layers to inputs.
    inputs:     a dict containing input_types and input_placeholders for each key and value pair, respecively.
                for spectralnet, this means the input_types 'Unlabeled' and 'Orthonorm'*
    layers:     a list of dicts containing all layers to be used in the network, where each dict describes
                one such layer. each dict requires the key 'type'. all other keys are dependent on the layer
                type
    kernel_initializer: initialization configuration passed to keras (see keras initializers)
    returns:    outputs, a dict formatted in much the same way as inputs. it contains input_types and
                output_tensors for each key and value pair, respectively, where output_tensors are
                the outputs of the input_placeholders in inputs after each layer in layers is applied
    * this is necessary since spectralnet takes multiple inputs and performs special computations on the
      orthonorm layer
    '''
    outputs = {}
    for key in inputs:
        outputs[key] = inputs[key]

    for layer in layers:
        # check for l2_reg argument
        l2_reg = layer.get('l2_reg')
        if l2_reg:
            l2_reg = l2(layer['l2_reg'])

        if layer['type'] == 'softplus_reg':
            l = Dense(layer['size'], activation='softplus', kernel_initializer=kernel_initializer, kernel_regularizer=l2(0.001), name=layer.get('name'))
        elif layer['type'] == 'softplus':
            l = Dense(layer['size'], activation='softplus', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'softmax':
            l = Dense(layer['size'], activation='softmax', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'tanh':
            l = Dense(layer['size'], activation='tanh', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'relu':
            l = Dense(layer['size'], activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'selu':
            l = Dense(layer['size'], activation='selu', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'Conv2D':
            l = Conv2D(layer['channels'], kernel_size=layer['kernel'], activation='relu', data_format='channels_last', kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'BatchNormalization':
            l = BatchNormalization(name=layer.get('name'))
        elif layer['type'] == 'MaxPooling2D':
            l = MaxPooling2D(pool_size=layer['pool_size'], name=layer.get('name'))
        elif layer['type'] == 'Dropout':
            l = Dropout(layer['rate'], name=layer.get('name'))
        elif layer['type'] == 'Flatten':
            l = Flatten(name=layer.get('name'))
        else:
            raise ValueError("Invalid layer type '{}'".format(layer['type']))

        for k in inputs:
            if (outputs[k].shape[-1]!=None):
                outputs[k]=l(outputs[k])

    return outputs




