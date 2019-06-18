"""Util for Snakefile_reconstruction"""
import tensorflow as tf
from keras import layers
from keras.models import load_model, Model
from keras.layers import deserialize as layer_from_config

from keras.optimizers import *
import numpy as np


def copy_layer(input_layer):
    """get the config"""
    config_correct = {}
    config = input_layer.get_config()
    config_correct['class_name'] = input_layer.__class__.__name__
    config_correct['config'] = config
    #print(input_layer.get_config())
    return config_correct


# layer 19_slice
def slice(tensor):
    """helper for a lamda layer"""
    out = tensor[:, 444:524, :]
    return out


def slice_output_shape(input_shape):
    """helper for a lamda layer"""
    shape_1 = input_shape[0]
    shape_2 = 80
    shape_3 = input_shape[2]
    return (shape_1, shape_2, shape_3)


# layer 19_padding
def pad(tensor):
    """helper for a lamda layer"""
    paddings = tf.constant([[0, 0], [0, 10], [0, 0]])
    out = tf.pad(tensor, paddings, "CONSTANT", constant_values=0)
    return out


def pad_output_shape(input_shape):
    """helper for a lamda layer"""
    shape_1 = input_shape[0]
    shape_2 = 90
    shape_3 = input_shape[2]

    return (shape_1, shape_2, shape_3)

# layer 19_reshape_0
def reshape_0(tensor):
    """helper for a lamda layer"""
    row = tf.shape(tensor)[0]
    og_shape = tensor.get_shape().as_list()
    shape_list = [row, og_shape[1], og_shape[2], 1]
    out = tf.reshape(tensor, shape_list)
    return out


def reshape_output_shape_0(input_shape):
    """helper for a lamda layer""" 
    shape_1 = input_shape[0]
    shape_2 = input_shape[1]
    shape_3 = input_shape[2]
    return(shape_1, shape_2, shape_3, 1)

# layer 19_reshape
def reshape(tensor):
    """helper for a lamda layer"""
    row = tf.shape(tensor)[0]
    shape_list = [row, -1]
    out = tf.reshape(tensor, shape_list)
    return out


def reshape_output_shape(input_shape):
    """helper for a lamda layer"""
    shape_1 = input_shape[0]
    shape_2 = 384
    return(shape_1, shape_2)



def weight_difference(weights_og, new_weights):
    """calculate the difference between weights of the same layer from the new and the old model; used when troubleshooting"""
    difference = weights_og - new_weights
    return(np.absolute(difference).sum())


def github_model_builder(loaded_bottleneck_model, top_layer):
    """model rebuilding based on the github/basepair/models.py/binary_seq_multitask()"""
    weights = {}
    input1 = layers.Input(loaded_bottleneck_model.layers[0].input_shape[1:])
    weights[0] = loaded_bottleneck_model.layers[0].get_weights()
    # first_conv
    input_layer = loaded_bottleneck_model.layers[1]
    weights[1] = input_layer.get_weights()
    first_conv = layer_from_config(copy_layer(input_layer))(input1)
    prev_layers = [first_conv]
    for i in range(1, 10):
        if i == 1:
            prev_sum = first_conv
        else:
            prev_sum = layers.add(prev_layers)
            weights[i*2-1] = loaded_bottleneck_model.layers[i*2-1].get_weights()
        input_layer = loaded_bottleneck_model.layers[i*2]
        weights[i*2] = input_layer.get_weights()
        conv_output = layer_from_config(copy_layer(input_layer))(prev_sum)
        prev_layers.append(conv_output)
    combined_conv = layers.add(prev_layers, name='final_conv')
    weights[19] = loaded_bottleneck_model.layers[-1].get_weights()
    # layer 19_slice
    lambda_layer = layers.Lambda(
        slice, output_shape=slice_output_shape)(combined_conv)
    lambda_layer.trainable = False
    # layer 19_padding
    padding_layer = layers.Lambda(
        pad, output_shape=pad_output_shape)(lambda_layer)
    padding_layer.trainable = False
    # layer 19_reshape_0
    reshaping_layer_0 = layers.Lambda(
        reshape_0, output_shape=reshape_output_shape_0)(padding_layer)
    reshaping_layer_0.trainable = False
    pooling_layer = layers.MaxPooling2D(pool_size=(
        15, 1), strides=None, padding="same")(reshaping_layer_0)
    pooling_layer.trainable = False
    reshaping_layer = layers.Lambda(
        reshape, output_shape=reshape_output_shape)(pooling_layer)
    reshaping_layer.trainable = False
    # layer 20
    input_layer = top_layer.layers[0]
    weights[20] = input_layer.get_weights()
    lr = layer_from_config(copy_layer(input_layer))(reshaping_layer)
    new_model = Model(inputs=input1, outputs=lr)
    for i in range(20):
        new_model.layers[i].set_weights(weights[i])
    new_model.layers[-1].set_weights(weights[20])
    btnk_model = Model(inputs=input1, outputs=combined_conv)
    for i in range(20):
        btnk_model.layers[i].set_weights(weights[i])
    print(loaded_bottleneck_model.summary())
    print(top_layer.summary())
    print(new_model.summary())
    print(btnk_model.summary())
    return new_model, btnk_model


def model_builder(loaded_bottleneck_model, top_layer):
    """same as github_model_builder but build layerby layer without looping"""
    input1 = layers.Input(loaded_bottleneck_model.layers[0].input_shape[1:])
    weights_0 = loaded_bottleneck_model.layers[0].get_weights()
    # layer 1
    input_layer = loaded_bottleneck_model.layers[1]
    weights_1 = input_layer.get_weights()
    conv1d_1 = layer_from_config(copy_layer(input_layer))(input1)
    # layer 2
    input_layer = loaded_bottleneck_model.layers[2]
    weights_2 = input_layer.get_weights()
    conv1d_2 = layer_from_config(copy_layer(input_layer))(conv1d_1)
    # layer 3
    input_layer = loaded_bottleneck_model.layers[3]
    print(input_layer.get_config())
    weights_3 = input_layer.get_weights()
    add_1 = layers.Add()([conv1d_1, conv1d_2])
    # layer 4
    input_layer = loaded_bottleneck_model.layers[4]
    weights_4 = input_layer.get_weights()
    conv1d_3 = layer_from_config(copy_layer(input_layer))(add_1)
    # layer 5
    input_layer = loaded_bottleneck_model.layers[5]
    weights_5 = input_layer.get_weights()
    print(input_layer.get_config())
    add_2 = layers.Add()([add_1, conv1d_3])
    # layer 6
    input_layer = loaded_bottleneck_model.layers[6]
    weights_6 = input_layer.get_weights()
    conv1d_4 = layer_from_config(copy_layer(input_layer))(add_2)
    # layer 7
    input_layer = loaded_bottleneck_model.layers[7]
    weights_7 = input_layer.get_weights()
    print(input_layer.get_config())
    add_3 = layers.Add()([add_2, conv1d_4])
    # layer 8
    input_layer = loaded_bottleneck_model.layers[8]
    weights_8 = input_layer.get_weights()
    conv1d_5 = layer_from_config(copy_layer(input_layer))(add_3)
    # layer 9
    input_layer = loaded_bottleneck_model.layers[9]
    weights_9 = input_layer.get_weights()
    print(input_layer.get_config())
    add_4 = layers.Add()([add_3, conv1d_5])
    # layer 10
    input_layer = loaded_bottleneck_model.layers[10]
    weights_10 = input_layer.get_weights()
    conv1d_6 = layer_from_config(copy_layer(input_layer))(add_4)
    # layer 11
    input_layer = loaded_bottleneck_model.layers[11]
    weights_11 = input_layer.get_weights()
    print(input_layer.get_config())
    add_5 = layers.Add()([add_4, conv1d_6])
    # layer 12
    input_layer = loaded_bottleneck_model.layers[12]
    weights_12 = input_layer.get_weights()
    conv1d_7 = layer_from_config(copy_layer(input_layer))(add_5)
    # layer 13
    input_layer = loaded_bottleneck_model.layers[13]
    weights_13 = input_layer.get_weights()
    print(input_layer.get_config())
    add_6 = layers.Add()([add_5, conv1d_7])
    # layer 14
    input_layer = loaded_bottleneck_model.layers[14]
    weights_14 = input_layer.get_weights()
    conv1d_8 = layer_from_config(copy_layer(input_layer))(add_6)
    # layer 15
    input_layer = loaded_bottleneck_model.layers[15]
    weights_15 = input_layer.get_weights()
    print(input_layer.get_config())
    add_7 = layers.Add()([add_6, conv1d_8])
    # layer 16
    input_layer = loaded_bottleneck_model.layers[16]
    weights_16 = input_layer.get_weights()
    conv1d_9 = layer_from_config(copy_layer(input_layer))(add_7)
    # layer 17
    input_layer = loaded_bottleneck_model.layers[17]
    weights_17 = input_layer.get_weights()
    print(input_layer.get_config())
    add_8 = layers.Add()([add_7, conv1d_9])
    # layer 18
    input_layer = loaded_bottleneck_model.layers[18]
    weights_18 = input_layer.get_weights()
    conv1d_10 = layer_from_config(copy_layer(input_layer))(add_8)
    # layer 19
    input_layer = loaded_bottleneck_model.layers[19]
    weights_19 = input_layer.get_weights()
    print(input_layer.get_config())
    add_9 = layers.Add()([add_8, conv1d_10])
    # layer 19_slice
    lambda_layer = layers.Lambda(slice, output_shape=slice_output_shape)(add_9)
    lambda_layer.trainable = False
    # layer 19_padding
    padding_layer = layers.Lambda(
        pad, output_shape=pad_output_shape)(lambda_layer)
    padding_layer.trainable = False
    # layer 19_reshape_0
    reshaping_layer_0 = layers.Lambda(
        reshape_0, output_shape=reshape_output_shape_0)(padding_layer)
    reshaping_layer_0.trainable = False
    pooling_layer = layers.MaxPooling2D(pool_size=(
        15, 1), strides=None, padding="same")(reshaping_layer_0)
    pooling_layer.trainable = False
    reshaping_layer = layers.Lambda(
        reshape, output_shape=reshape_output_shape)(pooling_layer)
    reshaping_layer.trainable = False
    # layer 20
    input_layer = top_layer.layers[0]
    weights_20 = input_layer.get_weights()
    lr = layer_from_config(copy_layer(input_layer))(reshaping_layer)
    # weights
    new_model = Model(inputs=input1, outputs=lr)
    bpnk_model = Model(inputs=input1, outputs=add_9)
    new_model.layers[0].set_weights(weights_0)
    new_model.layers[1].set_weights(weights_1)
    new_model.layers[2].set_weights(weights_2)
    new_model.layers[3].set_weights(weights_3)
    new_model.layers[4].set_weights(weights_4)
    new_model.layers[5].set_weights(weights_5)
    new_model.layers[6].set_weights(weights_6)
    new_model.layers[7].set_weights(weights_7)
    new_model.layers[8].set_weights(weights_8)
    new_model.layers[9].set_weights(weights_9)
    new_model.layers[10].set_weights(weights_10)
    new_model.layers[11].set_weights(weights_11)
    new_model.layers[12].set_weights(weights_12)
    new_model.layers[13].set_weights(weights_13)
    new_model.layers[14].set_weights(weights_14)
    new_model.layers[15].set_weights(weights_15)
    new_model.layers[16].set_weights(weights_16)
    new_model.layers[17].set_weights(weights_17)
    new_model.layers[18].set_weights(weights_18)
    new_model.layers[19].set_weights(weights_19)
    # for i in range(20,26):
    # new_model.layers[i].set_weights(weights)
    new_model.layers[-1].set_weights(weights_20)

    bpnk_model.layers[0].set_weights(weights_0)
    bpnk_model.layers[1].set_weights(weights_1)
    bpnk_model.layers[2].set_weights(weights_2)
    bpnk_model.layers[3].set_weights(weights_3)
    bpnk_model.layers[4].set_weights(weights_4)
    bpnk_model.layers[5].set_weights(weights_5)
    bpnk_model.layers[6].set_weights(weights_6)
    bpnk_model.layers[7].set_weights(weights_7)
    bpnk_model.layers[8].set_weights(weights_8)
    bpnk_model.layers[9].set_weights(weights_9)
    bpnk_model.layers[10].set_weights(weights_10)
    bpnk_model.layers[11].set_weights(weights_11)
    bpnk_model.layers[12].set_weights(weights_12)
    bpnk_model.layers[13].set_weights(weights_13)
    bpnk_model.layers[14].set_weights(weights_14)
    bpnk_model.layers[15].set_weights(weights_15)
    bpnk_model.layers[16].set_weights(weights_16)
    bpnk_model.layers[17].set_weights(weights_17)
    bpnk_model.layers[18].set_weights(weights_18)
    bpnk_model.layers[19].set_weights(weights_19)

    new_model.compile(optimizer="Adam", loss="mean_squared_error")
    bpnk_model.compile(optimizer="Adam", loss="mean_squared_error")
    return new_model, bpnk_model
