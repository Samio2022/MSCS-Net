# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 16:56:48 2025

@author: Asim
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Conv3D, MaxPooling3D, BatchNormalization, Activation,
                                     Dropout, Conv3DTranspose, concatenate, Layer, GlobalAveragePooling3D,
                                     Reshape, Multiply, Add, Concatenate)

# --- Conv3DBN Layer ---
class Conv3DBN(Layer):
    def __init__(self, filters, num_row, num_col, num_z, padding='same', kernel_initializer='he_normal',
                 strides=(1, 1, 1), activation='relu', name=None, **kwargs):
        super(Conv3DBN, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.num_row = num_row
        self.num_col = num_col
        self.num_z = num_z
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.strides = strides
        self.activation = activation

    def build(self, input_shape):
        self.conv_layer = Conv3D(self.filters, (self.num_row, self.num_col, self.num_z),
                                 strides=self.strides, padding=self.padding, use_bias=False,
                                 kernel_initializer=self.kernel_initializer)
        self.batch_norm = BatchNormalization(axis=4, scale=False)

    def call(self, inputs):
        x = self.conv_layer(inputs)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = Activation(self.activation)(x)
        return x

# --- Deep Contextual Attention (DCA) Block ---
class DeepContextualAttention(Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(DeepContextualAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.global_avg_pool = GlobalAveragePooling3D()
        self.dense1 = Conv3D(input_shape[-1] // self.reduction_ratio, (1, 1, 1), activation='relu')
        self.dense2 = Conv3D(input_shape[-1], (1, 1, 1), activation='sigmoid')
        self.conv_spatial = Conv3D(1, (3, 3, 3), padding='same', activation='sigmoid')

    def call(self, inputs):
        # Channel-wise attention
        avg_pool = self.global_avg_pool(inputs)
        avg_pool = Reshape((1, 1, 1, inputs.shape[-1]))(avg_pool)
        dense1 = self.dense1(avg_pool)
        dense2 = self.dense2(dense1)
        channel_att = Multiply()([inputs, dense2])
        # Spatial attention
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([max_pool, avg_pool])
        conv1 = self.conv_spatial(concat)
        spatial_att = Multiply()([inputs, conv1])
        # Combine channel-wise and spatial attention
        out = Add()([channel_att, spatial_att])
        return out

# --- Deep Multi-Scale Attention (DMSA) Block ---
class DMSA_Block(Layer):
    def __init__(self, U, kernel_initializer='he_normal', padding='same', activation='relu',
                 se_ratio=0.2, alpha=1.67, **kwargs):
        super(DMSA_Block, self).__init__(**kwargs)
        self.U = U
        self.alpha = alpha
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.activation = activation
        self.se_ratio = se_ratio

    def build(self, input_shape):
        # Stronger width to compensate parameter drop from bigger kernels!
        W = int(self.alpha * self.U)
        num_filters_3 = W
        num_filters_5 = W
        num_filters_7 = W
        out_filters = num_filters_3 + num_filters_5 + num_filters_7

        # Main path convolutions
        self.conv3x3 = Conv3DBN(num_filters_3, 3, 3, 3, padding=self.padding, activation=self.activation,
                                kernel_initializer=self.kernel_initializer)
        self.conv5x5 = Conv3DBN(num_filters_5, 5, 5, 5, padding=self.padding, activation=self.activation,
                                kernel_initializer=self.kernel_initializer)
        self.conv7x7 = Conv3DBN(num_filters_7, 7, 7, 7, padding=self.padding, activation=self.activation,
                                kernel_initializer=self.kernel_initializer)

        # Shortcut path
        self.shortcut_conv = Conv3DBN(out_filters, 1, 1, 1, padding=self.padding, activation=self.activation,
                                      kernel_initializer=self.kernel_initializer)

        # Channel and Spatial Attention Layer
        self.Deep_Contextual_Attention_layer = DeepContextualAttention(reduction_ratio=int(1 / self.se_ratio))

        # Adjustment layer for channel mismatch
        self.channel_adjustment_conv = Conv3D(out_filters, (1, 1, 1), padding=self.padding, activation=None,
                                              kernel_initializer=self.kernel_initializer)
        self.batch_norm = BatchNormalization(axis=4)

    def call(self, inputs):
        x1 = self.conv3x3(inputs)
        x2 = self.conv5x5(x1)
        x3 = self.conv7x7(x2)
        main_path_out = Concatenate(axis=4)([x1, x2, x3])
        if self.se_ratio > 0:
            attention_out = self.Deep_Contextual_Attention_layer(main_path_out)
        else:
            attention_out = main_path_out
        shortcut_out = self.shortcut_conv(inputs)
        num_channels_main = main_path_out.shape[-1]
        num_channels_shortcut = shortcut_out.shape[-1]
        if num_channels_main != num_channels_shortcut:
            shortcut_out = self.channel_adjustment_conv(shortcut_out)
        combined_out = Add()([shortcut_out, attention_out])
        combined_out = self.batch_norm(combined_out)
        if self.activation is not None:
            combined_out = Activation(self.activation)(combined_out)
        return combined_out

# --- CBS Block ---
def CBS_block_3D(filters, kernel_size, strides=(1, 1, 1), padding='same', kernel_initializer='he_uniform'):
    def layer(x):
        x = Conv3D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('silu')(x)
        return x
    return layer

# --- SPPF Block ---
class SPPF(Layer):
    def __init__(self, filters, pool_sizes=[3, 5, 7], activation='leaky_relu', final_activation='silu', **kwargs):
        super(SPPF, self).__init__(**kwargs)
        self.filters = filters
        self.pool_sizes = pool_sizes
        self.activation = activation
        self.final_activation = final_activation

    def build(self, input_shape):
        self.cbl1 = Sequential([
            Conv3D(self.filters, (1, 1, 1), padding='same'),
            BatchNormalization(),
            Activation(self.activation)
        ])
        self.cbl2 = Sequential([
            Conv3D(self.filters * (len(self.pool_sizes) + 1), (1, 1, 1), padding='same'),
            BatchNormalization(),
            Activation(self.final_activation)
        ])
        self.pool_layers = [MaxPooling3D(pool_size=(size, size, size), strides=(1, 1, 1), padding='same')
                            for size in self.pool_sizes]
        super(SPPF, self).build(input_shape)

    def call(self, x):
        x_cbl1 = self.cbl1(x)
        pooled_outputs = [pool_layer(x_cbl1) for pool_layer in self.pool_layers]
        x_concat = concatenate([x_cbl1] + pooled_outputs, axis=-1)
        x_cbl2 = self.cbl2(x_concat)
        return x_cbl2

# --- MSCS-Net Model ---
def MSCS_Net(IMG_HEIGHT=192, IMG_WIDTH=192, IMG_DEPTH=16, IMG_CHANNELS=2, num_classes=2, kernel_initializer='he_uniform'):
    base_filters = 16  

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))

    # Downsampling path
    E1 = CBS_block_3D(base_filters * 2,(3, 3, 3), strides=(2, 2, 2), padding='same')(inputs)
    E2 = DMSA_Block(U=base_filters, kernel_initializer=kernel_initializer)(E1)
    E3 = CBS_block_3D(base_filters * 2, (3, 3, 3), strides=(2, 2, 2), padding='same')(E2)
    E4 = DMSA_Block(U=base_filters * 2, kernel_initializer=kernel_initializer)(E3)
    E5 = CBS_block_3D(base_filters * 4, (3, 3, 3), strides=(2, 2, 2), padding='same')(E4)
    E6 = DMSA_Block(U=base_filters * 4, kernel_initializer=kernel_initializer)(E5)
    E7 = CBS_block_3D(base_filters * 8, (3, 3, 3), strides=(2, 2, 2), padding='same')(E6)
    E8 = DMSA_Block(U=base_filters * 8, kernel_initializer=kernel_initializer)(E7)

    # Bottleneck SPPF
    SPFF = SPPF(base_filters * 16, pool_sizes=[3, 5, 7])(E8)
    SPFFADDC1 = concatenate([SPFF, E7])

    # Decoder path
    D1 = Conv3D(base_filters * 8, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(SPFFADDC1)
    D1 = Dropout(0.3)(D1)
    D1 = Conv3D(base_filters * 8, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(D1)
    D2 = Conv3DTranspose(base_filters * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(D1)
    C2 = concatenate([D2, E5])
    D3 = Conv3D(base_filters * 4, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(C2)
    D3 = Dropout(0.3)(D3)
    D3 = Conv3D(base_filters * 4, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(D3)
    D4 = Conv3DTranspose(base_filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(D3)
    C3 = concatenate([D4, E3])
    D5 = Conv3D(base_filters * 2, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(C3)
    D5 = Dropout(0.3)(D5)
    D5 = Conv3D(base_filters * 2, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(D5)
    D6 = Conv3DTranspose(base_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(D5)
    C4 = concatenate([D6, E1])
    D7 = Conv3D(base_filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(C4)
    D7 = Dropout(0.3)(D7)
    D7 = Conv3D(base_filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(D7)
    D8 = Conv3DTranspose(base_filters // 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(D7)
    C5 = concatenate([D8, inputs])
    D9 = Conv3D(base_filters // 2, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(C5)
    D9 = Dropout(0.3)(D9)
    D9 = Conv3D(base_filters // 2, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(D9)

    # Output
    output = Conv3D(num_classes, (1, 1, 1), activation='softmax', name='segmentation_output')(D9)

    model = Model(inputs=inputs, outputs=output)
    model.build((None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    model.summary()
    return model

# Build and print model
model = MSCS_Net(192, 192, 16, 2, 2)

print("Input Shape:", model.input_shape)
print("Output Shape:", model.output_shape)


