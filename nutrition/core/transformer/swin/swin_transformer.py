import tensorflow as tf

from nutrition.core.transformer.swin import DropPath
from nutrition.core.transformer.swin.swin_config import SwinConfig
from nutrition.core.transformer.swin.window_attention import WindowAttention
from nutrition.core.transformer.swin.window_utility import window_partition, window_reverse
import numpy as np
import copy


class SwinTransformer(tf.keras.layers.Layer):
    def __init__(
            self,
            swin_config: SwinConfig,
            **kwargs,
    ):
        super(SwinTransformer, self).__init__(**kwargs)

        self.attn_mask = None
        self.swin_config = copy.deepcopy(swin_config)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            self.swin_config.dim,
            window_size=(self.swin_config.window_size, self.swin_config.window_size),
            num_heads=self.swin_config.num_heads,
            qkv_bias=self.swin_config.qkv_bias,
            dropout_rate=self.swin_config.dropout_rate,
        )
        self.drop_path = DropPath(self.swin_config.dropout_rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.swin_config.num_mlp),
                tf.keras.layers.Activation(tf.keras.activations.gelu),
                tf.keras.layers.Dropout(self.swin_config.dropout_rate),
                tf.keras.layers.Dense(self.swin_config.dim),
                tf.keras.layers.Dropout(self.swin_config.dropout_rate),
            ]
        )

        if min(self.swin_config.num_patch) < self.swin_config.window_size:
            self.swin_config.shift_size = 0
            self.swin_config.window_size = min(self.swin_config.num_patch)

    def build(self, input_shape):
        if self.swin_config.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.swin_config.num_patch
            h_slices = (
                slice(0, -self.swin_config.window_size),
                slice(-self.swin_config.window_size, -self.swin_config.shift_size),
                slice(-self.swin_config.shift_size, None),
            )
            w_slices = (
                slice(0, -self.swin_config.window_size),
                slice(-self.swin_config.window_size, -self.swin_config.shift_size),
                slice(-self.swin_config.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.swin_config.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.swin_config.window_size * self.swin_config.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.swin_config.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.swin_config.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.swin_config.shift_size, -self.swin_config.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.swin_config.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.swin_config.window_size * self.swin_config.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.swin_config.window_size, self.swin_config.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.swin_config.window_size, height, width, channels
        )
        if self.swin_config.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.swin_config.shift_size, self.swin_config.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x
