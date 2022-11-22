class SwinConfig:
    def __init__(self,
                 input_shape=None,
                 patch_size=None,
                 embed_dim=None,
                 num_heads=None,
                 window_size=None,
                 shift_size=0,
                 num_mlp=None,
                 qkv_bias=True,
                 dropout_rate=None,
                 output_size=1):
        """

        :type num_patch_x: numeric
        """
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.dim = embed_dim  # number of input dimensions
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of attention window
        self.shift_size = shift_size  # size of shifting window
        self.num_mlp = num_mlp  # number of MLP nodes
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.output_size = output_size
        self.num_patch_x = input_shape[0] // patch_size[0]
        self.num_patch_y = input_shape[1] // patch_size[1]
        self.num_patch = (self.num_patch_x, self.num_patch_y)  # number of embedded patches
