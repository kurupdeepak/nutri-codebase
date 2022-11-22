from nutrition.core.transformer import ModelCreator
from nutrition.core.transformer.swin import SwinConfig

def test_swin_model():
    swin_config = SwinConfig(input_shape=(256, 256, 3),
                             patch_size=(2, 2),
                             embed_dim=64,
                             num_heads=8,
                             window_size=2,
                             shift_size=0,
                             num_mlp=256,
                             qkv_bias=True,
                             dropout_rate=0.03,
                             output_size=1)
    swin_model = ModelCreator.create_swin(swin_config)
    print("Swin Model Created, summary below")
    print(swin_model.summary())


test_swin_model()
