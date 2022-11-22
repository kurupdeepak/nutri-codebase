from nutrition.core.transformer import VITConfig, ModelCreator


def test_vit_model():
    vit_config = VITConfig(image_size=256,
                           patch_size=16,
                           projection_dim=64,
                           num_heads=8,
                           transformer_layers=4,
                           mlp_head_units=[2048, 1024, 512, 64, 32],
                           output_shape=1)
    vit_model = ModelCreator.create_vit(vit_config)
    print("VIT Model Created, summary below")
    print(vit_model.summary())


test_vit_model()
