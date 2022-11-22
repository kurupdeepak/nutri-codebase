from nutrition.core.data import DatasetConfig
from nutrition.core.data import DataLoader

LOGGER = "data-test -> "


def test_local(debug=False):
    data_config = DatasetConfig(base_dir='E://nutrition5k_dataset',
                                image_dir='/imagery/realsense_overhead/',
                                splits_dir="/dish_ids/splits",
                                metadata_dir="/metadata")
    data_loader = DataLoader(data_config=data_config, debug=debug)

    d, di = data_loader.get_data()
    train, test = data_loader.get_splits()
    print(f"{LOGGER} Dish shape = {d.shape}")
    print(f"{LOGGER} Dish Ingredient shape = {di.shape}")
    print(f"{LOGGER} Already split train ids = {train.shape}")
    print(f"{LOGGER} Already split test ids = {test.shape}")


test_local(True)
