import tensorflow as tf
import tensorflow_addons as tfa
import datetime


class CallbackHelper:
    def __init__(self, base_dir, logs_dir="/logs"):
        self.base_dir = base_dir

        self.logs_base_dir = base_dir + logs_dir

        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def create_callback(self, callback_name, experiment=None, change_time=False, tb_write_images=True, tb_hist_freq=1,
                        es_monitor="val_loss", es_patience=10, lrs_schedule=None, lrs_verbose=0):
        callback = None

        if callback_name == 'MC':
            log_file = self.get_log_file(callback_name, experiment, change_time)
            callback = tf.keras.callbacks.ModelCheckpoint(log_file)

        if callback_name == 'TB':
            log_file = self.get_log_file(callback_name, experiment, change_time)
            callback = tf.keras.callbacks.TensorBoard(log_dir=log_file,
                                                      write_images=tb_write_images,
                                                      histogram_freq=tb_hist_freq)
        if callback_name == 'ES':
            callback = tf.keras.callbacks.EarlyStopping(monitor=es_monitor,
                                                        patience=es_patience)
        if callback_name == 'TQDM':
            callback = tfa.callbacks.TQDMProgressBar()

        if callback_name == 'LRS':
            callback = tf.keras.callbacks.LearningRateScheduler(lrs_schedule, lrs_verbose=lrs_verbose)

        print("Log file used = ", log_file)

        return callback

    def get_log_file(self, callback_name, experiment, change_time):
        if change_time:
            self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = self.logs_base_dir + experiment + "/" + self.timestamp + "/" + callback_name
        else:
            log_path = self.logs_base_dir + experiment + "/" + self.timestamp + "/" + callback_name
        return log_path
