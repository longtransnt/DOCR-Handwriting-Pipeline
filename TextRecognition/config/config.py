import yaml


class ConfigLoader(object):
    cfg = None

    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ConfigLoader.__instance is None:
            ConfigLoader()
        return ConfigLoader.__instance

    def __init__(self):
        if ConfigLoader.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ConfigLoader.__instance = self

        with open("./config/config.yml", "r", encoding='utf-8') as yml_file:
            dashed_line = '-' * 100
            print(f"{dashed_line}\nLoaded config file")
            self.cfg = yaml.load(yml_file, Loader=yaml.FullLoader)

            """ Displays """
            self.model_summary = self.cfg['display']['model_summary']
            self.trainable_params_summary = self.cfg['display']['trainable_params_summary']
            self.optimizer_summary = self.cfg['display']['optimizer_summary']

            """ Language """
            self.chars = self.cfg['language']['alphabet']
            self.batch_max_length = self.cfg['language']['max_length']

            """ Preprocess """
            self.pre_out_path = self.cfg['preprocess']['output_path']

            """ Training data"""
            self.batch_img_height = self.cfg['data']['image_height']
            self.batch_img_width = self.cfg['data']['image_width']
            self.pad_and_keep_ratio = self.cfg['data']['pad_and_keep_ratio']
            self.batch_size = self.cfg['data']['batch_size']

            """ Data path"""
            self.train_path = self.cfg['data']['train_path']
            self.valid_path = self.cfg['data']['valid_path']
            self.test_path = self.cfg['data']['test_path']
            self.pretrained_path = ""  # this variable takes arguments from terminal

            """ Training config"""
            self.worker_count = int(self.cfg['train']['worker_count'])
            # self.saved_model = self.cfg['']['']
            self.fine_tuning = self.cfg['train']['fine_tune']
            self.learning_rate = float(self.cfg['train']['learning_rate'])
            self.rho = float(self.cfg['train']['rho_rate'])
            self.epsilon = float(self.cfg['train']['epsilon'])
            self.grad_clip = float(self.cfg['train']['gradient_clipping'])
            self.val_interval = int(self.cfg['train']['validation_interval'])
            self.num_iter = int(self.cfg['train']['total_interval_count'])
            self.experiment_name = self.cfg['train']['model_name']
            self.random_seed = int(self.cfg['train']['random_seed'])
            self.save_interval = float(self.cfg['train']['save_interval'])

            """ ResNet config"""
            self.color_channel = self.cfg['ResNet']['input_color_channel']
            self.output_feature_channel = self.cfg['ResNet']['output_feature_channel']

            """" BiLSTM """
            self.hidden_size = self.cfg['BiLSTM']['hidden_size']

            self.saved_model = ""


if __name__ == '__main__':
    c = ConfigLoader()
    print(c)
