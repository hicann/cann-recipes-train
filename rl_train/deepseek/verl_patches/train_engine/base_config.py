# Adapted from
# https://gitcode.com/ascend/MindSpeed-RL/blob/2.1.0/mindspeed_rl/config_cls/base_config.py
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.


class BaseConfig:
    '''
    Base configuration class.
    From Mindspeed-RL.
    '''

    def __repr__(self):
        '''Represent the model config as a string for easy reading'''
        return f"<{self.__class__.__name__} {vars(self)}>"

    def update(self, config_dict, model_config_dict=None):
        '''
        Method to update parameters from a config dictionary
        '''
        if 'model' in config_dict:
            self.update(model_config_dict[config_dict['model']])

        for key, value in config_dict.items():
            if key == 'model':
                continue

            key = key.replace('-', '_')
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"The key: {key} is missing, causing the setup to fail. Please check."
                                 f" If necessary, register it in the config file.")

    def items(self):
        return self.__dict__.items()

    def dict(self):
        return self.__dict__
