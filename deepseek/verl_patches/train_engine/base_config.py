# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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