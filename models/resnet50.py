from keras.applications.resnet50 import ResNet50 as KerasResNet50
from keras.layers import (Flatten, Dense, Dropout)
from keras.models import Model

import config
from .base_model import BaseModel#reference in the current directory


class ResNet50(BaseModel):#BaseModel is ResNet50's father
    noveltyDetectionLayerName = 'fc1'
    noveltyDetectionLayerSize = 2048

    def __init__(self, *args, **kwargs):
        super(ResNet50, self).__init__(*args, **kwargs)#call the father's __init__, and self means ResNet50's father: BaseModel
        if not self.freeze_layers_number:
            # we chose to train the top 2 identity blocks and 1 convolution block
            self.freeze_layers_number = 80

    def _create(self):
        base_model = KerasResNet50(include_top=False, input_tensor=self.get_input_tensor())#create base_model
        self.make_net_layers_non_trainable(base_model)#make net layers non trainable

        x = base_model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        # we could achieve almost the same accuracy without this layer, buy this one helps later
        # for novelty detection part and brings much more useful features.
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(config.classes), activation='softmax', name='predictions')(x)

        self.model = Model(input=base_model.input, output=predictions)


def inst_class(*args, **kwargs):
    return ResNet50(*args, **kwargs)
