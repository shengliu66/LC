''' Modified from https://github.com/alinlab/LfF/blob/master/module/util.py '''

import torch.nn as nn
from module.resnet import resnet20
from module.mlp import *
from torchvision.models import resnet18, resnet50
from transformers import BertForSequenceClassification, BertConfig

def get_model(model_tag, num_classes, bias = True):
    if model_tag == "ResNet20":
        return resnet20(num_classes)
    elif model_tag == "ResNet20_OURS":
        model = resnet20(num_classes)
        model.fc = nn.Linear(128, num_classes)
        return model
    elif model_tag == "ResNet18":
        print('bringing no pretrained resnet18 ...')
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
        return model
    elif model_tag == "MLP":
        return MLP(num_classes=num_classes)
    elif model_tag == "mlp_DISENTANGLE":
        return MLP_DISENTANGLE(num_classes=num_classes, bias = bias)
    elif model_tag == "mlp_DISENTANGLE_EASY":
        return MLP_DISENTANGLE_EASY(num_classes=num_classes)
    elif model_tag == 'resnet_DISENTANGLE':
        print('bringing no pretrained resnet18 disentangle...')
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(1024//2, num_classes)
        return model
    elif model_tag == 'resnet_DISENTANGLE_pretrained':
        print('bringing pretrained resnet18 disentangle...')
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(1024//2, num_classes)
        return model

    elif model_tag == 'resnet_50_pretrained':
        print('bringing pretrained resnet50 for water bird...')
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
        return model
    elif model_tag == 'resnet_50':
        print('bringing pretrained resnet50 for water bird...')
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_classes)
        return model
    elif 'bert' in model_tag:
        if model_tag[-3:] == '_pt':
            model_name = model_tag[:-3]
        else:
            model_name = model_tag

        config_class = BertConfig
        model_class = BertForSequenceClassification
        config = config_class.from_pretrained(model_name,
                  num_labels=num_classes,
                  finetuning_task='civilcomments')
        model = model_class.from_pretrained(model_name, from_tf=False, 
                                          config=config)
        model.activation_layer = 'bert.pooler.activation'
        return model 
    else:
        raise NotImplementedError
