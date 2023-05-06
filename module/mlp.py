''' Modified from https://github.com/alinlab/LfF/blob/master/module/mlp.py'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_DISENTANGLE(nn.Module):
    def __init__(self, num_classes = 10, bias = True):
        super(MLP_DISENTANGLE, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100, bias = bias),
            nn.ReLU(),
            nn.Linear(100, 100, bias = bias),
            nn.ReLU(),
            nn.Linear(100, 16, bias = bias),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(16, num_classes))

        # self.projection_head = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(16,16, bias = bias))


    def extract(self, x):
        x = x.view(x.size(0), -1) / 255
        feat = self.feature(x)
        # feat = self.projection_head(x)
        return feat

    def extract_rawfeature(self, x):
        x = x.view(x.size(0), -1) / 255
        feat = self.feature(x)
        return feat

    def predict(self, x):
        prediction = self.fc(x)
        return prediction

    def forward(self, x, mode=None, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = x = self.feature(x)
        final_x = self.fc(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x




class MLP_DISENTANGLE_EASY(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP_DISENTANGLE_EASY, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 16),
            nn.ReLU()
        )
        self.fc = nn.Linear(32, num_classes)
        self.fc2 = nn.Linear(16, num_classes)

    def extract(self, x):
        x = x.view(x.size(0), -1) / 255
        feat = self.feature(x)
        return feat

    def predict(self, x):
        prediction = self.classifier(x)
        return prediction

    def forward(self, x, mode=None, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = x = self.feature(x)
        final_x = self.classifier(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x




class MLP(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.classifier = nn.Linear(16, num_classes)


    def forward(self, x, mode=None, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = x = self.feature(x)
        final_x = self.classifier(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x

class MLP_DISENTANGLE_SHENG(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP_DISENTANGLE_SHENG, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )

        self.task_head_target = nn.Sequential(
            nn.Linear(16,16),
            nn.ReLU(),
            # nn.Linear(16,16),
            # nn.ReLU(),
            )
        
        self.task_head_bias = nn.Sequential(
            nn.Linear(16,16),
            nn.ReLU(),
            # nn.Linear(16,16),
            # nn.ReLU(),
            )

        self.fc_target = nn.Linear(16, num_classes)
        self.fc_bias = nn.Linear(16, num_classes)
        # self.classifier_bias = nn.Linear(16, num_classes)


    def extract_target(self, x):
        x = x.view(x.size(0), -1) / 255
        x = self.feature(x)
        feat = self.task_head_target(x)
        return feat

    def extract_bias(self, x):
        x = x.view(x.size(0), -1) / 255
        x = self.feature(x)
        feat = self.task_head_bias(x)
        return feat

    def predict_target(self, x):
        prediction = self.fc_target(x)
        return prediction

    def predict_bias(self, x):
        prediction = self.fc_bias(x)
        return prediction

    def forward(self, x, mode=None, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = x = self.feature(x)
        feat_target = self.task_head_target(x)
        feat_bias = self.task_head_bias(x)

        final_x_target = self.fc_target(feat_target)
        final_x_bias = self.fc_bias(feat_bias)

        if mode == 'tsne' or mode == 'mixup':
            return feat_target, feat_bias, final_x_target, final_x_bias
        else:
            if return_feat:
                return final_x_target, feat_target, final_x_bias, feat_bias
            else:
                return final_x_target, final_x_bias


class Noise_MLP(nn.Module):
    def __init__(self, n_dim=16, n_layer=3):
        super(Noise_MLP, self).__init__()

        layers = []
        for i in range(n_layer):
            layers.append(nn.Linear(n_dim, n_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, z):
        x = self.style(z)
        return x
