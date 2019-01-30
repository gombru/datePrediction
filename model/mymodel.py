import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import myinceptionv3
import math

class MyModel(nn.Module):

    def __init__(self, gpu=0):

        super(MyModel, self).__init__()
        c = {}
        c['num_classes'] = 2
        c['num_objects']
        c['object_features_len'] = 2048 * c['num_objects']
        c['gpu'] = gpu
        self.cnn = myinceptionv3.my_inception_v3(pretrained=True, aux_logits=False)
        self.fcm = justInception(c)
        self.initialize_weights()

    def forward(self, image, object_features):
        image_features = self.cnn(image)
        x = self.fcm(image_features, object_features)
        return x

    def initialize_weights(self):
        for m in self.mm.modules(): # Initialize only fcm weights
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class FCM(nn.Module):

    def __init__(self, c):
        super(FCM, self).__init__()

        self.fc1 = BasicFC(c['object_features_len'], 4096)
        self.fc2 = BasicFC(4096, 2048)
        self.fc2 = BasicFC(4096, 2048)
        self.fc3 = BasicFC(2048, c['num_classes'])

    def forward(self, image_features, object_features):

        # Reduce dimensionality fo object features to 2048
        object_features = self.fc1(object_features)
        object_features = self.fc2(object_features)

        object_features = F.dropout(object_features, training=self.training)

        x = torch.cat((image_features, object_features), dim=1)

        x = self.fc2(x)
        x = self.fc3(x)

        return x


class justInception(nn.Module):

    def __init__(self, c):
        super(justInception, self).__init__()

        self.fc1 = BasicFC(2048, c['num_classes'])

    def forward(self, image_features, object_features):

        x = self.fc1(image_features)

        return x


class BasicFC(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)