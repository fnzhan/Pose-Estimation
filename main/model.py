import os
import os.path as osp
import sys

cur_dir = osp.dirname(os.path.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, '..', '..', 'common'))
sys.path.insert(0, osp.join(cur_dir, '..'))

import torch
import torch.nn as nn
from torch.nn import functional as F
from common.nets.resnet import ResNetBackbone
from main.config import cfg
from IPython import embed
import numpy as np
from torchvision.models.resnet import BasicBlock, Bottleneck


class Upsampling_Net(nn.Module):

    def __init__(self, joint_num):
        self.inplanes = 2048
        self.outplanes = 256

        super(Upsampling_Net, self).__init__()

        self.deconv_layers = self._make_deconv_layer(3)
        self.final_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

    def _make_deconv_layer(self, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=self.outplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(self.outplanes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = self.outplanes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        x = self.relu(self.bn(x))

        return x

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class limbHeatmap_Net(nn.Module):

    def __init__(self, joint_num):
        self.inplanes = 2048
        self.outplanes = 128

        super(limbHeatmap_Net, self).__init__()

        self.deconv_layers = self._make_deconv_layer(3)

        self.hm_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.final_layer1 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

    def _make_deconv_layer(self, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=self.outplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(self.outplanes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = self.outplanes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        hm = self.hm_layer(x)
        return x, hm

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPP_Bottleneck(nn.Module):
    def __init__(self):
        super(ASPP_Bottleneck, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(256, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(256, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w),
                             mode="bilinear")

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img],
                        1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        return out


class Serial_Net(nn.Module):

    def __init__(self, joint_num):
        self.inplanes = 2048
        self.outplanes = 128

        super(Serial_Net, self).__init__()
        self.up_feature_bottleneck1 = Bottleneck(inplanes=256, planes=64, stride=1)
        self.up_feature_bottleneck2 = Bottleneck(inplanes=256, planes=64, stride=1)
        self.limb_feature_bottlenneck1 = Bottleneck(inplanes=256, planes=64, stride=1)
        self.limb_feature_bottlenneck2 = Bottleneck(inplanes=256, planes=64, stride=1)
        self.limb_feature_bottlenneck3 = Bottleneck(inplanes=256, planes=64, stride=1)
        self.limb_feature_bottlenneck4 = Bottleneck(inplanes=256, planes=64, stride=1)

        self.aspp = ASPP_Bottleneck()

        self.down_sample_conv1 = nn.Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.down_sample_bn1 = nn.BatchNorm2d(256)

        self.down_sample_conv2 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.down_sample_bn2 = nn.BatchNorm2d(256)

        self.final_layer1 = nn.Conv2d(
            in_channels=256,
            out_channels=18,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.final_layer2 = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_feature, limb_feature):
        limb_feature = torch.cat([up_feature, limb_feature], dim=1)
        limb_feature = self.relu(self.down_sample_bn1(self.down_sample_conv1(limb_feature)))
        up_feature = self.up_feature_bottleneck1(up_feature)
        up_feature = self.up_feature_bottleneck2(up_feature)
        limb_feature = self.limb_feature_bottlenneck1(limb_feature)
        limb_feature = self.limb_feature_bottlenneck2(limb_feature)
        hm_2d = self.final_layer1(limb_feature)

        limb_feature = self.limb_feature_bottlenneck3(limb_feature)
        up_feature = self.aspp(up_feature)
        limb_feature = torch.cat([up_feature, limb_feature], dim=1)
        limb_feature = self.relu(self.down_sample_bn2(self.down_sample_conv2(limb_feature)))

        limb_feature = self.limb_feature_bottlenneck4(limb_feature)
        vox = self.final_layer2(limb_feature)

        return up_feature, limb_feature, hm_2d, vox

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Final_block(nn.Module):

    def __init__(self, joint_num):
        super(Final_block, self).__init__()
        self.bottleNeck1 = Bottleneck(inplanes=512, planes=128, stride=1)
        self.bottleNeck2 = Bottleneck(inplanes=512, planes=128, stride=1)
        self.bottleNeck3 = Bottleneck(inplanes=512, planes=128, stride=1)
        self.final_conv = nn.Conv2d(in_channels=512, out_channels=18 * cfg.depth_dim, kernel_size=1,
                                    stride=1, padding=0)

    def forward(self, x):
        x = self.bottleNeck3(self.bottleNeck2(self.bottleNeck1(x)))
        x = self.final_conv(x)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def soft_argmax(heatmaps, joint_num):
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim * cfg.output_shape[0] * cfg.output_shape[1]))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim, cfg.output_shape[0], cfg.output_shape[1]))

    accu_x = heatmaps.sum(dim=(2, 3))
    accu_y = heatmaps.sum(dim=(2, 4))
    accu_z = heatmaps.sum(dim=(3, 4))

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(1, cfg.output_shape[1] + 1).type(torch.cuda.FloatTensor),
                                                devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(1, cfg.output_shape[0] + 1).type(torch.cuda.FloatTensor),
                                                devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(1, cfg.depth_dim + 1).type(torch.cuda.FloatTensor),
                                                devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True) - 1
    accu_y = accu_y.sum(dim=2, keepdim=True) - 1
    accu_z = accu_z.sum(dim=2, keepdim=True) - 1

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out


class ResPoseNet(nn.Module):
    def __init__(self, backbone, up, limb, sel, final, joint_num):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.up = up
        self.sel = sel
        self.limb = limb
        self.final = final
        self.joint_num = joint_num

    def forward(self, input_img, target=None):
        # import time
        # t1 = time.time()
        x = self.backbone(input_img)
        up_feature = self.up(x)
        limb_feature, limb_heatmap = self.limb(x)
        up_feature, limb_feature, hm2, vox = self.sel(up_feature, limb_feature)
        x = torch.cat([up_feature, limb_feature], dim=1)
        x = self.final(x)
        # t2 = time.time()
        # embed()
        coord = soft_argmax(x, self.joint_num)
        # coord = coord[:17]
        # restore coordinates to original space
        # embed()
        coord[:, :, 0] = coord[:, :, 0] / 64. * (target['bbox'].view(-1, 1, 4)[:, :, 2]) + target['bbox'].view(-1, 1,
                                                                                                               4)[:, :,
                                                                                           0]
        coord[:, :, 1] = coord[:, :, 1] / 64. * (target['bbox'].view(-1, 1, 4)[:, :, 3]) + target['bbox'].view(-1, 1,
                                                                                                               4)[:, :,
                                                                                           1]

        coord_abs = coord.clone()
        coord_abs = coord_abs[:, :, :2]

        coord[:, :, :2] = coord[:, :, :2] - coord[:, :1, :2]
        coord[:, :, 2] = (coord[:, :, 2] / 64. * 2. - 1) * 1000.
        coord[:, 0, 2] = coord[:, 0, 2] - coord[:, 0, 2]

        # embed()
        coord = torch.cat([coord_abs, coord], dim=2)
        # embed()
        coord = coord.view(-1, 90)
        return coord


class res_fc(nn.Module):
    def __init__(self):
        super(res_fc, self).__init__()
        self.fc1 = nn.Linear(1024, 1024, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0)
        self.fc2 = nn.Linear(1024, 1024, bias=False)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0)
        self.fc3 = nn.Linear(1024, 1024, bias=False)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        input = self.dropout2(self.relu(self.bn2(self.fc2(input))))
        input = self.dropout3(self.relu(self.bn3(self.fc3(input))))

        x = input + x

        return x


class CoordinateNet(nn.Module):
    def __init__(self, posenet):
        super(CoordinateNet, self).__init__()
        self.posenet = posenet
        self.fc1 = nn.Linear(90, 1024, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = res_fc()
        self.block2 = res_fc()
        self.block3 = res_fc()
        self.block4 = res_fc()
        self.block5 = res_fc()
        self.fc2 = nn.Linear(1024, 18 * 3)

    def forward(self, input_img, target=None):
        uvz = self.posenet(input_img, target)

        x = self.relu(self.bn1(self.fc1(uvz)))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc2(x)

        return x

    def init_weight(self):
        return


def get_pose_net(cfg, is_train, joint_num):
    backbone = ResNetBackbone(cfg.resnet_type)
    up = Upsampling_Net(joint_num)
    sel = Serial_Net(joint_num)
    limb = limbHeatmap_Net(joint_num)
    final = Final_block(joint_num)
    if is_train:
        backbone.init_weights()
        up.init_weights()
        sel.init_weights()
        limb.init_weights()
        final.init_weights()

    model = ResPoseNet(backbone, up, limb, sel, final, joint_num)
    return model


def get_full_net(cfg, joint_num):
    posenet = get_pose_net(cfg, False, joint_num)
    model = CoordinateNet(posenet)
    return model


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


if __name__ == '__main__':
    model = get_full_net(cfg, 18)
    input = torch.randn((64, 3, 256, 256))
    print(model(input).shape)
