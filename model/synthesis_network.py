import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'instance3D':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def discriminate(D, fake_pool, input_label, test_image, use_pool=False):
    input_concat = torch.cat((input_label, test_image.detach()), dim=1)
    if use_pool:
        fake_query = fake_pool.query(input_concat)
        return D.forward(fake_query)
    else:
        return D.forward(input_concat)
    
def weights_init3D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
def define_D(input_nc, ndf, n_layers_D, norm='instance3D', use_sigmoid=False, num_D=1, getIntermFeat=False,
                gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator3D(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init3D)
    return netD


def feature_loss(opt, ori_img, syn_img, pred_real, pred_fake, ext_discriminator):
    criterionFeat = torch.nn.L1Loss()

    if opt.dimension.startswith('2'):
        loss_G_GAN_Feat = 0
        D_weights = 1.0 / 2
        for i in range(2):
            for j in range(len(pred_fake[i]) - 1):
                loss_G_GAN_Feat += D_weights * \
                                   criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
        ori_img = ori_img.expand(-1, 3, -1, -1)
        syn_img = syn_img.expand(-1, 3, -1, -1)
        feat_resize = nn.Upsample(size=(224, 224))
        feat_res_real = ext_discriminator(feat_resize(ori_img))
        feat_res_fake = ext_discriminator(feat_resize(syn_img))
        loss_G_GAN_Feat_ext = 0
        vgg_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        for tmp_i in range(len(feat_res_fake)):
            loss_G_GAN_Feat_ext += criterionFeat(feat_res_real[tmp_i].detach(), feat_res_fake[tmp_i]) * vgg_weights[tmp_i]

        return loss_G_GAN_Feat, loss_G_GAN_Feat_ext
    elif opt.dimension.startswith('3'):
        loss_G_GAN_Feat = 0
        D_weights = 1.0 / 3
        for i in range(3):
            for j in range(len(pred_fake[i]) - 1):
                loss_G_GAN_Feat += D_weights * \
                                   criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
        ori_img = ori_img.expand(-1, 3, -1, -1, -1)
        syn_img = syn_img.expand(-1, 3, -1, -1, -1)
        feat_res_real = ext_discriminator(ori_img)
        feat_res_fake = ext_discriminator(syn_img)
        loss_G_GAN_Feat_ext = 0
        res_weights = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        feature_level = ['layer1', 'layer2', 'layer3', 'layer4']
        for tmp_i in range(len(feature_level)):
            loss_G_GAN_Feat_ext += criterionFeat(feat_res_real[feature_level[tmp_i]].detach(),
                                               feat_res_fake[feature_level[tmp_i]]) * res_weights[tmp_i]

        return loss_G_GAN_Feat, loss_G_GAN_Feat_ext
    else:
        raise NotImplementedError
    

    
class MultiscaleDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=96, n_layers=3, norm_layer=nn.BatchNorm3d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator3D, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator3D(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool3d(3, stride=2, padding=[1, 1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=96, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
        

import torch
from torchvision.models.video import r3d_18
from torchvision.models.feature_extraction import create_feature_extractor


class Res3D(torch.nn.Module):
    def __init__(self):
        super(Res3D, self).__init__()
        return_nodes = {
            'layer1.1.relu': 'layer1',
            'layer2.1.relu': 'layer2',
            'layer3.1.relu': 'layer3',
            'layer4.1.relu': 'layer4',
        }
        res_3d = r3d_18(pretrained=True)
        self.features = create_feature_extractor(res_3d, return_nodes=return_nodes)


    def forward(self, x):
        res = self.features(x)

        return res