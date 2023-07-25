from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg


class CenterPivotConv4d(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        bsz, inch, ha, wa, hb, wb = x.size()
        out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()

        y = out1 + out2
        return y


class HPNLearner(nn.Module):
    def __init__(self, inch):
        super(HPNLearner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

        # Decoder layers
        # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())
        #
        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid):

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)


        # Decode the encoded 4D-tensor
        # hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        # logit_mask = self.decoder2(hypercorr_decoded)

        return hypercorr_encoded


def extract_feat_vgg(img, backbone, feat_ids, bottleneck_ids=None, lids=None):
    r""" Extract intermediate features from VGG """
    feats = []
    feat = img
    for lid, module in enumerate(backbone.features):
        feat = module(feat)
        if lid in feat_ids:
            feats.append(feat.clone())
    return feats


def extract_feat_res(img, backbone, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.conv1.forward(img)
    feat = backbone.bn1.forward(feat)
    feat = backbone.relu.forward(feat)
    feat = backbone.maxpool.forward(feat)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

    return feats


class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]


class HypercorrSqueezeNetwork(nn.Module):
    def __init__(self, backbone, grid_num, use_original_imgsize):
        super(HypercorrSqueezeNetwork, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.grid_num = grid_num
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.seg_feat_channels = 256
        reduce_dim = 256
        fea_dim = 1024 + 512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        cate_conv_num = 3
        self.mask_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()
        for i in range(cate_conv_num):
            in_dim = reduce_dim + 2 if i == 0 else self.seg_feat_channels
            self.mask_convs.append(nn.Sequential(
                nn.Conv2d(in_dim, self.seg_feat_channels, 3, stride=1, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=self.seg_feat_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.2)
            ))
            in_dim = 128 if i == 0 else self.seg_feat_channels
            self.cate_convs.append(nn.Sequential(
                nn.Conv2d(in_dim, self.seg_feat_channels, 3, stride=1, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=self.seg_feat_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.2)
            ))

        self.grid_conv = nn.Conv2d(self.seg_feat_channels, self.grid_num ** 2, 1)
        self.cls_conv = nn.Conv2d(self.seg_feat_channels, 2, kernel_size=1, stride=1, bias=True)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_img, support_img, support_mask):
        input_size = query_img.size()
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.mask_feature(support_feats, support_mask.clone())
            corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)

        query_feat_2 = query_feats[self.stack_ids[0]]
        query_feat_3 = query_feats[self.stack_ids[1]]
        query_feat_3 = F.interpolate(query_feat_3, (query_feat_2.size(2), query_feat_2.size(3)), mode='bilinear',
                                     align_corners=True)
        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        corr_query_mask = self.hpn_learner(corr)

        # mask branch
        x_range = torch.linspace(-1, 1, query_feat.shape[-1], device=query_feat.device)
        y_range = torch.linspace(-1, 1, query_feat.shape[-2], device=query_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([query_feat.shape[0], 1, -1, -1])
        x = x.expand([query_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        query_mask_feat = torch.cat([query_feat, coord_feat], 1)
        for i, mask_layer in enumerate(self.mask_convs):
            query_mask_feat = mask_layer(query_mask_feat)
        query_mask_feat = F.interpolate(query_mask_feat, scale_factor=2, mode='bilinear', align_corners=True)
        query_mask_feat = self.grid_conv(query_mask_feat)

        # cate branch
        h, w = corr_query_mask.size()[2:][0], corr_query_mask.size()[2:][1]
        for i, cate_layer in enumerate(self.cate_convs):
            if i == 0:
                query_cate_feat = F.interpolate(corr_query_mask, size=self.grid_num, mode='bilinear',
                                                align_corners=True)
            query_cate_feat = cate_layer(query_cate_feat)
        query_cate_feat = self.cls_conv(query_cate_feat)
        query_cate_feat = query_cate_feat.view(input_size[0], 2, -1)
        query_cate_feat = F.softmax(query_cate_feat, dim=1)
        query_mask_feat = query_mask_feat.sigmoid()
        out = torch.einsum("bcq, bqhw->bchw", query_cate_feat, query_mask_feat)

        if not self.use_original_imgsize:
            out = F.interpolate(out, support_img.size()[2:], mode='bilinear', align_corners=True)

        return out, query_cate_feat, query_mask_feat

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])

            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1: return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, logit_cate, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_cate = nn.AdaptiveAvgPool2d((self.grid_num, self.grid_num))(gt_mask).view(bsz, 1, -1)
        gt_mask = gt_mask.view(bsz, -1)
        mask_loss = self.cross_entropy_loss(logit_mask, gt_mask.long())
        gt_cate = (gt_cate - gt_cate.min(2)[0].unsqueeze(1)) / (
                gt_cate.max(2)[0].unsqueeze(1) - gt_cate.min(2)[0].unsqueeze(1) + 1e-7)
        gt_background = torch.ones(gt_cate.size(), device=gt_cate.device).view(bsz, 1, -1).float() - gt_cate
        gt_cate = torch.cat([gt_background, gt_cate], dim=1)
        cate_loss = (-gt_cate * torch.log(logit_cate)).sum(1).mean()

        return mask_loss + cate_loss

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging


if __name__ == "__main__":
    query_img = torch.rand(4, 3, 473, 473)
    query_mask = torch.randint(0, 2, (4, 473, 473)).float()
    support_img = torch.rand(4, 1, 3, 473, 473)
    support_mask = torch.rand(4, 1, 473, 473)
    # query_img = torch.rand(4, 3, 400, 400)
    # query_mask = torch.randint(0, 2, (4, 400, 400)).float()
    # support_img = torch.rand(4, 1, 3, 400, 400)
    # support_mask = torch.rand(4, 1, 400, 400)
    model = HypercorrSqueezeNetwork('resnet50', 12, False)
    out = model(query_img, support_img.squeeze(), support_mask.squeeze(1))
    print(out.shape)
