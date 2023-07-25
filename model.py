import torch.nn as nn
import torch
import resnet as models
import torch.nn.functional as F

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class MANet(nn.Module):
    def __init__(self, backbone_layers=50, grid_num=12, shot=1, use_original_imgsize=False):
        super(MANet, self).__init__()
        self.grid_num = grid_num
        self.shot = shot
        if backbone_layers == 50:
            resnet = models.resnet50(pretrained=True)
        elif backbone_layers == 101:
            resnet = models.resnet101(pretrained=True)
        else:
            resnet = models.resnet152(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.use_original_imgsize = use_original_imgsize

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        reduce_dim = 256
        fea_dim = 1024 + 512
        self.seg_feat_channels = 256
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
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

            self.cate_convs.append(nn.Sequential(
                nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, 3, stride=1, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=self.seg_feat_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.2)
            ))

        self.grid_conv = nn.Conv2d(self.seg_feat_channels, self.grid_num ** 2, 1)
        self.sim_conv = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels * 2 + 1, self.seg_feat_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.cls_conv = nn.Conv2d(self.seg_feat_channels, 2, kernel_size=1, stride=1, bias=True)

    def forward(self, query_img, support_img, support_mask):
        input_size = query_img.size()

        # Query feature
        with torch.no_grad():
            query_feat_0 = self.layer0(query_img)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        # support feature
        supp_feat_list = []
        final_supp_list = []
        mask_list = []
        for i in range(self.shot):
            mask = support_mask[:, i, :, :].float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                support_feat_0 = self.layer0(support_img[:, i, :, :])
                support_feat_1 = self.layer1(support_feat_0)
                support_feat_2 = self.layer2(support_feat_1)
                support_feat_3 = self.layer3(support_feat_2)
                mask = F.interpolate(mask, size=(support_feat_3.size(2), support_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)

                support_feat_4 = self.layer4(support_feat_3 * mask)
                final_supp_list.append(support_feat_4)

            support_feat = torch.cat([support_feat_3, support_feat_2], 1)
            support_feat = self.down_supp(support_feat)
            support_feat = Weighted_GAP(support_feat, mask)
            supp_feat_list.append(support_feat)

        # average feature when shot>1
        if self.shot > 1:
            support_feat = supp_feat_list[0]
            for i in range(1, len(supp_feat_list)):
                support_feat += supp_feat_list[i]
            support_feat /= len(supp_feat_list)

        # calculate similarity
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                        similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]),
                                       mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear',
                                        align_corners=True)

        query_mask_feat = query_feat
        query_cate_feat = query_feat

        # mask branch
        x_range = torch.linspace(-1, 1, query_mask_feat.shape[-1], device=query_mask_feat.device)
        y_range = torch.linspace(-1, 1, query_mask_feat.shape[-2], device=query_mask_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([query_mask_feat.shape[0], 1, -1, -1])
        x = x.expand([query_mask_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        query_mask_feat = torch.cat([query_mask_feat, coord_feat], 1)
        for i, mask_layer in enumerate(self.mask_convs):
            query_mask_feat = mask_layer(query_mask_feat)
        query_mask_feat = F.interpolate(query_mask_feat, scale_factor=2, mode='bilinear', align_corners=True)
        query_mask_feat = self.grid_conv(query_mask_feat)

        # cate branch
        h, w = corr_query_mask.size()[2:][0], corr_query_mask.size()[2:][1]
        support_cate_feat_avg = support_feat.expand(-1, -1, h, w)
        print(h, w)
        query_cate_feat = torch.cat([query_cate_feat, support_cate_feat_avg, corr_query_mask], dim=1)
        query_cate_feat = self.sim_conv(query_cate_feat)
        for i, cate_layer in enumerate(self.cate_convs):
            if i == 0:
                query_cate_feat = F.interpolate(query_cate_feat, size=self.grid_num, mode='bilinear',
                                                align_corners=True)
            query_cate_feat = cate_layer(query_cate_feat)
        query_cate_feat = self.cls_conv(query_cate_feat)
        query_cate_feat = query_cate_feat.view(input_size[0], 2, -1)
        query_cate_feat = F.softmax(query_cate_feat, dim=1)
        query_mask_feat = query_mask_feat.sigmoid()
        out = torch.einsum("bcq, bqhw->bchw", query_cate_feat, query_mask_feat)
        if not self.use_original_imgsize:
            out = F.interpolate(out, input_size[-2:], mode='bilinear', align_corners=True)
        return out, query_cate_feat, query_mask_feat

    def train_mode(self):
        self.train()
        self.layer0.eval()
        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()
        self.layer4.eval()

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



if __name__ == "__main__":
    query_img = torch.rand(4, 3, 473, 473)
    query_mask = torch.randint(0, 2, (4, 473, 473)).float()
    support_img = torch.rand(4, 1, 3, 473, 473)
    support_mask = torch.rand(4, 1, 473, 473)
    model = MANet(50, 12, 1, False)
    out, _, _ = model(query_img, support_img, support_mask)
    print(out.size())