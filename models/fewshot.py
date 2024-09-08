from collections import OrderedDict
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import Encoder
from torch.nn.parameter import Parameter

class FewShotSeg(nn.Module):

    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}

        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)), ]))

        self.device = torch.device('cuda')
        # self.alpha = torch.Tensor([0.8, 0.2])
        # self.alpha = torch.Tensor([1.0])
        self.alpha = torch.Tensor([0.1, 0.8, 0.1])
        self.criterion = nn.NLLLoss()

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)

        img_fts, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:n_ways * n_shots * batch_size].view(
            batch_size, n_ways, n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        qry_fts = [img_fts[dic][n_ways * n_shots * batch_size:].view(
            batch_size, n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        supp_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask],
                                dim=0).view(batch_size, n_ways, n_shots, *img_size)

        self.t = tao[n_ways * n_shots * batch_size:]
        self.thresh_pred = [self.t for _ in range(n_ways)]

        self.t_ = tao[:n_ways * n_shots * batch_size]
        self.thresh_pred_supp = [self.t_ for _ in range(n_ways)]

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[[self.getFeatures(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])
                             for shot in range(n_shots)] for way in range(n_ways)] for n in
                           range(len(supp_fts))]
            fg_prototypes = [self.getFGPrototype(supp_fg_fts[n]) for n in
                             range(len(supp_fts))]

            ###### Get query predictions ######
            qry_pred = [torch.stack(
                [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])
                 for way in range(n_ways)], dim=1) for n in
                range(len(qry_fts))]

            ###### Prototype Optimization  (only for test) ######
            update_iters = 100
            if (not self.training) and update_iters > 0:
                opt_fg_prototypes = []
                opt_fg_prototypes_Set = []
                for n in range(len(supp_fts)):
                    opt_fg_prototypes_set = self.optimizingPrototype(supp_fts[n], fg_prototypes[n], supp_mask,
                                                            update_iters, epi)
                    opt_fg_prototypes_Set.append(opt_fg_prototypes_set)

                    opt_fg_prototypes.append([opt_fg_prototypes_set[100]])

                qry_pred = [torch.stack(
                    [self.getPred(qry_fts[n][epi], opt_fg_prototypes[n][way], self.thresh_pred[way]) for way in
                    range(n_ways)], dim=1) for n in range(len(qry_fts))]

                qry_pred_up = [F.interpolate(qry_pred[n], size=img_size, mode='bilinear', align_corners=True)
                        for n in range(len(qry_fts))]
                pred = [self.alpha[n] * qry_pred_up[n] for n in
                        range(len(qry_fts))]
                preds0 = torch.sum(torch.stack(pred, dim=0), dim=0) / torch.sum(self.alpha)
                preds = torch.cat((1.0 - preds0, preds0), dim=1)
                outputs.append(preds)

            if self.training:
            ###### Combine predictions of different feature maps ######
                qry_pred_up = [F.interpolate(qry_pred[n], size=img_size, mode='bilinear', align_corners=True)
                           for n in range(len(qry_fts))]
                pred = [self.alpha[n] * qry_pred_up[n] for n in
                        range(len(qry_fts))]
                preds0 = torch.sum(torch.stack(pred, dim=0), dim=0) / torch.sum(self.alpha)
                preds = torch.cat((1.0 - preds0, preds0), dim=1)
                outputs.append(preds)

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.NEWalignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                   [qry_fts[n][epi] for n in range(len(qry_fts))],
                                                   preds, supp_mask[epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])
        return output, align_loss / batch_size

    def optimizingPrototype(self, fts, prototype, lable, update_iters, epi):
        prototype_set = prototype[0].unsqueeze(0)
        prototype_ = Parameter(torch.stack(prototype, dim=0))

        optimizer = torch.optim.Adam([prototype_], lr=0.001)
        while update_iters > 0:
            with ((torch.enable_grad())):
                pred = torch.stack([self.getPred(fts[epi].squeeze(0), prototype_[0], self.thresh_pred_supp[0])],
                                   dim=1)
                pred_up = F.interpolate(pred, size=lable.size()[-2:], mode='bilinear',
                                        align_corners=True)
                pred_ups = torch.cat((1.0 - pred_up, pred_up), dim=1)
                gt_lable = lable.squeeze(0).squeeze(0)
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss = self.criterion(log_prob, gt_lable.long())

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            prototype_new = prototype_.data
            prototype_set = torch.cat((prototype_set, prototype_new), dim=0)
            update_iters += -1

        return prototype_set

    def getPred(self, fts, prototype, thresh):
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * 20
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))
        return pred

    def getFeatures(self, fts, mask):
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
                     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
        return masked_fts

    def getFGPrototype(self, fg_fts):
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]

        return fg_prototypes

    def NEWalignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()

        # Compute the support loss
        # loss = torch.zeros(1).to(self.device)
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getFGPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                # Get predictions
                supp_pred0 = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred_supp[way])
                              for n in range(len(supp_fts))]

                supp_pred = [F.interpolate(supp_pred0[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]

                # Combine predictions of different feature maps
                preds = [self.alpha[n] * supp_pred[n] for n in range(len(supp_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways
        return loss
