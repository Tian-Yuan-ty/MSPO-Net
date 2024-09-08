import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

fts = torch.rand((1,1,1,512,53,53))
prototype=[torch.rand((1,512))]
pred =torch.rand((1,1,53,53))
lable=torch.rand((1,1,1,417,417))

criterion = nn.NLLLoss()
bce_loss = nn.BCELoss()

prototype_ = Parameter(torch.stack(prototype, dim=0))
optimizer = torch.optim.Adam([prototype_], lr=0.01)

with ((torch.enable_grad())):
     pred_mask0 = torch.sum(pred, dim=-3) # tensor(1,53,53)
     pred_mask1 = torch.stack((1.0 - pred_mask0, pred_mask0), dim=1).argmax(dim=1, keepdim=True)# tensor(1,1,53,53)
     print(pred_mask1.grad_fn)
     pred_mask = pred_mask1.repeat([*fts.shape[1:-2], 1, 1]) # tensor(1,512,53,53)
     print(pred_mask.grad_fn)
     bg_fts = fts[0] * (1 - pred_mask)
     print(bg_fts.grad_fn)
     fg_fts = torch.zeros_like(fts[0])
     print(fg_fts.grad_fn)
     for way in range(1):
             fg_fts += prototype_[way].unsqueeze(-1).unsqueeze(-1).repeat(*pred.shape) \
                             * pred_mask[way][None, ...]
     new_fts = bg_fts + fg_fts
     print(new_fts.grad_fn)
     fts_norm = torch.sigmoid((fts[0] - fts[0].min()) / (fts[0].max() - fts[0].min()))
     print(fts_norm.grad_fn)
     new_fts_norm = torch.sigmoid((new_fts - new_fts.min()) / (new_fts.max() - new_fts.min()))
     print(new_fts_norm.grad_fn)
     loss = bce_loss(fts_norm, new_fts_norm)
     print(loss)
     print(loss.grad_fn)

     pred_up = F.interpolate(pred, size=lable.size()[-2:], mode='bilinear', align_corners=True) # tensor(1,1,53,53) to tensor(1,1,417,417)
     pred_up_0 = torch.stack((1.0 - pred_up, pred_up), dim=1) # tensor(1, 2, 1, 417, 417)
     mask_fg = pred_up_0.argmax(dim=1).squeeze(0) # tensor(1, 417, 417)
     masks = torch.stack((1.0 - mask_fg, mask_fg), dim=1) # tensor(1,2,417, 417)
     gt_lable = lable.squeeze(0).squeeze(0)# tensor(1, 417, 417)
     eps = torch.finfo(torch.float32).eps
     log_prob = torch.log(torch.clamp(masks, eps, 1 - eps))
     loss2 = criterion(log_prob, gt_lable.long())
     print(loss2)

'''
with ((torch.enable_grad())):
    pred_up = F.interpolate(pred, size=lable.size()[-2:], mode='bilinear', align_corners=True)
    pred_mask_fg = pred_up.squeeze(0)
    pred_mask = torch.stack((1.0 - pred_mask_fg, pred_mask_fg), dim=1)
    gt_lable = lable.squeeze(0).squeeze(0)
    eps = torch.finfo(torch.float32).eps
    log_prob = torch.log(torch.clamp(pred_mask, eps, 1 - eps))
    loss = criterion(log_prob, gt_lable.long())
    # loss = bce_loss(pred_mask_fg, gt_lable)
    print(loss)
    loss = loss.requires_grad_()
'''
#optimizer.zero_grad()
#loss2.backward()
#optimizer.step()
#print(torch.equal(prototype_[0], prototype[0]))
