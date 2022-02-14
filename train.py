# -*- coding:utf-8 -*-

"""
@author:gz
@file:train.py
@time:2021/2/1116:08
"""
import os
import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim, nn
from dataloader import MyDataset
from models.sr_transformer import SRVisionTransformer
from losses.loss import BinaryDiceLoss
from einops import rearrange
import torch.nn.functional as F
import config

def mkdirs(path):
    if not os.path.exists(path): os.makedirs(path)

def dice_metric(output, target):
    output = (output > 0).float()
    dice = ((output * target).sum() * 2 + 0.01) / (output.sum() + target.sum() + 0.01)
    return dice


def acc_m(output, target):
    output = (output > 0).float()
    target, output = target.view(-1), output.view(-1)
    acc = (target == output).sum().float() / target.shape[0]
    return acc


def sen_m(output, target):
    output = (output > 0).float()
    target, output = target.view(-1), output.view(-1)
    p = (target * output).sum().float()
    sen = (p + 0.01) / (output.sum() + 0.01)
    return sen


def spe_m(output, target):
    output = (output > 0).float()
    target, output = target.view(-1), output.view(-1)
    tn = target.shape[0] - (target.sum() + output.sum() - (target * output).sum().float())
    spe = (tn + 0.01) / (target.shape[0] - output.sum() + 0.01)
    return spe

def voe_metric(output, target):
    output = (output > 0).float()
    voe = ((output.sum() + target.sum() - (target * output).sum().float() * 2) + 0.01) / (
    output.sum() + target.sum() - (target * output).sum().float() + 0.01)
    return voe.item()


def rvd_metric(output, target):
    output = (output > 0).float()
    rvd = ((output.sum() + 0.01)/ (target.sum() + 0.01) - 1)
    return rvd.item()

def dice_loss(output, target):
    output = torch.sigmoid(output).float()
    loss = 1 - ((output * target).sum() * 2 + 0.01) / (output.sum() + target.sum() + 0.01)
    return loss

def load_checkpoint_model(model, ckpt_best, device):
    state_dict = torch.load(ckpt_best, map_location=device)
    # model.load_state_dict(state_dict['state_dict'])
    model.load_state_dict(state_dict)
    return model

def train_epoch(epoch, model, device, dl, optimizer, criterion, criterion2):
    model.train()
    bar = tqdm(dl)
    bar.set_description_str("%02d" % epoch)
    loss_v, dice_v, loss_pre, loss_sur, ii = 0, 0, 0, 0, 0
    diceloss = BinaryDiceLoss(p=1, reduction='mean')
    for x2, mask in bar:
        x2 = rearrange(x2, 'b (n h) (m w) c -> (b n m) c h w', c=3, n=8, m=8, h=64, w=64).float()
        x2_tar = torch.mean(x2, dim=1).unsqueeze_(1)
        sur, outputs = model(x2.to(device))

        mask = rearrange(mask, 'b (n h) (m w) -> (b n m) 1 h w', n=8, m=8, h=64, w=64)
        mask = mask.float().to(device)
        loss2 = criterion2(sur,x2_tar.to(device))
        loss3 = diceloss(outputs,mask)#dice_loss(outputs,mask)
        loss1 = criterion(outputs, mask)
        loss = 0.01*loss3+0.01*loss2+0.98*loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dice = dice_metric(outputs, mask)
        dice_v += dice
        loss_v += loss3.item()
        loss_pre += loss1.item()
        loss_sur += loss2.item()
        ii += 1
        bar.set_postfix(loss=loss.item(), dice=dice)
    return loss_v / ii, dice_v / ii, loss_pre / ii, loss_sur / ii


@torch.no_grad()
def val_epoch(model,device, dl, criterion):
    model.eval()
    loss_v, dice_v, voe_v, rvd_v, acc_v, sen_v, spe_v, ii = 0, 0, 0, 0, 0, 0, 0, 0
    for x2, mask in dl:
        x2 = rearrange(x2, 'b (n h) (m w) c -> (b n m) c h w', c=3, n=8, m=8, h=64, w=64).float()
        sur, outputs = model(x2.to(device))
        mask = rearrange(mask, 'b (n h) (m w) -> (b n m) 1 h w', n=8, m=8, h=64, w=64)
        mask = mask.float().to(device)

        loss_v += criterion(outputs, mask).item()
        dice_v += dice_metric(outputs, mask)
        voe_v += voe_metric(outputs, mask)
        rvd_v += rvd_metric(outputs, mask)
        acc_v += acc_m(outputs, mask)
        sen_v += sen_m(outputs, mask)
        spe_v += spe_m(outputs, mask)
        ii += 1
    return loss_v / ii, dice_v / ii, voe_v / ii, rvd_v / ii, acc_v / ii, sen_v / ii, spe_v / ii


def train(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(opt.seed)

    model = SRVisionTransformer(
        patch_h=opt.patch_h,
        patch_w=opt.patch_w,
        emb_dim=opt.emb_dim,
        mlp_dim=opt.mlp_dim,
        num_heads=opt.num_heads,
        num_layers=opt.num_layers,
        in_channels=opt.channels,
        attn_dropout_rate=opt.attn_dropout_rate,
        dropout_rate=opt.dropout_rate)
    if opt.w:
        model.load_state_dict(torch.load(opt.w))

    model = model.to(device)
    model = nn.DataParallel(model)

    root_dir = opt.dataset_path
    train_image_root = 'train'
    val_image_root = 'vali'

    train_dataset = MyDataset(model_type=train_image_root, data_filename=root_dir)
    train_dl = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)

    val_dataset = MyDataset(model_type=val_image_root, data_filename=root_dir)
    val_dl = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    optimizer = optim.Adam(params=model.parameters(), lr=opt.lr)
    criterion = nn.BCEWithLogitsLoss()
    criterion2 = nn.MSELoss()
    # log information
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, factor=0.1, patience=10)
    best_dice_epoch, best_dice, b_voe, b_rvd, train_loss, train_dice, b_acc, b_sen, b_spe = 0, 0, 0, 0, 0, 0, 0, 0, 0
    pre_loss, sur_loss = 0, 0
    save_dir = os.path.join(opt.ckpt, datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + "_" + opt.name
    mkdirs(save_dir)
    w_dice_best = os.path.join(save_dir, 'w_dice_best.pth')

    fout_log = open(os.path.join(save_dir, 'log.txt'), 'w')
    print(len(train_dataset), len(val_dataset), save_dir)

    for epoch in range(opt.max_epoch):
        if not opt.eval:
            train_loss, train_dice, pre_loss, sur_loss = train_epoch(epoch, model,device, train_dl, optimizer, criterion,
                                                                     criterion2)
        val_loss, val_dice, voe_v, rvd_v, acc_v, sen_v, spe_v = val_epoch(model,device, val_dl, criterion)
        if best_dice < val_dice:
            best_dice, best_dice_epoch, b_voe, b_rvd, b_acc, b_sen, b_spe = val_dice, epoch, voe_v, rvd_v, acc_v, sen_v, spe_v
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), w_dice_best)

        lr = optimizer.param_groups[0]['lr']
        log = "%02d train_loss:%0.3e, train_dice:%0.5f,pre_loss:%0.3e,sur_loss:%0.3e, val_loss:%0.3e, val_dice:%0.5f, lr:%.3e\n best_dice:%.5f, voe:%.5f, rvd:%.5f, acc:%.5f, sen:%.5f, spe:%.5f(%02d)\n" % (
            epoch, train_loss, train_dice, pre_loss, sur_loss, val_loss, val_dice, lr, best_dice, b_voe, b_rvd, b_acc,
            b_sen, b_spe, best_dice_epoch)
        print(log)
        fout_log.write(log)
        fout_log.flush()
        scheduler.step(val_loss)

    fout_log.close()


if __name__ == '__main__':
    parser = config.args
    train(parser)
