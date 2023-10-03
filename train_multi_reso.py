import argparse
import logging
import os

import torch
from torch import distributed
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from dataset import get_dataloader
from torch.utils.data import DataLoader
from lr_scheduler import PolyScheduler
from losses import CosFace, ArcFace, CurricularFace
from partial_fc import PartialFC
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_features import extract_feature, save_feat


try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    torch.cuda.set_device(args.local_rank)
    cfg = get_config(args.config)

    os.makedirs(cfg.output, exist_ok=True)  #make dir according to the output path set in cfg
    init_logging(rank, cfg.output)
    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )   #only for rank=0 machine

    train_loader = get_dataloader(
        cfg.rec, local_rank=args.local_rank, batch_size=cfg.batch_size, dali=cfg.dali, test=False, resolution=cfg.resolution, upsample=cfg.upsample, load_type=cfg.load_type)

    backbone = get_model(
        cfg.network, cfg.resolution,  dropout = 0.0, fp16 = cfg.fp16, num_features = cfg.embedding_size ).cuda()

    if cfg.pretrained and rank == 0:
        for load_path in os.listdir(cfg.pretrained_path):
            if load_path.startswith('model'):
                print("Start loading from %s"%load_path)
                weights = torch.load(os.path.join(cfg.pretrained_path, load_path))
                backbone.load_state_dict(weights, strict=False)

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank]) #,find_unused_parameters=True)
    
    if cfg.fix_trunk:
        fix_para = cfg.fix_params
        for name, param in backbone.named_parameters():
            for i, fix in enumerate(fix_para):
                if fix in name and 'bn' not in name:
                    param.requires_grad = False
                if i==0 and fix+'.0' in name:
                    param.requires_grad = True
        for name, param in backbone.named_parameters():
            if param.requires_grad == False:
                print(name)
    
    backbone.train()

    if cfg.loss == "arcface":
        margin_loss = ArcFace()
    elif cfg.loss == "cosface":
        margin_loss = CosFace()
    elif cfg.loss == "curricularface":
        margin_loss = CurricularFace()
    else:
        raise

    module_partial_fc = PartialFC(
        margin_loss,
        cfg.embedding_size, 
        cfg.num_classes, 
        cfg.sample_rate, 
        cfg.fp16
    )
    
    if cfg.pretrained:
        for load_path in os.listdir(cfg.pretrained_path):
            if load_path.startswith('softmax') and str(rank) == load_path[15]:
                print("Start loading from %s"%load_path)                  
                weights = torch.load(os.path.join(cfg.pretrained_path, load_path))
                module_partial_fc.load_state_dict(weights, strict=False)
    
    module_partial_fc.train().cuda()

    if cfg.fix_classifier:
        opt_params=[{"params":backbone.parameters(),},]
        if rank == 0:
            print("classifier is fixed !")
        if cfg.fix_trunk:
            opt_params=[{"params":filter(lambda p: p.requires_grad, backbone.parameters()),},]
            
    else:
        opt_params=[
            {"params": backbone.parameters(), },
            {"params": module_partial_fc.parameters(), },
        ]
    opt = torch.optim.SGD(
        params=opt_params,
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay
    )
    total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // total_batch_size * cfg.num_epoch
    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step
    )

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer, image_size = (cfg.resolution, cfg.resolution)
    )

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        writer=summary_writer
    )

    if cfg.save_feat:
        print("Start saving features in the training sets")
        train_loader = get_dataloader(cfg.rec, local_rank=args.local_rank, batch_size=cfg.batch_size, dali=cfg.dali, test=True)
        save_feat(backbone, train_loader, cfg.output)
        print("Finish saving features")
        return

    loss_am = AverageMeter()
    start_epoch = 0
    global_step = 0
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)[0]
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels, opt)
            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step()
            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 200:
                    callback_verification(global_step, backbone)
        
        path_pfc = os.path.join(cfg.output, "softmax_fc_gpu_{}.pt".format(rank))
        torch.save(module_partial_fc.state_dict(), path_pfc)
        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            # TODO: filter out the trunk params before saving the branches
            torch.save(backbone.module.state_dict(), path_module)
        
        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed BTNet Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
