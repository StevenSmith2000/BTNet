from easydict import EasyDict as edict

config = edict()  #can obtain the value by attribute
config.loss = "curricularface"
config.network = "r50"
config.resume = False
config.output = 'output/ms1mv3_r50_reso28'
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.02
config.verbose = 2000
config.dali = False

config.rec = "/dataset/ms1m-retinaface-t1"  #dataset to ms1mv3
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 10
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
# Model
config.resolution = 28
config.pretrained = True
config.pretrained_path = 'output/ms1mv3_r50_reso112'
config.save_feat = False
# Update settings
config.fix_classifier = True #True: BCT training
config.fix_trunk = True #True: update branches only
# Dataset
config.upsample = False   # True: upsample to 112 after downsampling
config.load_type = 0  # 0:fixed size 112   1:random size in [112,56,28,14,7]
# Loss weight
config.feat_loss_weight = 1
config.embed_loss_weight = 1
config.kd_loss_weight = 0.5
# Distillation
config.distill_feat_index = 1
# 0: layer1
# 1: layer2
# 2: layer3
# 3: layer4
config.fix_params = ['layer2','layer3','layer4','fc','features']
