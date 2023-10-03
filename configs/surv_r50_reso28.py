from easydict import EasyDict as edict

config = edict()  #can obtain the value by attribute
config.loss = "curricularface"
config.network = "r50"
config.resume = False
config.output = 'output/surv_r50_reso28'
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.02
config.verbose = 2000
config.dali = False

config.rec = '/datasets/survface' #path to QMUL-SurvFace
config.num_classes = 5319
config.num_image = 220890
config.num_epoch = 10
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.save_feat = False
# Model
config.resolution = 28
config.pretrained = True
config.pretrained_path = 'output/surv_r50_reso112'
# Update settings
config.fix_classifier = True
config.fix_trunk = True
# Dataset
config.upsample = False
config.load_type = 0
# Loss weight
config.feat_loss_weight = 1
config.embed_loss_weight = 1
config.kd_loss_weight = 0.5
# Distillation
config.distill_feat_index = 1
config.fix_params = ['layer2','layer3','layer4','fc','features']
