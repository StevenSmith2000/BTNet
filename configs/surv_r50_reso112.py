from easydict import EasyDict as edict

config = edict()  #can obtain the value by attribute
config.loss = "curricularface"
config.network = "r50"
config.resume = False
config.output = 'output/surv_r50_reso112'
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
config.resolution = 112
config.pretrained = True
config.pretrained_path = 'output/ms1mv3_r50_reso112'
# Update settings
config.fix_classifier = False
config.fix_trunk = False
# Dataset
config.upsample = False # True: upsample to 112 after downsampling
config.load_type = 0 # 0:fixed size 112   1:random size in [112,56,28,14,7]

