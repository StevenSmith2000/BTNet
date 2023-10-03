from easydict import EasyDict as edict

config = edict()  #can obtain the value by attribute
config.loss = "curricularface"
config.network = "r50"
config.resume = False
config.output = 'output/ms1mv3_r50_reso112'
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.2
config.verbose = 2000
config.dali = False

config.rec = "/dataset/ms1m-retinaface-t1"  #dataset to ms1mv3
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 25
config.warmup_epoch = 2
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.save_feat = False

# Model
config.resolution = 112
config.pretrained = False
config.pretrained_path = None
# Update settings
config.fix_classifier = False #True: BCT training
config.fix_trunk = False  #True: update branches only
# Dataset
config.upsample = True # True: upsample to 112 after downsampling
config.load_type = 1 # 0:fixed size 112   1:random size in [112,56,28,14,7]
