# Evaluation on QMUL-SurvFace 1:N identification with a single model
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

from backbones import get_model

device = torch.device("cuda:0")

#Training dataset
class QUML_trainset(Dataset):
    def __init__(self, transform, img_size=112):
        self.transform = transform
        self.img_size = img_size
        self.img_files = []
        self.labels = []
        self.class_dict = {}

        train_dir = "training_set"
        name_list = sorted(os.listdir(train_dir))
        ID = 0
        for name in name_list:
            name_dir = os.path.join(train_dir,name)
            img_list = os.listdir(name_dir)
            for img_name in img_list:
                img_dir = os.path.join(name_dir,img_name)
                self.img_files.append(img_dir)
                self.labels.append(ID)
            ID+=1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.open(self.img_files[idx]).resize((self.img_size,self.img_size))
        img = self.transform(img)
        return img,label

#Evaluation dataset
class QUML_evalset(Dataset):
    def __init__(self, set_arg, transform, meta_path, img_size=112):
        self.set_arg = set_arg
        self.transform = transform
        self.img_size = img_size
        self.img_files = []
        self.labels = []
        self.class_dict = {}

        if set_arg == "U":
            base = "Face_Identification_Test_Set/unmated_probe"
            base = os.path.join(meta_path, base)
            imgs = os.listdir(base)
            self.labels = [-1]*len(imgs)
            for img in imgs:
                self.img_files.append(os.path.join(base,img))
        else:
            if set_arg == "G":
                base = "Face_Identification_Test_Set/gallery"
                base = os.path.join(meta_path, base)
                meta = loadmat(meta_path+"Face_Identification_Test_Set/gallery_img_ID_pairs.mat")
                imgs = meta["gallery_set"].reshape(-1)
                labels = (meta["gallery_ids"]-1).reshape(-1).tolist()
            elif set_arg == "K":
                base = "Face_Identification_Test_Set/mated_probe"
                base = os.path.join(meta_path, base)
                meta = loadmat(meta_path+"Face_Identification_Test_Set/mated_probe_img_ID_pairs.mat")
                imgs = meta["mated_probe_set"].reshape(-1)
                labels = (meta["mated_probe_ids"]-1).reshape(-1).tolist()
            ID = -1
            for img,lab in zip(imgs,labels):
                if lab not in self.class_dict.keys():
                    ID += 1
                    self.class_dict[lab] = ID
                self.img_files.append(os.path.join(base,img[0]))
                self.labels.append(ID)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.open(self.img_files[idx]).resize((self.img_size,self.img_size))
        img = self.transform(img)
        return img,label

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def dir_at_far(dir_tensor, far):
    idx = torch.argmin(torch.abs(dir_tensor[:,2]-far))  #find the dir which is closest to far
    return dir_tensor[idx, 1].item()    #known the far, find the corresponding dir


if __name__ =="__main__":
    meta_path = '../../fr/SurvFace/'
    #prepare evaluation set
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(torch.FloatTensor([0.5,0.5,0.5]),torch.FloatTensor([0.5,0.5,0.5]))

    ])

    Gset = QUML_evalset("G",trf,meta_path=meta_path)
    Kset = QUML_evalset("K",trf,meta_path=meta_path)
    Uset = QUML_evalset("U",trf,meta_path=meta_path)

    num_cls = len(set(Gset.labels))
    print("num_cls:", num_cls)
    Gloader = DataLoader(Gset, batch_size=200, num_workers=4)
    Kloader = DataLoader(Kset, batch_size=200, num_workers=4)
    Uloader = DataLoader(Uset, batch_size=200, num_workers=4)

    #load model
    model = 'r50'
    weight = 'output/surv_r50_reso112/model.pt'
    net = get_model(model, resolution=112, fp16=False)
    net.to(device)
    net.load_state_dict(torch.load(weight))
    # extract features
    net.eval()
    with torch.no_grad():
        # make gallery prototypes: averaged features
        G_feat = torch.zeros(num_cls, 512).to(device)
        cardinality = torch.zeros(num_cls, dtype=torch.int64).to(device)
        for batch,(img,label) in enumerate(Gloader):
            img,label=img.to(device),label.to(device)
            feat = net(img)[0]
            for i in range(label.size(0)):
                G_feat[label[i]] += feat[i]
                cardinality[label[i]] += 1
        G_feat = torch.div(G_feat.T, cardinality).T
        print("G_feat:",G_feat.shape)
        # extract features of known probe set K
        for batch,(img,label) in enumerate(Kloader):
            img,label=img.to(device),label.to(device)
            if batch==0:
                K_feat = net(img)[0]
                K_label = label
            else:
                K_feat = torch.cat((K_feat, net(img)[0]), dim=0)
                K_label = torch.cat((K_label, label),dim=0)
        print("K_feat:",K_feat.shape)
        # extract features of unknown probe set U
        for batch,(img,label) in enumerate(Uloader):
            img = img.to(device)
            if batch==0:
                U_feat = net(img)[0]
            else:
                U_feat = torch.cat((U_feat, net(img)[0]), dim=0)
        print("U_feat:",U_feat.shape)
    #compute cosine similarity
    G_feat = F.normalize(G_feat, dim=1)
    K_feat = F.normalize(K_feat, dim=1)
    U_feat = F.normalize(U_feat, dim=1)
    K_sim = torch.mm(K_feat, G_feat.T)
    U_sim = torch.mm(U_feat, G_feat.T)

    K_val, pred = torch.topk(K_sim, k=20, dim=1) #top-20
    U_val, _ = torch.max(U_sim, dim=1)

    # compute DIR & FAR w.r.t. different thresholds
    corr_mask = pred.eq(K_label.view(-1,1))
    DIR_ = torch.zeros(1000,3)
    for i,th in enumerate(torch.linspace(min(K_val.min(),U_val.min()), U_val.max(), 1000)):
        mask = corr_mask & (K_val > th)
        dir_ = mask.sum().item()/K_feat.size(0)
        far_ = (U_val>th).sum().item()/U_feat.size(0)
        DIR_[i] = torch.FloatTensor([th, dir_, far_])

    for far in [0.01,0.1,0.2,0.3]:
        print("TPIR20 @ FAR={}: {:.2f}%".format(far,dir_at_far(DIR_, far)*100))
