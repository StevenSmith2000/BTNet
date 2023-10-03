# Evaluation on QMUL-SurvFace 1:N identification with BTNet
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import auc
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
    def __init__(self, set_arg, transform, meta_path, img_size=112, reso=112, branch_select='ceil', reso_indicator='max'):
        self.set_arg = set_arg
        self.transform = transform
        self.img_size = img_size
        self.img_files = []
        self.labels = []
        self.class_dict = {}
        self.reso_indicator = reso_indicator
        if branch_select == 'bottom':
            self.branches={
            112:[112,1000],
            28:[28,111],
            14:[14,27],
            7:[0,13]
            }
        elif branch_select == 'near':
            self.branches = {
            112: [71,1000],
            28: [22,70],
            14: [11,21],
            7: [0, 10]
            }
        elif branch_select == 'ceil':
            self.branches={
            112:[29,1000],
            28:[15,28],
            14:[8,14],
            7:[0,7]
            }
        else:
            raise NotImplementedError(f'The branch selection stragety {branch_select} is not supported.')

        self.reso = reso
        self.branch = self.branches[reso]

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

    def __getitem__(self, idx):      # Select branch conditioned on the input reso

        img = Image.open(self.img_files[idx])
        w,h = img.size
        if self.reso_indicator == 'max':
            reso = max(w,h)
        elif self.reso_indicator == 'avg':
            reso = (w+h)//2
        elif self.reso_indicator == 'min':
            reso = min(w,h)
        else:
            raise NotImplementedError(f'The resolution indicator {self.reso_indicator} is not supported.')
        if reso < self.branch[0] or reso > self.branch[1]:
            return (None,None)
        img = img.resize((self.reso,self.reso))
        img = self.transform(img)
        label = self.labels[idx]
        return img,label


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def dir_at_far(dir_tensor, far):
    idx = torch.argmin(torch.abs(dir_tensor[:,2]-far))  #find the dir which is closest to far
    return dir_tensor[idx, 1].item()    #known the far, find the corresponding dir

def my_collate(batch):
    if isinstance(batch, list):
        batch = [(image, image_id) for (image, image_id) in batch if image is not None]
    if batch==[]:
        return (None,None)
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='do survface 1:N test')
    parser.add_argument('--meta_path', default='/dataset/SurvFace/', help='path to SurvFace dataset.')
    parser.add_argument('--model', default='r50', type=str, help='')
    parser.add_argument('--weight112', default='output/surv_r50_reso112/model.pt', type=str, help='path to TNet(reso112)')
    parser.add_argument('--weight28', default='output/surv_r50_reso28/model.pt' , type=str, help='path to BNet(reso28)')
    parser.add_argument('--weight14', default='output/surv_r50_reso14/model.pt', type=str, help='path to BNet(reso14)')
    parser.add_argument('--weight7', default='output/surv_r50_reso7/model.pt', type=str, help='path to BNet(reso7)')
    parser.add_argument('--reso_indicator', default='max', choices=['max','avg','min'], help='get a resolution indicator from (H,W)')
    parser.add_argument('--branch_select', default='ceil', choices=['ceil', 'near', 'bottom'], help='strategy to select the branch')
    parser.add_argument('--batch_size', default=200, type=int,  help='')

    args = parser.parse_args()

    meta_path = args.meta_path
    #prepare evaluation set
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(torch.FloatTensor([0.5,0.5,0.5]),torch.FloatTensor([0.5,0.5,0.5]))

    ])

    reso_list=[112,28,14,7]
    Gsets = [QUML_evalset("G",trf, meta_path=meta_path, reso=i, branch_select=args.branch_select, reso_indicator=args.reso_indicator) for i in reso_list]
    Ksets = [QUML_evalset("K", trf, meta_path=meta_path, reso=i, branch_select=args.branch_select, reso_indicator=args.reso_indicator) for i in reso_list]
    Usets = [QUML_evalset("U", trf, meta_path=meta_path, reso=i, branch_select=args.branch_select, reso_indicator=args.reso_indicator) for i in reso_list]

    num_samples = [len(set(Gset.labels)) for Gset in Gsets]
    num_cls = num_samples[0]
    print("num_cls:", num_cls)
  
    Gloaders = [DataLoader(Gset, batch_size=args.batch_size, num_workers=4, collate_fn = my_collate) for Gset in Gsets]
    Kloaders = [DataLoader(Kset, batch_size=args.batch_size, num_workers=4, collate_fn = my_collate) for Kset in Ksets]
    Uloaders = [DataLoader(Uset, batch_size=args.batch_size, num_workers=4, collate_fn = my_collate) for Uset in Usets]

    #load model
    model = args.model

    weights = [args.weight112,        #112
               args.weight28,        #28
               args.weight14,        #14
               args.weight7        #7
                ]

    nets = [get_model(model, resolution=r, fp16=False) for r in reso_list]

    for weight, net in zip(weights,nets):
        net.to(device)
        net.load_state_dict(torch.load(weight))
        # extract features
        net.eval()

    with torch.no_grad():
        # make gallery prototypes: averaged features
        G_feat = torch.zeros(num_cls, 512).to(device)
        G_cardinality = torch.zeros(num_cls, dtype=torch.int64).to(device)
        for Gloader,net in zip(Gloaders,nets):
            for batch,(img,label) in enumerate(Gloader):
                
                if img is None or label is None:
                    continue
                img,label=img.to(device),label.to(device)
                feat = net(img)[0]
                for i in range(label.size(0)):   #None or do not have None
                    G_feat[label[i]] += feat[i]
                    G_cardinality[label[i]] += 1
        G_feat = torch.div(G_feat.T, G_cardinality).T
        print("G_feat:",G_feat.shape)
        # extract features of known probe set K

        flag_ = 0
        for Kloader,net in zip(Kloaders,nets):
            flag = 0
            for batch, (img, label) in enumerate(Kloader):
                if img is None or label is None:
                    continue
                img, label = img.to(device), label.to(device)
                if batch == 0 or flag == 0:
                    K_feat = net(img)[0]
                    K_label = label
                    flag = 1
                else:
                    K_feat = torch.cat((K_feat, net(img)[0]), dim=0)
                    K_label = torch.cat((K_label, label), dim=0)
            if flag_ == 0 and flag == 1:
                K_f= K_feat
                K_l=K_label
                flag_ = 1
                print("K_branch:",K_feat.shape)
            elif flag_ == 1 and flag == 1:
                K_f = torch.cat((K_f, K_feat), dim=0)
                K_l = torch.cat((K_l, K_label),dim=0)
                print("K_branch:",K_feat.shape)
            else:
                pass
            
        print("K_feat:", K_f.shape)
        # extract features of unknown probe set U
        flag_ = 0
        for Uloader,net in zip(Uloaders,nets):
            flag = 0
            for batch, (img, label) in enumerate(Uloader):
                if img is None or label is None:
                    continue
                img = img.to(device)
                if batch == 0 or flag == 0:
                    U_feat = net(img)[0]
                    flag = 1
                else:
                    U_feat = torch.cat((U_feat, net(img)[0]), dim=0)

            if flag_ == 0 and flag == 1:
                U_f= U_feat
                flag_ = 1
            elif flag_ == 1 and flag == 1:
                U_f = torch.cat((U_f, U_feat), dim=0)
            else:
                pass
        print("U_feat:", U_f.shape)
    #compute cosine similarity
    G_feat = F.normalize(G_feat, dim=1)
    K_feat = F.normalize(K_f, dim=1)
    U_feat = F.normalize(U_f, dim=1)
    K_sim = torch.mm(K_feat, G_feat.T)
    U_sim = torch.mm(U_feat, G_feat.T)

    K_val, pred = torch.topk(K_sim, k=20, dim=1) #top-20
    U_val, _ = torch.max(U_sim, dim=1)

    K_max,_ = torch.max(K_sim, dim=1)

    # compute DIR & FAR w.r.t. different thresholds
    corr_mask = pred.eq(K_l.view(-1,1))
    DIR_ = torch.zeros(1000,3)
    for i,th in enumerate(torch.linspace(min(K_val.min(),U_val.min()), U_val.max(), 1000)):
        mask = corr_mask & (K_val > th)
        dir_ = mask.sum().item()/K_feat.size(0)
        far_ = (U_val>th).sum().item()/U_feat.size(0)
        DIR_[i] = torch.FloatTensor([th, dir_, far_])

    print("auc:",auc(DIR_[:,2],DIR_[:,1]))

    for far in [0.01,0.1,0.2,0.3]:
        print("TPIR20 @ FAR={}: {:.2f}%".format(far,dir_at_far(DIR_, far)*100))
