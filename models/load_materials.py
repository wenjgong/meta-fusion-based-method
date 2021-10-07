from __future__ import print_function
import torch
import numpy as np
print(torch.__version__)
import torch.utils.data
import torchvision.transforms as transforms
from models import DebinMeng_train

cate2label = {'CK+':{0: 'Happy', 1: 'Angry', 2: 'Fear',
                     'Angry': 1,'Fear': 2,'Happy': 0},
              'SMIC':{0: 'surprise', 1: 'positive', 2: 'negative',
                     'positive': 1,'negative': 2,'surprise': 0},

              'casme2':{0: 'surprise', 1: 'repression', 2: 'happiness', 3: 'disgust', 4: 'others',
                     'surprise': 0,'repression': 1,'happiness': 2,'disgust': 3,'others': 4},

              'casme2-4':{0: 'surprise', 1: 'repression', 2: 'happiness', 3: 'disgust',
                     'surprise': 0,'repression': 1,'happiness': 2,'disgust': 3},

              'casme': {0: 'disgust', 1: 'surprise', 2: 'repression', 3: 'tense',
                         'disgust': 0, 'surprise': 1, 'repression': 2, 'tense': 3}}
cate2label = cate2label['casme2-4']


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ep_per_batch=1):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.ep_per_batch = ep_per_batch

        label = np.array(label)
        self.catlocs = []
        for c in range(max(map(int,label)) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                classes = np.random.choice(len(self.catlocs), self.n_cls,
                                           replace=False)
                #for c in classes:
                for c in range(len(self.catlocs)):
                    l = np.random.choice(self.catlocs[c], self.n_per,
                                         replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch)  # bs * n_cls * n_per
            yield batch.view(-1)

def LoadAFEW(root_train, list_train, root_eval, list_eval):
    norm_params = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
    normalize = transforms.Normalize(**norm_params)
    default_transform = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        #normalize,
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = DebinMeng_train.TripleImageDataset(
        video_root=root_train,
        video_list=list_train,
        rectify_label=cate2label,
        transform=default_transform,
        kuochong=True
    )

    val_dataset = DebinMeng_train.TripleImageDataset(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cate2label,
        transform=default_transform,
        kuochong=False)

    n_train_way=4
    n_train_shot=5
    n_query=1
    train_sampler = CategoriesSampler(
        train_dataset.index, 30,
        n_train_way, n_train_shot + n_query,
        ep_per_batch=4)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    sup_sampler=CategoriesSampler(
        train_dataset.index, 1,
        n_train_way, n_train_shot,
        ep_per_batch=1)
    sup_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sup_sampler,
                                             num_workers=8, pin_memory=True)
    val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=val_dataset.__len__(),
                                               num_workers=8, pin_memory=True)
    '''
    val_sampler = CategoriesSampler(
        val_dataset.index, 8,
        n_train_way, n_train_shot + n_query,
        ep_per_batch=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler,
                                               num_workers=8, pin_memory=True)
    '''
    '''
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=0, pin_memory=True)
    '''

    return train_loader, sup_loader, val_loader

def LoadParameter(_structure, _parameterDir):

    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):

            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()

    return model
