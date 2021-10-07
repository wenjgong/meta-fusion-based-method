import argparse
import os
import yaml

import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import datasets
import models
import utils
import utils.few_shot as fs
import matplotlib.pyplot as plt
from datasets.samplers import CategoriesSampler

from torchvision import transforms
from models import DebinMeng_train
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
cate2label = cate2label['SMIC']


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(config):
    svname = args.name
    if svname is None:
        svname = 'meta_{}-{}shot'.format(
                config['train_dataset'], config['n_shot'])
        svname += '_' + config['model'] + '-' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag
    #print(svname)  meta_mini-imagenet-1shot_meta-baseline-resnet12
    save_path = os.path.join('./save', svname)
    #print(save_path)  ./save\meta_mini-imagenet-1shot_meta-baseline-resnet12
    #utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    #print(writer)
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    n_way, n_shot = config['n_way'], config['n_shot']  # 'n_way': 5, 'n_shot': 1
    n_query = config['n_query']   #'n_query': 15

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']  #'ep_per_batch': 4
    else:
        ep_per_batch = 1

    c1 = 0.0
    c2 = 0.0
    c3 = 0.0
    c4 = 0.0
    c5 = 0.0
    c = 0.0
    y_true = list()
    y_pred = list()
    #print(len(y_pred))
    random.seed(0)
    #chf=random.choice([1,2,3,4,5,6,8,9,11,12,13,14,15,18,19,20])
    chf = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26])
    for i in range(1,27):

        if i==1:
            i=chf
        elif i==chf:
            i=1

        '''
        if i==7 or i==10 or i==16 or i==17:
            continue
        '''


        if i==10 or i==18:
            continue



        '''
        if i==18:
            continue
        '''

        print("subject ",i)

        '''
        arg_rootTrain = './data1/all2'
        arg_listTrain = './data1/3class_train' + str(i) + '.txt'

        arg_rooteval = './data1/all2'
        arg_listeval = './data1/3class_eval' + str(i) + '.txt'
        '''


        '''
        arg_rootTrain = './data1/casme'
        arg_listTrain = './data1/casme_train' + str(i) + '.txt'

        arg_rooteval = './data1/casme'
        arg_listeval = './data1/casme_eval' + str(i) + '.txt'
        '''

        arg_rootTrain = './data1/all2'
        arg_listTrain = './data1/list_train' + str(i) + '.txt'

        arg_rooteval = './data1/all2'
        arg_listeval = './data1/list_eval' + str(i) + '.txt'


        '''
        arg_rootTrain = './data1/all2'
        arg_listTrain = './data1/5class_train'+ str(i) +'.txt'

        arg_rooteval = './data1/all2'
        arg_listeval = './data1/5class_eval'+ str(i) +'.txt'
        '''



        '''
        arg_rootTrain = './Data/HS'
        arg_listTrain = './Data/smic_train' + str(i) + '.txt'

        arg_rooteval = './Data/HS'
        arg_listeval = './Data/smic_eval' + str(i) + '.txt'
        '''




        setup_seed(4)
        train_loader, sup_loader, val_loader = models.load_materials.LoadAFEW(arg_rootTrain, arg_listTrain,
                                                                       arg_rooteval, arg_listeval)

        default_transform = transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
            # normalize,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset1 = DebinMeng_train.TripleImageDataset(
            video_root=arg_rootTrain,
            video_list=arg_listTrain,
            rectify_label=cate2label,
            transform=default_transform,
            kuochong=True
        )
        train_loader1 = DataLoader(train_dataset1, config['batch_size1'], shuffle=True,
                                   num_workers=8, pin_memory=True)
        val_dataset1 = DebinMeng_train.TripleImageDataset(
            video_root=arg_rooteval,
            video_list=arg_listeval,
            rectify_label=cate2label,
            transform=default_transform,
            kuochong=False
        )
        val_loader1 = DataLoader(val_dataset1, config['batch_size1'], shuffle=True,
                                   num_workers=8, pin_memory=True)


        '''
        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        default_transform = transforms.Compose([
            transforms.Resize((80,80)),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = ImageFolder("/home/gwj/zy/few-shot-meta-baseline-master/materials/train_casme2", transform=default_transform)
        utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            len(train_dataset.classes)))
        #if config.get('visualize_datasets'):  # 'visualize_datasets': True
        #    utils.visualize_dataset(train_dataset, 'train_dataset', writer)
        train_sampler = CategoriesSampler(
            train_dataset.targets, config['train_batches'],
            n_train_way, n_train_shot + n_query,
            ep_per_batch=ep_per_batch)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                  num_workers=8, pin_memory=True)
        '''
        # train
        '''
        train_dataset = datasets.make(config['train_dataset'],
                                      **config['train_dataset_args'])
        utils.log('train dataset: {} (x{}), {}'.format(
                train_dataset[0][0].shape, len(train_dataset),
                train_dataset.n_classes))
        if config.get('visualize_datasets'):  #'visualize_datasets': True
            utils.visualize_dataset(train_dataset, 'train_dataset', writer)
        train_sampler = CategoriesSampler(
                train_dataset.label, config['train_batches'],
                n_train_way, n_train_shot + n_query,
                ep_per_batch=ep_per_batch)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                  num_workers=8, pin_memory=True)
        '''
        # tval
        '''
        if config.get('tval_dataset'):
            tval_dataset = ImageFolder("/home/gwj/zy/few-shot-meta-baseline-master/materials/test_casme2",
                                        transform=default_transform)
            utils.log('tval dataset: {} (x{}), {}'.format(
                tval_dataset[0][0].shape, len(tval_dataset),
                len(tval_dataset.classes)))
            # if config.get('visualize_datasets'):  # 'visualize_datasets': True
            #    utils.visualize_dataset(train_dataset, 'train_dataset', writer)
            tval_sampler = CategoriesSampler(
                tval_dataset.targets, 10,
                n_way, n_shot + n_query,
                ep_per_batch=4)
            tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                    num_workers=8, pin_memory=True)
        else:
            tval_loader = None
        '''
        '''
        if config.get('tval_dataset'):
            tval_dataset = datasets.make(config['tval_dataset'],
                                         **config['tval_dataset_args'])
            utils.log('tval dataset: {} (x{}), {}'.format(
                    tval_dataset[0][0].shape, len(tval_dataset),
                    tval_dataset.n_classes))
            if config.get('visualize_datasets'):
                utils.visualize_dataset(tval_dataset, 'tval_dataset', writer)
            tval_sampler = CategoriesSampler(
                    tval_dataset.label, 200,
                    n_way, n_shot + n_query,
                    ep_per_batch=4)
            tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                     num_workers=8, pin_memory=True)
        else:
            tval_loader = None
        '''
        # val
        '''
        val_dataset = ImageFolder("/home/gwj/zy/few-shot-meta-baseline-master/materials/test_casme2",
                                   transform=default_transform)
        utils.log('val dataset: {} (x{}), {}'.format(
            val_dataset[0][0].shape, len(val_dataset),
            len(val_dataset.classes)))
        # if config.get('visualize_datasets'):  # 'visualize_datasets': True
        #    utils.visualize_dataset(train_dataset, 'train_dataset', writer)
        val_sampler = CategoriesSampler(
            val_dataset.targets, 10,
            n_way, n_shot + n_query,
            ep_per_batch=4)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                                 num_workers=8, pin_memory=True)
        '''
        '''
        val_dataset = datasets.make(config['val_dataset'],
                                    **config['val_dataset_args'])
        utils.log('val dataset: {} (x{}), {}'.format(
                val_dataset[0][0].shape, len(val_dataset),
                val_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(val_dataset, 'val_dataset', writer)
        val_sampler = CategoriesSampler(
                val_dataset.label, 200,
                n_way, n_shot + n_query,
                ep_per_batch=4)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                                num_workers=8, pin_memory=True)
        '''

        ########

        #### Model and optimizer ####

        '''
        #pre
        if config.get('load'):
            model_sv = torch.load(config['load'])
            model = models.load(model_sv)
        else:
            model = models.make(config['model1'], **config['model_args1'])

            if config.get('load_encoder1'):
                encoder = models.load(torch.load(config['load_encoder1'])).encoder
                model.encoder.load_state_dict(encoder.state_dict())
        optimizer, lr_scheduler = utils.make_optimizer1(
            model.parameters(),
            config['optimizer'], **config['optimizer_args1'])
        max_epoch = config['max_epoch1']
        max_va = 0.
        timer_used = utils.Timer()
        timer_epoch = utils.Timer()

        for epoch in range(1, max_epoch + 1):
            timer_epoch.s()
            aves_keys = ['tl', 'ta', 'vl', 'va']
            aves = {k: utils.Averager() for k in aves_keys}
            # train
            model.train()

            for data, label in tqdm(train_loader1, desc='train', leave=False):
                data, label = data.cuda(), label.cuda()
                logits = model(data)
                loss = F.cross_entropy(logits, label.long())
                acc = utils.compute_acc(logits, label.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                aves['tl'].add(loss.item())
                aves['ta'].add(acc)

                logits = None
                loss = None
            model.eval()
            for data, label in tqdm(val_loader1, desc='val', leave=False):
                data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    logits = model(data)
                    loss = F.cross_entropy(logits, label.long())
                    acc = utils.compute_acc(logits, label.long())

                aves['vl'].add(loss.item())
                aves['va'].add(acc)

            # post
            if lr_scheduler is not None:
                lr_scheduler.step()
            for k, v in aves.items():
                aves[k] = v.item()
            t_epoch = utils.time_str(timer_epoch.t())
            t_used = utils.time_str(timer_used.t())
            t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
            if epoch <= max_epoch:
                epoch_str = str(epoch)
            else:
                epoch_str = 'ex'
            log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                epoch_str, aves['tl'], aves['ta'])
            log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
            if epoch <= max_epoch:
                log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
            else:
                log_str += ', {}'.format(t_epoch)
            utils.log(log_str)

            if config.get('_parallel'):
                model_ = model.module
            else:
                model_ = model

            training = {
                'epoch': epoch,
                'optimizer': config['optimizer'],
                'optimizer_args': config['optimizer_args1'],
                'optimizer_sd': optimizer.state_dict(),
            }
            save_obj = {
                'file': __file__,
                'config': config,

                'model': config['model1'],
                'model_args': config['model_args1'],
                'model_sd': model_.state_dict(),

                'training': training,
            }
            if epoch <= max_epoch:
                torch.save(save_obj, os.path.join('./save/pre/smic', str(i) + 'epoch-last.pth'))
        '''




        #micro
        if config.get('load'):
            model_sv = torch.load(config['load'])
            model = models.load(model_sv)
        else:
            model = models.make(config['model'], **config['model_args'])

            if config.get('load_encoder'):
                encoder = models.load(torch.load(os.path.join('./save/pre/casme2-4', str(i) + 'epoch-last.pth'))).encoder
                #encoder = models.load(torch.load(config['load_encoder'])).encoder
                model.encoder.load_state_dict(encoder.state_dict())

        if config.get('_parallel'):
            model = nn.DataParallel(model)

        utils.log('num params: {}'.format(utils.compute_n_params(model)))

        optimizer, lr_scheduler = utils.make_optimizer(
                model.parameters(),
                config['optimizer'], **config['optimizer_args'])




        ########

        max_epoch = config['max_epoch']
        save_epoch = config.get('save_epoch')
        max_va = 0.
        timer_used = utils.Timer()
        timer_epoch = utils.Timer()

        aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
        trlog = dict()
        for k in aves_keys:
            trlog[k] = []

        train_loss_list=[]
        train_acc_list=[]
        test_loss_list=[]
        test_acc_list=[]
        for epoch in range(1, max_epoch + 1):
            timer_epoch.s()
            aves = {k: utils.Averager() for k in aves_keys}

            # train
            model.train()
            if config.get('freeze_bn'):
                utils.freeze_bn(model)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            np.random.seed(epoch)
            for data, _ in tqdm(train_loader, desc='train', leave=False):
                #print(data.shape)
                x_shot, x_query = fs.split_shot_query(
                        data.cuda(), n_train_way, n_train_shot, n_query,
                        ep_per_batch=ep_per_batch)
                label = fs.make_nk_label(n_train_way, n_query,
                        ep_per_batch=ep_per_batch).cuda()

                logits,xx_shot,xx_query = model(x_shot, x_query)
                logits=logits.view(-1, n_train_way)
                '''
                logits = model(x_shot, x_query).view(-1, n_train_way * 5)
                labels=torch.ones(label.size(0)).cuda()
                for i in range(label.size(0)):
                    max=0
                    ways=label[i]
                    for j in range(5):
                        if logits[i][ways*5+j]>max:
                            max=logits[i][ways*5+j]
                            nums=j
                    labels[i]=ways*5+nums
                loss = F.cross_entropy(logits, labels.long())
                '''
                '''
                logits = model(x_shot, x_query).view(-1, n_train_way * 5)
                labels1 = torch.ones(label.size(0)).cuda()
                labels2 = torch.ones(label.size(0)).cuda()
                labels3 = torch.ones(label.size(0)).cuda()
                labels4 = torch.ones(label.size(0)).cuda()
                labels5 = torch.ones(label.size(0)).cuda()
                for i in range(label.size(0)):

                    ways=label[i]
                    max1 = 0
                    max2=0
                    max3=0
                    max4=0
                    max5=0
                    nums1=0
                    nums2=0
                    nums3=0
                    nums4=0
                    nums5=0
                    for j in range(5):
                        if logits[i][ways*5+j]>max1:
                            max5=max4
                            nums5=nums4
                            max4=max3
                            nums4=nums3
                            max3=max2
                            nums3=nums2
                            max2=max1
                            nums2=nums1
                            max1=logits[i][ways*5+j]
                            nums1=j
                        else:
                            if logits[i][ways*5+j]>max2:
                                max5 = max4
                                nums5 = nums4
                                max4 = max3
                                nums4 = nums3
                                max3=max2
                                nums3=nums2
                                max2=logits[i][ways*5+j]
                                nums2=j
                            else:
                                if logits[i][ways*5+j]>max3:
                                    max5 = max4
                                    nums5 = nums4
                                    max4 = max3
                                    nums4 = nums3
                                    max3=logits[i][ways*5+j]
                                    nums3=j
                                else:
                                    if logits[i][ways*5+j]>max4:
                                        max5 = max4
                                        nums5 = nums4
                                        max4 = logits[i][ways * 5 + j]
                                        nums4 = j
                                    else:
                                        if logits[i][ways*5+j]>max5:
                                            max5 = logits[i][ways * 5 + j]
                                            nums5 = j

                    labels1[i]=ways*5+nums1
                    labels2[i]=ways*5+nums2
                    labels3[i] = ways * 5 + nums3
                    labels4[i] = ways * 5 + nums4
                    labels5[i] = ways * 5 + nums5
                loss1 = F.cross_entropy(logits, labels1.long())
                loss2 = F.cross_entropy(logits, labels2.long())
                loss3 = F.cross_entropy(logits, labels3.long())
                loss4 = F.cross_entropy(logits, labels4.long())
                loss5 = F.cross_entropy(logits, labels5.long())
                #loss=loss1*0.5+loss2*0.3+loss3*0.2
                #loss=(loss1*2+loss2)/3
                #loss=(loss1*5+loss2*4+loss3*3+loss4*2+loss5)/15
                loss = (loss1 * 4 + loss2 * 3 + loss3 * 2 + loss4) / 10
                '''


                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                aves['tl'].add(loss.item())
                aves['ta'].add(acc)

                logits = None; loss = None

            # eval
            model.eval()

            for name, loader, name_l, name_a in [
                    ('tval', val_loader, 'tvl', 'tva'),
                    ('val', val_loader, 'vl', 'va')]:

                if (config.get('tval_dataset') is None) and name == 'tval':
                    continue

                np.random.seed(0)
                for p, (input_cha, index) in enumerate(sup_loader):
                    for q, (input_cha1, index1) in enumerate(val_loader):
                        target = index.cuda()
                        input_var = torch.autograd.Variable(input_cha).cuda()
                        # target_var = torch.autograd.Variable(target)

                        target1 = index1.cuda()
                        input_var1 = torch.autograd.Variable(input_cha1).cuda()
                        label = target1.long()

                        shot_shape = [1, n_train_way, n_train_shot]
                        query_shape = [1, input_var1.shape[0]]
                        img_shape = input_var.shape[-3:]
                        x_shot = input_var.view(*shot_shape,*img_shape)
                        x_query = input_var1.view(*query_shape,*img_shape)
                        with torch.no_grad():
                            logits1,xx_shot1,xx_query1 = model(x_shot, x_query)
                            logits1=logits1.view(-1, n_way)
                            '''
                            logits = model(x_shot, x_query).view(-1, n_way*5)
                            labels = torch.ones(label.size(0)).cuda()
                            for i in range(label.size(0)):
                                max = 0
                                ways = label[i]
                                for j in range(5):
                                    if logits[i][ways * 5 + j] > max:
                                        max = logits[i][ways * 5 + j]
                                        nums = j
                                labels[i] = ways * 5 + nums
                            loss = F.cross_entropy(logits, labels.long())
                            '''
                            '''
                            logits = model(x_shot, x_query).view(-1, n_train_way * 5)
                            labels1 = torch.ones(label.size(0)).cuda()
                            labels2 = torch.ones(label.size(0)).cuda()
                            labels3 = torch.ones(label.size(0)).cuda()
                            labels4 = torch.ones(label.size(0)).cuda()
                            labels5 = torch.ones(label.size(0)).cuda()
                            for i in range(label.size(0)):

                                ways = label[i]
                                max1 = 0
                                max2 = 0
                                max3 = 0
                                max4 = 0
                                max5 = 0
                                nums1 = 0
                                nums2 = 0
                                nums3 = 0
                                nums4 = 0
                                nums5 = 0
                                for j in range(5):
                                    if logits[i][ways * 5 + j] > max1:
                                        max5 = max4
                                        nums5 = nums4
                                        max4 = max3
                                        nums4 = nums3
                                        max3 = max2
                                        nums3 = nums2
                                        max2 = max1
                                        nums2 = nums1
                                        max1 = logits[i][ways * 5 + j]
                                        nums1 = j
                                    else:
                                        if logits[i][ways * 5 + j] > max2:
                                            max5 = max4
                                            nums5 = nums4
                                            max4 = max3
                                            nums4 = nums3
                                            max3 = max2
                                            nums3 = nums2
                                            max2 = logits[i][ways * 5 + j]
                                            nums2 = j
                                        else:
                                            if logits[i][ways * 5 + j] > max3:
                                                max5 = max4
                                                nums5 = nums4
                                                max4 = max3
                                                nums4 = nums3
                                                max3 = logits[i][ways * 5 + j]
                                                nums3 = j
                                            else:
                                                if logits[i][ways * 5 + j] > max4:
                                                    max5 = max4
                                                    nums5 = nums4
                                                    max4 = logits[i][ways * 5 + j]
                                                    nums4 = j
                                                else:
                                                    if logits[i][ways * 5 + j] > max5:
                                                        max5 = logits[i][ways * 5 + j]
                                                        nums5 = j

                                labels1[i] = ways * 5 + nums1
                                labels2[i] = ways * 5 + nums2
                                labels3[i] = ways * 5 + nums3
                                labels4[i] = ways * 5 + nums4
                                labels5[i] = ways * 5 + nums5
                            loss1 = F.cross_entropy(logits, labels1.long())
                            loss2 = F.cross_entropy(logits, labels2.long())
                            loss3 = F.cross_entropy(logits, labels3.long())
                            loss4 = F.cross_entropy(logits, labels4.long())
                            loss5 = F.cross_entropy(logits, labels5.long())
                            #loss=loss1*0.5+loss2*0.3+loss3*0.2
                            # loss=(loss1*2+loss2)/3
                            #loss=(loss1*5+loss2*4+loss3*3+loss4*2+loss5)/15
                            loss=(loss1*4+loss2*3+loss3*2+loss4)/10
                            '''

                            loss = F.cross_entropy(logits1, label)
                            acc = utils.compute_acc(logits1, label)
                            #a=(torch.argmax(logits, dim=1) == label).float()
                            #b=(label == 0).float()
                            #c=a+b




                        aves[name_l].add(loss.item())
                        aves[name_a].add(acc)
                '''
                for data, _ in tqdm(loader, desc=name, leave=False):
                    x_shot, x_query = fs.split_shot_query(
                            data.cuda(), n_way, n_shot, n_query,
                            ep_per_batch=4)
                    label = fs.make_nk_label(n_way, n_query,
                            ep_per_batch=4).cuda()
    
                    with torch.no_grad():
                        logits = model(x_shot, x_query).view(-1, n_way)
                        loss = F.cross_entropy(logits, label)
                        acc = utils.compute_acc(logits, label)
                    
                    aves[name_l].add(loss.item())
                    aves[name_a].add(acc)
                '''

            train_loss_list.append(aves['tl'].v)
            train_acc_list.append(aves['ta'].v)
            test_loss_list.append(aves['vl'].v)
            test_acc_list.append(aves['va'].v)


            _sig = int(_[-1])

            # post
            if lr_scheduler is not None:
                lr_scheduler.step()

            for k, v in aves.items():
                aves[k] = v.item()
                trlog[k].append(aves[k])

            t_epoch = utils.time_str(timer_epoch.t())
            t_used = utils.time_str(timer_used.t())
            t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
            utils.log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
                    'val {:.4f}|{:.4f}, {} {}/{} (@{})'.format(
                    epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
                    aves['vl'], aves['va'], t_epoch, t_used, t_estimate, _sig))

            writer.add_scalars('loss', {
                'train': aves['tl'],
                'tval': aves['tvl'],
                'val': aves['vl'],
            }, epoch)
            writer.add_scalars('acc', {
                'train': aves['ta'],
                'tval': aves['tva'],
                'val': aves['va'],
            }, epoch)

            if config.get('_parallel'):
                model_ = model.module
            else:
                model_ = model

            training = {
                'epoch': epoch,
                'optimizer': config['optimizer'],
                'optimizer_args': config['optimizer_args'],
                'optimizer_sd': optimizer.state_dict(),
            }
            save_obj = {
                'file': __file__,
                'config': config,

                'model': config['model'],
                'model_args': config['model_args'],
                'model_sd': model_.state_dict(),

                'training': training,
            }
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
            torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

            if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(save_obj,
                        os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

            if aves['va'] > max_va:
                max_va = aves['va']
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

            writer.flush()


        #macro
        if config.get('load'):
            model_sv = torch.load(config['load'])
            model = models.load(model_sv)
        else:
            model = models.make(config['model'], **config['model_args'])

            if config.get('load_encoder'):
                #encoder = models.load(torch.load(os.path.join('./save/pre/casme2-4', str(i) + 'epoch-last.pth'))).encoder
                encoder = models.load(torch.load(config['load_encoder'])).encoder
                model.encoder.load_state_dict(encoder.state_dict())

        if config.get('_parallel'):
            model = nn.DataParallel(model)

        utils.log('num params: {}'.format(utils.compute_n_params(model)))

        optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])

        ########

        max_epoch = config['max_epoch']
        save_epoch = config.get('save_epoch')
        max_va = 0.
        timer_used = utils.Timer()
        timer_epoch = utils.Timer()

        aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
        trlog = dict()
        for k in aves_keys:
            trlog[k] = []

        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        for epoch in range(1, max_epoch + 1):
            timer_epoch.s()
            aves = {k: utils.Averager() for k in aves_keys}

            # train
            model.train()
            if config.get('freeze_bn'):
                utils.freeze_bn(model)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            np.random.seed(epoch)
            for data, _ in tqdm(train_loader, desc='train', leave=False):
                # print(data.shape)
                x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_train_way, n_train_shot, n_query,
                    ep_per_batch=ep_per_batch)
                label = fs.make_nk_label(n_train_way, n_query,
                                         ep_per_batch=ep_per_batch).cuda()

                logits,xx_shot,xx_query = model(x_shot, x_query)
                logits=logits.view(-1, n_way)
                '''
                logits = model(x_shot, x_query).view(-1, n_train_way * 5)
                labels=torch.ones(label.size(0)).cuda()
                for i in range(label.size(0)):
                    max=0
                    ways=label[i]
                    for j in range(5):
                        if logits[i][ways*5+j]>max:
                            max=logits[i][ways*5+j]
                            nums=j
                    labels[i]=ways*5+nums
                loss = F.cross_entropy(logits, labels.long())
                '''
                '''
                logits = model(x_shot, x_query).view(-1, n_train_way * 5)
                labels1 = torch.ones(label.size(0)).cuda()
                labels2 = torch.ones(label.size(0)).cuda()
                labels3 = torch.ones(label.size(0)).cuda()
                labels4 = torch.ones(label.size(0)).cuda()
                labels5 = torch.ones(label.size(0)).cuda()
                for i in range(label.size(0)):

                    ways=label[i]
                    max1 = 0
                    max2=0
                    max3=0
                    max4=0
                    max5=0
                    nums1=0
                    nums2=0
                    nums3=0
                    nums4=0
                    nums5=0
                    for j in range(5):
                        if logits[i][ways*5+j]>max1:
                            max5=max4
                            nums5=nums4
                            max4=max3
                            nums4=nums3
                            max3=max2
                            nums3=nums2
                            max2=max1
                            nums2=nums1
                            max1=logits[i][ways*5+j]
                            nums1=j
                        else:
                            if logits[i][ways*5+j]>max2:
                                max5 = max4
                                nums5 = nums4
                                max4 = max3
                                nums4 = nums3
                                max3=max2
                                nums3=nums2
                                max2=logits[i][ways*5+j]
                                nums2=j
                            else:
                                if logits[i][ways*5+j]>max3:
                                    max5 = max4
                                    nums5 = nums4
                                    max4 = max3
                                    nums4 = nums3
                                    max3=logits[i][ways*5+j]
                                    nums3=j
                                else:
                                    if logits[i][ways*5+j]>max4:
                                        max5 = max4
                                        nums5 = nums4
                                        max4 = logits[i][ways * 5 + j]
                                        nums4 = j
                                    else:
                                        if logits[i][ways*5+j]>max5:
                                            max5 = logits[i][ways * 5 + j]
                                            nums5 = j

                    labels1[i]=ways*5+nums1
                    labels2[i]=ways*5+nums2
                    labels3[i] = ways * 5 + nums3
                    labels4[i] = ways * 5 + nums4
                    labels5[i] = ways * 5 + nums5
                loss1 = F.cross_entropy(logits, labels1.long())
                loss2 = F.cross_entropy(logits, labels2.long())
                loss3 = F.cross_entropy(logits, labels3.long())
                loss4 = F.cross_entropy(logits, labels4.long())
                loss5 = F.cross_entropy(logits, labels5.long())
                #loss=loss1*0.5+loss2*0.3+loss3*0.2
                #loss=(loss1*2+loss2)/3
                #loss=(loss1*5+loss2*4+loss3*3+loss4*2+loss5)/15
                loss = (loss1 * 4 + loss2 * 3 + loss3 * 2 + loss4) / 10
                '''

                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                aves['tl'].add(loss.item())
                aves['ta'].add(acc)

                logits = None;
                loss = None

            # eval
            model.eval()

            for name, loader, name_l, name_a in [
                ('tval', val_loader, 'tvl', 'tva'),
                ('val', val_loader, 'vl', 'va')]:

                if (config.get('tval_dataset') is None) and name == 'tval':
                    continue

                np.random.seed(0)
                for p, (input_cha, index) in enumerate(sup_loader):
                    for q, (input_cha1, index1) in enumerate(val_loader):
                        target = index.cuda()
                        input_var = torch.autograd.Variable(input_cha).cuda()
                        # target_var = torch.autograd.Variable(target)

                        target1 = index1.cuda()
                        input_var1 = torch.autograd.Variable(input_cha1).cuda()
                        label = target1.long()

                        shot_shape = [1, n_train_way, n_train_shot]
                        query_shape = [1, input_var1.shape[0]]
                        img_shape = input_var.shape[-3:]
                        x_shot = input_var.view(*shot_shape, *img_shape)
                        x_query = input_var1.view(*query_shape, *img_shape)
                        with torch.no_grad():
                            logits2,xx_shot2,xx_query2 = model(x_shot, x_query)
                            logits2=logits2.view(-1, n_way)
                            '''
                            logits = model(x_shot, x_query).view(-1, n_way*5)
                            labels = torch.ones(label.size(0)).cuda()
                            for i in range(label.size(0)):
                                max = 0
                                ways = label[i]
                                for j in range(5):
                                    if logits[i][ways * 5 + j] > max:
                                        max = logits[i][ways * 5 + j]
                                        nums = j
                                labels[i] = ways * 5 + nums
                            loss = F.cross_entropy(logits, labels.long())
                            '''
                            '''
                            logits = model(x_shot, x_query).view(-1, n_train_way * 5)
                            labels1 = torch.ones(label.size(0)).cuda()
                            labels2 = torch.ones(label.size(0)).cuda()
                            labels3 = torch.ones(label.size(0)).cuda()
                            labels4 = torch.ones(label.size(0)).cuda()
                            labels5 = torch.ones(label.size(0)).cuda()
                            for i in range(label.size(0)):

                                ways = label[i]
                                max1 = 0
                                max2 = 0
                                max3 = 0
                                max4 = 0
                                max5 = 0
                                nums1 = 0
                                nums2 = 0
                                nums3 = 0
                                nums4 = 0
                                nums5 = 0
                                for j in range(5):
                                    if logits[i][ways * 5 + j] > max1:
                                        max5 = max4
                                        nums5 = nums4
                                        max4 = max3
                                        nums4 = nums3
                                        max3 = max2
                                        nums3 = nums2
                                        max2 = max1
                                        nums2 = nums1
                                        max1 = logits[i][ways * 5 + j]
                                        nums1 = j
                                    else:
                                        if logits[i][ways * 5 + j] > max2:
                                            max5 = max4
                                            nums5 = nums4
                                            max4 = max3
                                            nums4 = nums3
                                            max3 = max2
                                            nums3 = nums2
                                            max2 = logits[i][ways * 5 + j]
                                            nums2 = j
                                        else:
                                            if logits[i][ways * 5 + j] > max3:
                                                max5 = max4
                                                nums5 = nums4
                                                max4 = max3
                                                nums4 = nums3
                                                max3 = logits[i][ways * 5 + j]
                                                nums3 = j
                                            else:
                                                if logits[i][ways * 5 + j] > max4:
                                                    max5 = max4
                                                    nums5 = nums4
                                                    max4 = logits[i][ways * 5 + j]
                                                    nums4 = j
                                                else:
                                                    if logits[i][ways * 5 + j] > max5:
                                                        max5 = logits[i][ways * 5 + j]
                                                        nums5 = j

                                labels1[i] = ways * 5 + nums1
                                labels2[i] = ways * 5 + nums2
                                labels3[i] = ways * 5 + nums3
                                labels4[i] = ways * 5 + nums4
                                labels5[i] = ways * 5 + nums5
                            loss1 = F.cross_entropy(logits, labels1.long())
                            loss2 = F.cross_entropy(logits, labels2.long())
                            loss3 = F.cross_entropy(logits, labels3.long())
                            loss4 = F.cross_entropy(logits, labels4.long())
                            loss5 = F.cross_entropy(logits, labels5.long())
                            #loss=loss1*0.5+loss2*0.3+loss3*0.2
                            # loss=(loss1*2+loss2)/3
                            #loss=(loss1*5+loss2*4+loss3*3+loss4*2+loss5)/15
                            loss=(loss1*4+loss2*3+loss3*2+loss4)/10
                            '''

                            loss = F.cross_entropy(logits2, label)
                            acc = utils.compute_acc(logits2, label)
                            # a=(torch.argmax(logits, dim=1) == label).float()
                            # b=(label == 0).float()
                            # c=a+b

                        aves[name_l].add(loss.item())
                        aves[name_a].add(acc)
                '''
                for data, _ in tqdm(loader, desc=name, leave=False):
                    x_shot, x_query = fs.split_shot_query(
                            data.cuda(), n_way, n_shot, n_query,
                            ep_per_batch=4)
                    label = fs.make_nk_label(n_way, n_query,
                            ep_per_batch=4).cuda()

                    with torch.no_grad():
                        logits = model(x_shot, x_query).view(-1, n_way)
                        loss = F.cross_entropy(logits, label)
                        acc = utils.compute_acc(logits, label)

                    aves[name_l].add(loss.item())
                    aves[name_a].add(acc)
                '''

            train_loss_list.append(aves['tl'].v)
            train_acc_list.append(aves['ta'].v)
            test_loss_list.append(aves['vl'].v)
            test_acc_list.append(aves['va'].v)

            _sig = int(_[-1])

            # post
            if lr_scheduler is not None:
                lr_scheduler.step()

            for k, v in aves.items():
                aves[k] = v.item()
                trlog[k].append(aves[k])

            t_epoch = utils.time_str(timer_epoch.t())
            t_used = utils.time_str(timer_used.t())
            t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
            utils.log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
                      'val {:.4f}|{:.4f}, {} {}/{} (@{})'.format(
                epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
                aves['vl'], aves['va'], t_epoch, t_used, t_estimate, _sig))

            writer.add_scalars('loss', {
                'train': aves['tl'],
                'tval': aves['tvl'],
                'val': aves['vl'],
            }, epoch)
            writer.add_scalars('acc', {
                'train': aves['ta'],
                'tval': aves['tva'],
                'val': aves['va'],
            }, epoch)

            if config.get('_parallel'):
                model_ = model.module
            else:
                model_ = model

            training = {
                'epoch': epoch,
                'optimizer': config['optimizer'],
                'optimizer_args': config['optimizer_args'],
                'optimizer_sd': optimizer.state_dict(),
            }
            save_obj = {
                'file': __file__,
                'config': config,

                'model': config['model'],
                'model_args': config['model_args'],
                'model_sd': model_.state_dict(),

                'training': training,
            }
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
            torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

            if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(save_obj,
                           os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

            if aves['va'] > max_va:
                max_va = aves['va']
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

            writer.flush()

        '''
        xx_shot = torch.cat([xx_shot1, xx_shot2], dim=-1)
        xx_query = torch.cat([xx_query1, xx_query2], dim=-1)
        xx_shot = xx_shot.mean(dim=-2)
        # print(x_shot.shape)
        xx_shot = F.normalize(xx_shot, dim=-1)
        # print(x_shot.shape)
        xx_query = F.normalize(xx_query, dim=-1)
        # print(x_query.shape)
        metric = 'dot'
        logits = utils.compute_logits(
            xx_query, xx_shot, metric=metric)
        logits=logits.view(-1, n_way)
        '''

        max1=0
        #print(i)
        if i==chf:
            for j in range(1, 11):
                logits=(logits1+1)+(logits2+1)*j/10
                #y_true=y_true+label.cpu().numpy().tolist()
                #y_pred=y_pred+torch.argmax(logits, dim=1).cpu().numpy().tolist()
                acc_num1 = (torch.argmax(logits, dim=1) == label).float().sum().item()
                if acc_num1>=max1:
                    max1=acc_num1
                    rate=j
                print(j/10)
                print(acc_num1)
                print(rate)
        else:
            logits = (logits1 + 1) + (logits2 + 1) * rate/10
            #print(len(logits1))
            #print(len(logits2))
            #print(len(logits))
            #print(len(y_true))
            #print(len(y_pred))
            y_true = y_true + label.cpu().numpy().tolist()
            y_pred = y_pred + torch.argmax(logits, dim=1).cpu().numpy().tolist()
            c = c+ (torch.argmax(logits, dim=1) == label).float().sum().item()
            c1 =c1+ ((torch.argmax(logits, dim=1) == label).float() + (label == 0).float() == 2).float().sum().item()
            c2 =c2+ ((torch.argmax(logits, dim=1) == label).float() + (label == 1).float() == 2).float().sum().item()
            c3 =c3+ ((torch.argmax(logits, dim=1) == label).float() + (label == 2).float() == 2).float().sum().item()
            c4 =c4+ ((torch.argmax(logits, dim=1) == label).float() + (label == 3).float() == 2).float().sum().item()
            #c5 =c5+ ((torch.argmax(logits, dim=1) == label).float() + (label == 4).float() == 2).float().sum().item()
            print(c, c1, c2, c3, c4)
            #print(len(y_true))
            #print(len(y_pred))

        '''
        x1=range(0,50)
        x2=range(0,50)
        y1=train_loss_list
        y2=train_acc_list
        y3=test_loss_list
        y4=test_acc_list
        plt.subplot(1,2,1)
        plt.plot(x1,y1)
        plt.plot(x1,y2,c="r")
        plt.title("train loss & accuracy")
        plt.subplot(1,2,2)
        plt.plot(x2,y3)
        plt.plot(x2,y4,c="r")
        plt.title("test loss & accuracy")
        plt.show()
        plt.savefig("train result.jpg")
        '''
    print(c,c1,c2,c3,c4)
    sns.set()
    C2 = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    sns.heatmap(C2, annot=True)
    print(C2)






if __name__ == '__main__':
    parser = argparse.ArgumentParser() #创建解析器 python train_meta.py --config configs/train_meta_mini.yaml
    parser.add_argument('--config') #添加参数
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args() #解析参数

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader) #load():返回一个对象
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    #print(config)
    utils.set_gpu(args.gpu) #指定多个GPU训练
    main(config)

