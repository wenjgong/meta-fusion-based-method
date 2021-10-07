import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        #print(x_tot.shape)
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        #print(x_shot.shape)
        #print(x_query.shape)
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)
        #print(x_shot.shape)
        #print(x_query.shape)
        '''
        for i in range(x_shot.size(0)):
            for j in range(x_shot.size(1)):
                min=1
                for k in range(x_shot.size(2)):
                    for l in range(k+1,x_shot.size(2)):
                        in1=x_shot[i,j,k,:]
                        in2=x_shot[i,j,l,:]
                        in1=in1.unsqueeze(0).float()
                        in2=in2.unsqueeze(0).float()
                        dis=torch.cosine_similarity(in1,in2).item()
                        if dis<min:
                            min=dis
                            minx=k
                            miny=l
                me=(x_shot[i,j,minx,:]+x_shot[i,j,miny,:])/2
                for k in range(x_shot.size(2)):
                    x_shot[i,j,k]=me
        '''
        '''
        if x_query.size(0)==1:
            flag=torch.ones(x_shot.size(0),x_shot.size(1),x_shot.size(2))
            for i in range(x_shot.size(0)):
                while True:
                    max=-1
                    for j in range(x_shot.size(1)):
                        for k in range(x_shot.size(2)):
                            if flag[i,j,k]!=0:
                                in1=x_shot[i,j,k,:]
                                in1 = in1.unsqueeze(0).float()
                                for j1 in range(x_shot.size(1)):
                                    for k1 in range(x_shot.size(2)):
                                        if j==j1 and k==k1:
                                            continue
                                        if flag[i,j1,k1]!=0:
                                            in2=x_shot[i,j1,k1,:]

                                            in2=in2.unsqueeze(0).float()
                                            dis=torch.cosine_similarity(in1,in2).item()
                                            if dis>max:
                                                max=dis
                                                minclass1=j
                                                minloc1=k
                                                minclass2=j1
                                                minloc2=k1
                    if minclass1==minclass2:
                        x_shot[i,minclass1,minloc1]=(x_shot[i,minclass1,minloc1,:]*flag[i,minclass1,minloc1]+x_shot[i,minclass2,minloc2,:]*flag[i,minclass2,minloc2])/(flag[i,minclass1,minloc1]+flag[i,minclass2,minloc2])
                        flag[i, minclass1, minloc1]=flag[i,minclass1,minloc1]+flag[i,minclass2,minloc2]
                        flag[i, minclass2, minloc2]=0
                    else:
                        break
            yuce=torch.ones(x_query.size(0),x_query.size(1),x_shot.size(1)).cuda()
            #yuce.requires_grad=True
            for i in range(x_shot.size(0)):
                for j in range(x_query.size(1)):
                    in1 = x_query[i, j, :]
                    in1 = in1.unsqueeze(0).float()
                    for k in range(x_shot.size(1)):
                        max=-1
                        for m in range(x_shot.size(2)):
                            if flag[i,k,m]!=0:
                                in2=x_shot[i,k,m,:]

                                in2 = in2.unsqueeze(0).float()
                                dis = torch.cosine_similarity(in1, in2).item()
                                if dis>max:
                                    max=dis
                        yuce[i, j, k] = max
            return yuce
        '''
        '''
        if x_query.size(0) == 1:
            yuce = torch.ones(x_query.size(0), x_query.size(1), x_shot.size(1)).cuda()
            for i in range(x_shot.size(0)):
                for j in range(x_query.size(1)):
                    in1 = x_query[i, j, :]
                    in1 = in1.unsqueeze(0).float()
                    for k in range(x_shot.size(1)):
                        max = -1
                        for m in range(x_shot.size(2)):
                            in2 = x_shot[i, k, m, :]
    
                            in2 = in2.unsqueeze(0).float()
                            dis = torch.cosine_similarity(in1, in2).item()
                            if dis > max:
                                max = dis
                        yuce[i, j, k] = max
            return yuce
            '''









        xx_shot=x_shot
        xx_query=x_query

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            #print(x_shot.shape)
            x_shot = F.normalize(x_shot, dim=-1)
            #print(x_shot.shape)
            x_query = F.normalize(x_query, dim=-1)
            #print(x_query.shape)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'
        #x_shot=x_shot.reshape(x_shot.size(0),-1,x_shot.size(-1))


        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        return logits,xx_shot,xx_query



