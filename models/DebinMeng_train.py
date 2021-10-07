#coding=utf-8
import os, sys, shutil
from torchvision.transforms import ToPILImage
from torchvision import transforms
import random as rd
from PIL import Image
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.utils.data as data
import cv2
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import pdb
import csv
import os
from numpy import linalg
from scipy import signal
from pylab import *
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random
try:
    import cPickle as pickle
except:
    import pickle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_imgs_tsn(video_root, video_list, rectify_label,kuochong):
    imgs_first = list()
    imgs_second = list()
    imgs_third = list()
    maxa=0
    maxb=0
    maxm=0
    with open(video_list, 'r') as imf:
        index = []
        flag = []
        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name = video_label[0]  # name of video
            label = rectify_label[video_label[1]]  # label of video
            start= int(video_label[2])
            aplex= int(video_label[3])-start


            video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
            ###  for sampling triple imgs in the single video_path  ####

            img_lists = os.listdir(video_path)

            l=len(video_name)


            #img_lists.sort(key = lambda x : int(x[int(l)-6+5:-4]))  # sort files by ascending
            img_lists.sort(key=lambda x: int(x[7:-4]))
            #print(video_name,int(l)-1)
            img_count = len(img_lists)  # number of frames in video
            num_per_part = int(img_count) // 3

            #aplex = int((img_count-1) / 2)

            #if int(img_count) >=2:
                #for i in range(img_count):

                    #random_select_first = random.randint(0, num_per_part)
                    #random_select_second = random.randint(num_per_part, num_per_part * 2)
                    #random_select_third = random.randint(2 * num_per_part, len(img_lists) - 1)

            img_path_first = os.path.join(video_path, img_lists[0])
            img_path_second = os.path.join(video_path, img_lists[1])
            img_path_third = os.path.join(video_path, img_lists[aplex+1])

            Image1 = Image.open(img_path_second).convert('L')
            Image2 = Image.open(img_path_third).convert('L')
            transform = transforms.Compose([
                transforms.Resize((80, 80)),
            ])
            Image1 = transform(Image1)
            Image2 = transform(Image2)
            flow = cv2.calcOpticalFlowFarneback(np.array(Image1), np.array(Image2), None, 0.5, 3, 15, 3, 5, 1.2, 0)

            a = flow[:, :, 0]
            b = flow[:, :, 1]
            m=(a**2+b**2)**0.5

            max1=np.max(a)
            max2=np.min(a)
            max1=abs(max1)
            max2=abs(max2)
            if max1>maxa:
                maxa=max1
            if max2>maxa:
                maxa=max2
            max1 = np.max(b)
            max2 = np.min(b)
            max1 = abs(max1)
            max2 = abs(max2)
            if max1 > maxb:
                maxb = max1
            if max2 > maxb:
                maxb = max2
            max1 = np.max(m)
            max2 = np.min(m)
            max1 = abs(max1)
            max2 = abs(max2)
            if max1 > maxm:
                maxm = max1
            if max2 > maxm:
                maxm = max2





                #img_path_third = os.path.join(video_path, img_lists[random_select_third])

            imgs_first.append(img_path_first)
            imgs_second.append(img_path_second)
            imgs_third.append(img_path_third)
                #imgs_third.append(img_path_third)



            ###  return video frame index  #####
            index.append(np.ones(1) * label)  # id: 0 : 379
            flag.append(0)

            if kuochong is True:
                imgs_first.append(img_path_first)
                imgs_second.append(img_path_second)
                imgs_third.append(img_path_third)
                index.append(np.ones(1) * label)
                flag.append(1)



            '''
            if label!=4 and label!=3 and kuochong is True:
                imgs_first.append(img_path_first)
                imgs_second.append(img_path_second)
                imgs_third.append(img_path_third)
                index.append(np.ones(1) * label)
                flag.append(1)
            '''



        index = np.concatenate(index, axis=0)
        # index = index.astype(int)
    return imgs_first, imgs_second, imgs_third, index,flag,maxa,maxb,maxm



def load_imgs_total_frame(video_root, video_list, rectify_label):
    imgs_first = list()

    with open(video_list, 'r') as imf:
        index = []
        video_names = []
        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name = video_label[0]  # name of video
            label = rectify_label[video_label[1]]  # label of video

            video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
            ###  for sampling triple imgs in the single video_path  ####

            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_count = len(img_lists)  # number of frames in video

            for frame in img_lists:
                # pdb.set_trace()
                imgs_first.append((os.path.join(video_path, frame), label))
            ###  return video frame index  #####
            video_names.append(video_name)
            index.append(np.ones(img_count) * id)
        index = np.concatenate(index, axis=0)
        # index = index.astype(int)
    return imgs_first, index

class VideoDataset(data.Dataset):
    def __init__(self, video_root, video_list, rectify_label=None, transform=None, csv = False):

        self.imgs_first, self.index = load_imgs_total_frame(video_root, video_list, rectify_label)
        self.transform = transform

    def __getitem__(self, index):

        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)

        return img_first, target_first, self.index[index]

    def __len__(self):
        return len(self.imgs_first)

#
# function defines the gaussian function used for convolution which returns the
def GaussianFunction(x, sigma):
    if sigma == 0:
        return 0
    else:
        g = (1/math.sqrt(2*math.pi*sigma*sigma))*math.exp(-x*x)/(2*sigma*sigma)
    return g

# function returns the gaussian kernel using the GaussianFunction of size 3x3
def GaussianMask(sigma):
    g = []
    for i in range(-2, 3):			#creating a gaussian kernel of size 3x3
        g1 = GaussianFunction(i,sigma)
        g2 = GaussianFunction(i-0.5, sigma)
        g3 = GaussianFunction(i+0.5, sigma)
        gaussian = (g1+g2+g3)/3
        g.append(gaussian)
    return g

sigma = 1.5
G = [] # Gaussian Kernel
G = GaussianMask(sigma)

def DownSample(I):
    Ix = Iy  = []
    I = np.array(I)
    S = np.shape(I)											#shape of the image
    for i in range(S[0]):
        Ix.extend([signal.convolve(I[i,:],G,'same')])		#convolution of the I[i] with G
    Ix = np.array(np.matrix(Ix))
    Iy = Ix[::2, ::2]										#selects the alternate column and row
    return Iy

def UpSample(I):
    I = np.array(I)
    S = np.shape(I)

    Ix = np.zeros((S[0], 2*S[1]))			#inserting alternate rows of zeros
    Ix[:, ::2] = I
    S1 = np.shape(Ix)
    Iy = np.zeros((2*S1[0], S1[1]))		#inserting alternate columns of zeros
    Iy[::2, :] = Ix
    Ig = cv2.GaussianBlur(Iy, (5,5), 1.5, 1.5)		#instead of using the user-defined gaussian function, I am using the Gaussian Blur functtion for double the size of gaussian kernel size
    return Ig

def LucasKanade(I1, I2):
    I1 = np.array(I1)
    I2 = np.array(I2)
    S = np.shape(I1)

    Ix = signal.convolve2d(I1,[[-0.25,0.25],[-0.25,0.25]],'same') + signal.convolve2d(I2,[[-0.25,0.25],[-0.25,0.25]],'same')
    Iy = signal.convolve2d(I1,[[-0.25,-0.25],[0.25,0.25]],'same') + signal.convolve2d(I2,[[-0.25,-0.25],[0.25,0.25]],'same')
    It = signal.convolve2d(I1,[[0.25,0.25],[0.25,0.25]],'same') + signal.convolve2d(I2,[[-0.25,-0.25],[-0.25,-0.25]],'same')

    features = cv2.goodFeaturesToTrack(I1, 10000, 0.01, 10)
    features = np.int0(features)

    u = v = np.ones((S))
    for l in features:
        j,i = l.ravel()
        IX = ([Ix[i-1,j-1],Ix[i,j-1],Ix[i-1,j-1],Ix[i-1,j],Ix[i,j],Ix[i+1,j],Ix[i-1,j+1],Ix[i,j+1],Ix[i+1,j-1]])
        IY = ([Iy[i-1,j-1],Iy[i,j-1],Iy[i-1,j-1],Iy[i-1,j],Iy[i,j],Iy[i+1,j],Iy[i-1,j+1],Iy[i,j+1],Iy[i+1,j-1]])
        IT = ([It[i-1,j-1],It[i,j-1],It[i-1,j-1],It[i-1,j],It[i,j],It[i+1,j],It[i-1,j+1],It[i,j+1],It[i+1,j-1]])

        # Using the minimum least squares solution approach
        LK = (IX,IY)
        LK = matrix(LK)
        LK_T = array(matrix(LK))
        LK = array(np.matrix.transpose(LK))

        #Psedudo Inverse
        A1 = np.dot(LK_T,LK)
        A2 = np.linalg.pinv(A1)
        A3 = np.dot(A2,LK_T)
        (u[i,j],v[i,j]) = np.dot(A3,IT) # we have the vectors with minimized square error

    u = np.flipud(u)
    v = np.flipud(v)
    return u,v

def LucasKanadeIterative(I1, I2, u1, v1):
    I1 = np.array(I1)
    I2 = np.array(I2)
    S = np.shape(I1)
    u1 = np.round(u1)
    v1 = np.round(v1)
    u = np.zeros(S)
    v = np.zeros(S)

    for i in range(2, S[0]-2):
        for j in range(2, S[1]-2):
            I1new = I1[i-2:i+3,j-2:j+3] 		# picking 5x5 pixels at a time
            lr = (i-2)+v1[i,j]				 	#Low Row Index
            hr = (i+2)+v1[i,j] 					#High Row Index
            lc = (j-2)+u1[i,j] 					#Low Column Index
            hc = (j+2)+u1[i,j] 					#High Column Index

            #window search and selecting the last window if it goes out of bounds
            if(lr < 0):
                lr = 0
                hr = 4
            if(lc < 0):
                lc = 0
                hc = 4
            if(hr > (len(I1[:,0]))-1):
                lr = len(I1[:,0])-5
                hr = len(I1[:,0])-1
            if(hc > (len(I1[0,:]))-1):
                lc = len(I1[0,:])-5
                hc = len(I1[0,:])-1
            if(np.isnan(lr)):
                lr = i-2
                hr = i+2
            if(np.isnan(lc)):
                lc = j-2
                hc = j+2
            #Selecting the same window for the second frame
            I2new = I2[int(lr):int(hr+1),int(lc):int(hc+1)]
            # Now applying LK for each window of the 2 images
            Ix = signal.convolve2d(I1new,[[-0.25,0.25],[-0.25,0.25]],'same') + signal.convolve2d(I2new,[[-0.25,0.25],[-0.25,0.25]],'same')
            Iy = signal.convolve2d(I1new,[[-0.25,-0.25],[0.25,0.25]],'same') + signal.convolve2d(I2new,[[-0.25,-0.25],[0.25,0.25]],'same')
            It = signal.convolve2d(I1new,[[0.25,0.25],[0.25,0.25]],'same') + signal.convolve2d(I2new,[[-0.25,-0.25],[-0.25,-0.25]],'same')

            IX =np.transpose(Ix[1:5,1:5])
            IY = np.transpose(Iy[1:5,1:5])
            IT = np.transpose(It[1:5,1:5])

            IX = IX.ravel()
            IY = IY.ravel()
            IT = IT.ravel()

            LK = (IX,IY)
            LK = np.matrix(LK)
            LK_T = np.array(np.matrix(LK))
            LK = np.array(np.matrix.transpose(LK))

            A1 = np.dot(LK_T,LK)
            A2 = np.linalg.pinv(A1)
            A3 = np.dot(A2,LK_T)
            (u[i,j],v[i,j]) = np.dot(A3,IT)
    #u = u+u1
    #v = v + v1
    r = np.mat(transpose(LK))*np.mat(LK)
    r = 1.0/(linalg.cond(r))
    return u,v,r
def draw_flow(img,flow,step=8):
    img=np.array(img)
    h,w=img.shape[:2]
    y,x=np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1).astype(int)
    fx,fy=flow[y,x].T
    #fy=flow[4::8, 4::8, 1].reshape(-1)
    #fx=0
    lines=np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines=np.int32(lines+0.5)
    vis=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis,lines,0,(0,255,0))
    for (x1,y1),(_x2,_y2) in lines:
        cv2.circle(vis,(x1,y1),1,(0,255,0),-1)
    return vis

class TripleImageDataset(data.Dataset):
    def __init__(self, video_root, video_list, rectify_label=None, transform=None,kuochong=False):

        self.imgs_first, self.imgs_second, self.imgs_third, self.index,self.flag,self.maxa,self.maxb,self.maxm = load_imgs_tsn(video_root, video_list,
                                                                                           rectify_label,kuochong)
        self.transform = transform

    def __getitem__(self, index):

        HF = transforms.RandomHorizontalFlip(p=1)

        path_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            if flag==1:
                img_first = HF(img_first)
            img_first = self.transform(img_first)

        path_second = self.imgs_second[index]
        img_second = Image.open(path_second).convert("RGB")
        if self.transform is not None:
            if flag==1:
                img_second = HF(img_second)
            img_second = self.transform(img_second)

        path_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        if self.transform is not None:
            if flag==1:
                img_third = HF(img_third)
            img_third = self.transform(img_third)
        img_cha = img_third - img_second
        #img_cha=torch.abs(img_cha)
        #img_cha=img_cha/torch.max(img_cha)

        path_second = self.imgs_second[index]
        path_third = self.imgs_third[index]
        Image1 = Image.open(path_second).convert('L')
        transform = transforms.Compose([
            transforms.Resize((80, 80)),
        ])
        Image1 = transform(Image1)
        Image2 = Image.open(path_third).convert('L')
        Image2 = transform(Image2)
        # a=np.array(Image2)

        if flag == 1:
            Image1 = HF(Image1)
            Image2 = HF(Image2)

        
        flow = cv2.calcOpticalFlowFarneback(np.array(Image1), np.array(Image2), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        '''
        vis=draw_flow(Image2,flow)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(vis)
        plt.axis('off')
        plt.savefig("11111.png",bbox_inches='tight')
        '''
        a = flow[:, :, 0]
        b = flow[:, :, 1]
        m = np.ones_like(a)
        m = torch.tensor(m)
        a = torch.tensor(a)
        b = torch.tensor(b)
        m = (a ** 2 + b ** 2) ** 0.5

        m = m.view(1, m.size(0), m.size(1))
        a = a.view(1, a.size(0), a.size(1))
        b = b.view(1, b.size(0), b.size(1))
        # a = (a + self.maxa) / (self.maxa + self.maxa)
        # b = (b + self.maxb) / (self.maxb + self.maxb)
        # m = (m + self.maxm) / (self.maxm + self.maxm)
        a = a / self.maxa
        b = b / self.maxb
        m = m / self.maxm
        gl = torch.cat([a, b, m], dim=0)
        img = torch.cat((img_cha, gl), 1)




        return img, self.index[index]
        '''
        path_second = self.imgs_second[index]
        path_third = self.imgs_third[index]
        Image1 = Image.open(path_second).convert('L')
        if self.transform is not None:
            Image1 = self.transform(Image1)
        Image2 = Image.open(path_third).convert('L')
        if self.transform is not None:
            Image2 = self.transform(Image2)
        #a=np.array(Image2)
        flow = cv2.calcOpticalFlowFarneback(np.array(Image1), np.array(Image2), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #vis=draw_flow(Image2,flow)
        #plt.imshow(vis)
        #plt.savefig("11111.png")
        a = flow[:, :, 0]
        b = flow[:, :, 1]
        m = np.ones_like(a)
        m = torch.tensor(m)
        a = torch.tensor(a)
        b = torch.tensor(b)
        m=(a**2+b**2)**0.5


        m=m.view(1,m.size(0),m.size(1))
        a=a.view(1,a.size(0),a.size(1))
        b=b.view(1,b.size(0),b.size(1))
        #a = (a + self.maxa) / (self.maxa + self.maxa)
        #b = (b + self.maxb) / (self.maxb + self.maxb)
        #m = (m + self.maxm) / (self.maxm + self.maxm)
        a=a/self.maxa
        b=b/self.maxb
        m=m/self.maxm
        gl=torch.cat([a,b,m],dim=0)
        #gl=gl.to(torch.float32)
        #unloader = transforms.ToPILImage()
        #gl=unloader(gl)
        #if self.transform is not None:
        #    gl = self.transform(gl)

        return gl,self.index[index] #归一化 dy 把图像转tensor调到前面
        '''
        '''
        path_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)

        path_second = self.imgs_second[index]
        img_second = Image.open(path_second).convert("RGB")
        if self.transform is not None:
            img_second = self.transform(img_second)

        path_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        if self.transform is not None:
            img_third = self.transform(img_third)
        '''




        '''
        path_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        if self.transform is not None:
            img_third = self.transform(img_third)
        '''

        '''
        unloader = transforms.ToPILImage()
        image1 = img_cha.cpu().clone()  # clone the tensor
        image1 = image1.squeeze(0)  # remove the fake batch dimension
        image1 = unloader(image1)
        image1.save('example.jpg')
        image2 = img_first.cpu().clone()  # clone the tensor
        image2 = image2.squeeze(0)  # remove the fake batch dimension
        image2 = unloader(image2)
        image2.save('example1.jpg')
        '''
        '''
        img_cha = img_third - img_second
        img=torch.cat((img_first,img_cha),1)

        return img_cha, self.index[index]
        '''




    def __len__(self):
        return len(self.imgs_first)


