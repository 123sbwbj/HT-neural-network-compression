
import os
import cv2
import numpy as np


data_path = '原始数据集地址'
save_path='处理好的帧数据地址'

#获取video列表
def get_clips(class_name):
    pathclass=os.path.join(data_path,class_name)
    files = os.listdir(pathclass)
    files.sort()
    clip_list = []
    for this_file in files:
        pathfile=os.path.join(pathclass,this_file)
        clips = os.listdir(pathfile)
        clips.sort()
        for this_clip in clips:
            pathclip=os.path.join(pathfile,this_clip)
            clip_list.append( pathclip )
    return clip_list

#生成每个视频的帧文件
def load_traindata(inds,set):
    N = len(inds)
    p=0
    for i in range(N):
        frame_count =0
        cv=cv2.VideoCapture(set[0][inds[i]])
        if cv.isOpened():
            ravel,frames=cv.read()
        else:
            ravel=False
        this_clip=[]
        while (ravel):
            size=(120,160)#对应大小，改
            frames=cv2.resize(frames,size, interpolation = cv2.INTER_CUBIC)
            frames=frames.reshape(1,-1)
            if frames.shape[1] == 120*160*3:#对应大小，改
                frame_count = frame_count + 1
                this_clip.append(frames)
            ravel,frames=cv.read()
        if frame_count>p:
            p=frame_count
        this_clip = np.array(this_clip)
        # flatten the dimensions 1, 2 and 3
        if this_clip.shape[0]!=0:
            this_clip = this_clip.reshape(this_clip.shape[0],-1) # of shape (nb_frames, 120*160*3)
            this_clip = this_clip .astype('int8')
            this_clip=this_clip/255

            X = this_clip.transpose(1,0)
            Y = set[1][inds[i]]
            file_name=str(i)
            file_save=os.path.join(save_path,file_name)
            np.savez_compressed(file_save, images= X, labels=Y)
    return [1,1]


def data_trainandtest( ):
# Load the data --------------------------------------------------------------------------------------------------------
    np.random.seed(11111986)


    classes = ['basketball', 'biking', 'diving', 'golf_swing', 'horse_riding', 'soccer_juggling',
              'swing', 'tennis_swing', 'trampoline_jumping', 'volleyball_spiking', 'walking']

    labels1 = [None]*11
    labels2 = [None]*11
    labels3 = [None] * 11
    labels4 = [None] * 11
    labels5 = [None] * 11
    ran1=[None]*11
    ran2 = [None] * 11
    ran3 = [None] * 11
    ran4 = [None] * 11
    ran5 = [None] * 11
    sizes = np.zeros(11).astype('int')

    #将每一类视频数据均分到5个组中，并打乱顺序
    for k in range(11):
        rr1 = []
        rr2 = []
        rr3 = []
        rr4 = []
        rr5 = []
        this_clip = get_clips(classes[k])
        sizes[k] = int(len(this_clip))
        n_anchor1 = np.random.choice(len(this_clip), size=sizes[k]//5,replace=False)
        labels1[k]= np.repeat([k],n_anchor1.size )
        n_anchor1.sort()
        for n in range(0,sizes[k]//5):
            r = this_clip[n_anchor1[n]]
            rr1.append(r)
        n_anchor22 = np.setdiff1d(np.arange(len(this_clip)), n_anchor1)  # 筛选train：test =4:1
        n_anchor2 = np.random.choice(n_anchor22, size=sizes[k]//5,replace=False)
        labels2[k]= np.repeat([k],n_anchor2.size )
        n_anchor2.sort()
        for n in range(0,sizes[k]//5):
            r = this_clip[n_anchor2[n]]
            rr2.append(r)
        n_anchor33 = np.setdiff1d(n_anchor22, n_anchor2)  # 筛选train：test =4:1
        n_anchor3 = np.random.choice(n_anchor33, size=sizes[k]//5,replace=False)
        labels3[k]= np.repeat([k],n_anchor3.size )
        n_anchor3.sort()
        for n in range(0,sizes[k]//5):
            r = this_clip[n_anchor3[n]]
            rr3.append(r)
        n_anchor44 = np.setdiff1d(n_anchor33, n_anchor3)  # 筛选train：test =4:1
        n_anchor4 = np.random.choice(n_anchor44, size=sizes[k]//5,replace=False)
        labels4[k]= np.repeat([k],n_anchor4.size )
        n_anchor4.sort()
        for n in range(0,sizes[k]//5):
            r = this_clip[n_anchor4[n]]
            rr4.append(r)
        n_anchor5 = np.setdiff1d(n_anchor44, n_anchor4)  # 筛选train：test =4:1
        labels5[k]= np.repeat([k],n_anchor5.size )
        n_anchor5.sort()
        for n in range(0,len(n_anchor5)):
            r = this_clip[n_anchor5[n]]
            rr5.append(r)
        ran1[k]=rr1
        ran2[k]=rr2
        ran3[k]=rr3
        ran4[k]=rr4
        ran5[k]=rr5



    ran1=np.array( [item for sublist in ran1 for item in sublist] )#训练集 110*11=1210
    ran2 = np.array([item for sublist in ran2 for item in sublist])  # 训练集 110*11=1210
    ran3 = np.array([item for sublist in ran3 for item in sublist])  # 训练集 110*11=1210
    ran4 = np.array([item for sublist in ran4 for item in sublist])  # 训练集 110*11=1210
    ran5 = np.array([item for sublist in ran5 for item in sublist])  # 训练集 110*11=1210



    labels1 = np.array([item for sublist in labels1 for item in sublist])
    labels2 = np.array([item for sublist in labels2 for item in sublist])
    labels3 = np.array([item for sublist in labels3 for item in sublist])
    labels4 = np.array([item for sublist in labels4 for item in sublist])
    labels5 = np.array([item for sublist in labels5 for item in sublist])





    shuffle_inds = np.random.choice(range(len(ran1)), len(ran1), False)  # 打乱顺序
    ran1 = ran1[shuffle_inds]
    labels1 = labels1[shuffle_inds]

    shuffle_inds = np.random.choice(range(len(ran2)), len(ran2), False)  # 打乱顺序
    ran2= ran2[shuffle_inds]
    labels2 = labels2[shuffle_inds]

    shuffle_inds = np.random.choice(range(len(ran3)), len(ran3), False)  # 打乱顺序
    ran3 = ran3[shuffle_inds]
    labels3 = labels3[shuffle_inds]

    shuffle_inds = np.random.choice(range(len(ran4)), len(ran4), False)  # 打乱顺序
    ran4 = ran4[shuffle_inds]
    labels4 = labels4[shuffle_inds]

    shuffle_inds = np.random.choice(range(len(ran5)), len(ran5), False)  # 打乱顺序
    ran5 = ran5[shuffle_inds]
    labels5 = labels5[shuffle_inds]


    ran1 = np.hstack((ran1, ran2))
    labels1=np.hstack((labels1 ,labels2))
    ran1 = np.hstack((ran1, ran3))
    labels1 = np.hstack((labels1, labels3))
    ran1 = np.hstack((ran1, ran4))
    labels1=np.hstack((labels1 ,labels4))
    ran1 = np.hstack((ran1, ran5))
    labels1=np.hstack((labels1 ,labels5))

    train_inds= np.arange(len(ran1))


    set = [ran1[train_inds], labels1[train_inds]]


    n_tr =len(set[0])

    X, Y = load_traindata(np.arange(0, n_tr), set)  # full set



if __name__ == '__main__':
    data_trainandtest()

