import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from torch.utils.data import DataLoader
import time
from scipy.interpolate import interp1d
import tikzplotlib
sys.path.insert(0, '../source')
from dataset import VaryGeoDataset
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh, setAxisLabel,to4DTensor
from model import USCNN
from readOF import convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp
from decimal import Decimal, Context
import cv2
import glob

# 時刻0における境界の座標(x,y)を取得
OFBCCoord = Ofpp.parse_boundary_field('TemplateCase/0/C')
OFLOWC = OFBCCoord[b'low'][b'value']
OFUPC = OFBCCoord[b'up'][b'value']
OFLEFTC = OFBCCoord[b'left'][b'value']
OFRIGHTC = OFBCCoord[b'right'][b'value']
leftX = OFLEFTC[:,0]
leftY = OFLEFTC[:,1]
lowX = OFLOWC[:,0]
lowY = OFLOWC[:,1]
rightX = OFRIGHTC[:,0]
rightY = OFRIGHTC[:,1]
upX =  OFUPC[:,0]
upY = OFUPC[:,1]

# 境界(x,y)の分割数
ny = len(leftX)
nx = len(lowX)


# 温度場Tを取得
# idxをOpenFOAMのファイル名と一致させる関数
def decimal_normalize(f):
    # 指数表記を10進数表記に変える関数
    def _remove_exponent(d):
        return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()
    a = Decimal.normalize(Decimal(str(f)))
    b = _remove_exponent(a)
    return str(b)

# T0と最終的なモデルの出力と比べるための正解データを作成(今回はtime_step=0.05で初期条件+10個用意する)
time_step = Decimal(str(0.01))
T = []
for i in range(31):
    idx = decimal_normalize(i*time_step)
    OFPic = convertOFMeshToImage_StructuredMesh(nx,ny,'TemplateCase/0/C',['TemplateCase/'+idx+'/T'],[0,1,0,1],0.0,False)
    T.append(OFPic[:,:,2])
time_step = 0.01


# dξ,dηのスケール
h = 0.01

# 様々な変数を定義
NvarInput = 1
NvarOutput = 1
batchSize = 1
nEpochs = 10000
lr = 0.001
Ns = 1
nu = 0.01
#熱伝導係数(OpenFOAMのtransportProperties)
k = 1

# モデルの定義
model = USCNN(h,nx,ny,NvarInput,NvarOutput).to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)


# パディングを与えるメソッドの定義
padSingleSide = 1
udfpad = nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)


# myMeshはインスタンス変数として(x,y),(ξ,η)の座標やdx/dξ,dy/dξ,dx/dη,dy/dηや行列式を持つ
myMesh = hcubeMesh(T[0],leftX,leftY,rightX,rightY,lowX,lowY,upX,upY,h,True,True,tolMesh=1e-10,tolJoint=1)

# データセットの作成
MeshList = []
MeshList.append(myMesh)
train_set = VaryGeoDataset(MeshList)
training_data_loader = DataLoader(dataset=train_set, batch_size=batchSize)


# df/dx
def dfdx(f,dydeta,dydxi,Jinv):
    dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h
    dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
    dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
    dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
    dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h
    dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
    dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
    dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
    dfdx=Jinv*(dfdxi*dydeta-dfdeta*dydxi)
    return dfdx

# df/dy
def dfdy(f,dxdxi,dxdeta,Jinv):
    dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h
    dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
    dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
    dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
    dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h
    dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
    dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
    dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
    dfdy=Jinv*(dfdeta*dxdxi-dfdxi*dxdeta)
    return dfdy

# df/dt
def dfdt(T1,T0):
    dfdt=(T1-T0)/time_step
    return dfdt

# train関数
def train(epoch):
    startTime = time.time()
    Res = 0
    eV = 0
    for iteration, batch in enumerate(training_data_loader):
        [T0, JJinv, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta] = to4DTensor(batch)
        optimizer.zero_grad()
        output=model(T0)
        output_pad=udfpad(output)
        outputV=output_pad[:,0,:,:].reshape(output_pad.shape[0],1,output_pad.shape[2],output_pad.shape[3])

        for j in range(batchSize):
            outputV[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=0
            outputV[j,0,:padSingleSide,padSingleSide:-padSingleSide]=1
            outputV[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=1
            outputV[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=1
            outputV[j,0,0,0]=0.5*(outputV[j,0,0,1]+outputV[j,0,1,0])
            outputV[j,0,0,-1]=0.5*(outputV[j,0,0,-2]+outputV[j,0,1,-1])

        dvdx=dfdx(outputV,dydeta,dydxi,Jinv)
        d2vdx2=dfdx(dvdx,dydeta,dydxi,Jinv)
        dvdy=dfdy(outputV,dxdxi,dxdeta,Jinv)
        d2vdy2=dfdy(dvdy,dxdxi,dxdeta,Jinv)
        dvdt=dfdt(outputV,T0)
        continuity=torch.abs(dvdt-k*(d2vdy2+d2vdx2))
        loss=criterion(continuity,continuity*0)
        loss.backward()
        optimizer.step()
        loss_mass=criterion(continuity, continuity*0)
        Res+=loss_mass.item()
        CNNVNumpy=outputV[0,0,:,:].cpu().detach().numpy()
        eV=eV+np.sqrt(calMSE(T[1],CNNVNumpy)/calMSE(T[1],T[1]*0))

    print('Epoch is ', epoch)
    print("mRes Loss is", (Res / len(training_data_loader)))
    print("eV Loss is", (eV / len(training_data_loader)))

    # epoch数が5000の倍数,1500(=Epochs)の倍数,相対誤差が閾値以下になったら温度場の画像を保存
    # if epoch%5000==0 or epoch%nEpochs==0 or np.sqrt(calMSE(T[1],CNNVNumpy)/calMSE(T[1],T[1]*0))<0.1:
    if epoch % nEpochs == 0:
        torch.save(model, str(epoch)+'.pth')
        fig1=plt.figure()
        ax=plt.subplot(1,2,1)
        visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
                       coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
                       outputV[0,0,1:-1,1:-1].cpu().detach().numpy(),'horizontal',[0,1])
        setAxisLabel(ax,'p')
        ax.set_title('CNN '+r'$T$')
        ax.set_aspect('equal')
        ax=plt.subplot(1,2,2)
        visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
                       coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
                       T[1][1:-1,1:-1],'horizontal',[0,1])
        setAxisLabel(ax,'p')
        ax.set_aspect('equal')
        ax.set_title('FV '+r'$T$')
        fig1.tight_layout(pad=1)
        fig1.savefig(str(epoch)+'T.pdf',bbox_inches='tight')
        plt.close(fig1)
    return (Res/len(training_data_loader)),(eV/len(training_data_loader))

MRes=[]
EV=[]
TotalstartTime=time.time()

# 学習パート
for epoch in range(1,nEpochs+1):
    mres,ev=train(epoch)
    MRes.append(mres)
    EV.append(ev)
    '''
    if ev<0.1:
        break
    '''

TimeSpent=time.time()-TotalstartTime

plt.figure()
plt.plot(MRes,'-*',label='Equation Residual')
plt.xlabel('Epoch')
plt.ylabel('Residual')
plt.legend()
plt.yscale('log')
plt.savefig('convergence.pdf',bbox_inches='tight')
tikzplotlib.save('convergence.tikz')
plt.figure()
plt.plot(EV,'-x',label=r'$e_v$')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.yscale('log')
plt.savefig('error.pdf',bbox_inches='tight')
tikzplotlib.save('error.tikz')
EV=np.asarray(EV)
MRes=np.asarray(MRes)
np.savetxt('EV.txt',EV)
np.savetxt('MRes.txt',MRes)
np.savetxt('TimeSpent.txt',np.zeros([2,2])+TimeSpent)

##############################

# T_listには初期条件のT0とPhyGeoNetが予測するT1~T10の計11要素が入る
T_list = []
T0 = torch.tensor(T[0].reshape([1,1,T[0].shape[0],T[0].shape[1]]))
T_list.append(T0.float().to('cuda'))
for i in range(30):
    T_predict = model(T_list[i])
    T_predict = udfpad(T_predict)
    T_predict = T_predict[:, 0, :, :].reshape(T_predict.shape[0], 1, T_predict.shape[2], T_predict.shape[3])
    T_predict[0, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0
    T_predict[0, 0, :padSingleSide, padSingleSide:-padSingleSide] = 1
    T_predict[0, 0, padSingleSide:-padSingleSide, -padSingleSide:] = 1
    T_predict[0, 0, padSingleSide:-padSingleSide, 0:padSingleSide] = 1
    T_predict[0, 0, 0, 0] = 0.5 * (T_predict[0, 0, 0, 1] + T_predict[0, 0, 1, 0])
    T_predict[0, 0, 0, -1] = 0.5 * (T_predict[0, 0, 0, -2] + T_predict[0, 0, 1, -1])
    T_list.append(T_predict)

# 横軸(time),縦軸(予測したTとOpenFOAMのエラー)のグラフを作る
time_list = [time_step*i for i in range(31)]
eV_list = []
for i in range(31):
    CNNVNumpy = T_list[i][0, 0, :, :].cpu().detach().numpy()
    eV = np.sqrt(calMSE(T[i], CNNVNumpy) / calMSE(T[i], T[i] * 0))
    eV_list.append(eV)

plt.figure()
plt.plot(time_list,eV_list,label=r'$e_v$')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
plt.savefig('error_shift.pdf',bbox_inches='tight')

time_step = Decimal(str(0.01))

# T_listの要素(PhyGeoNetの出力)を画像(jpg)に変換して保存する
for i in range(31):
    fig = plt.figure()

    ax1 = plt.subplot(1, 2, 1)
    visualize2D(ax1, myMesh.x[1:-1, 1:-1], myMesh.y[1:-1, 1:-1], T_list[i][0, 0, 1:-1, 1:-1].cpu().detach().numpy(), 'horizontal', [0, 1])
    setAxisLabel(ax1, 'p')
    ax1.set_title('CNN ' + r'$T$'+' t='+str(i*time_step))
    ax1.set_aspect('equal')

    ax2 = plt.subplot(1, 2, 2)
    visualize2D(ax2, myMesh.x[1:-1,1:-1],myMesh.y[1:-1,1:-1],T[i][1:-1, 1:-1], 'horizontal', [0, 1])
    setAxisLabel(ax2, 'p')
    ax2.set_title('FV ' + r'$T$'+' t='+str(i*time_step))
    ax2.set_aspect('equal')

    fig.tight_layout(pad=1)
    fig.savefig(str(i) + 'T.jpg', bbox_inches='tight')
    plt.close(fig)

# T_listの要素-OpenFOAMをグレースケール画像にする
gray_list = [np.abs(T_list[i][0, 0, :, :].cpu().detach().numpy()-T[i]) for i in range(31)]
for i in range(31):
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    visualize2D(ax, myMesh.x[1:-1, 1:-1], myMesh.y[1:-1, 1:-1], gray_list[i], 'horizontal', [0, 1], cmap='gray_r')
    setAxisLabel(ax, 'p')
    ax.set_title('Gray Scale'+' t='+str(i*time_step))
    ax.set_aspect('equal')
    fig.tight_layout(pad=1)
    fig.savefig(str(i) + 'T_gray.jpg', bbox_inches='tight')
    plt.close(fig)

# 保存したOpenFOAMとPhyGeoNet出力の比較画像(jpg)を動画にする
size = (632,203)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
save = cv2.VideoWriter('video1.mp4',fourcc,2,size)

for i in range(len(glob.glob('*T.jpg'))):
    img = cv2.imread(str(i) + 'T.jpg')
    save.write(img)

save.release()

# 保存したT_listの要素-PhyGeoNetのグレースケール画像(jpg)を動画にする
size = (632,288)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
save = cv2.VideoWriter('video2.mp4',fourcc,2,size)

for i in range(len(glob.glob('*T_gray.jpg'))):
    img = cv2.imread(str(i) + 'T_gray.jpg')
    save.write(img)

save.release()