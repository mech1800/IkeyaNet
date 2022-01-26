import sys
import numpy
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
from dataset import FixGeoDataset
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh, setAxisLabel,to4DTensor
from model import USCNN, EarlyStopping
from readOF import convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp
from decimal import Decimal, Context
import cv2
import glob

# 時刻0における境界の座標(x,y)を取得
OFBCCoord = Ofpp.parse_boundary_field('TemplateCase0/0/C')
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

time_step = Decimal(str(0.01))

# 境界条件T=1について正解データを作成(今回はtime_step=0.05で初期条件+10個用意する)
T_boundary1 = []
for i in range(31):
    idx = decimal_normalize(i*time_step)
    OFPic = convertOFMeshToImage_StructuredMesh(nx,ny,'TemplateCase0/0/C',['TemplateCase0/'+idx+'/T'],[0,1,0,1],0.0,False)
    T_boundary1.append(OFPic[:,:,2])

# 境界条件T=7について正解データを作成(今回はtime_step=0.05で初期条件+10個用意する)
T_boundary7 = []
for i in range(31):
    idx = decimal_normalize(i*time_step)
    OFPic = convertOFMeshToImage_StructuredMesh(nx,ny,'TemplateCase6/0/C',['TemplateCase6/'+idx+'/T'],[0,1,0,1],0.0,False)
    T_boundary7.append(OFPic[:,:,2])

# batchsize=2のdatasetに入れるため
T_practice = [T_boundary1, T_boundary7]

# hardBCのときに使う
ParaList=[1,7]

time_step = 0.01


# dξ,dηのスケール
h = 0.01

# 様々な変数を定義
NvarInput = 1
NvarOutput = 1
batchSize = 2   # T=1,7
nEpochs = 10000
lr = 0.001
Ns = 1
nu = 0.01
# 熱伝導係数(OpenFOAMのtransportProperties)
k = 1
# sequence数
sequence = 30

# モデルの定義
model = USCNN(sequence, h,nx,ny,NvarInput,NvarOutput).to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
earlystopping = EarlyStopping(patience=1000)

# パディングを与えるメソッドの定義
padSingleSide = 1
udfpad = nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)


# myMeshはインスタンス変数として(x,y),(ξ,η)の座標やdx/dξ,dy/dξ,dx/dη,dy/dηや行列式を持つ
myMesh = hcubeMesh(leftX,leftY,rightX,rightY,lowX,lowY,upX,upY,h,True,True,tolMesh=1e-10,tolJoint=1)

# データセットの作成
train_set = FixGeoDataset(myMesh,T_practice)
training_data_loader = DataLoader(dataset=train_set, batch_size=batchSize)

# df/dx
def dfdx(f,dydeta,dydxi,Jinv):
    dfdxi_internal = (-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h
    dfdxi_left = (-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
    dfdxi_right = (11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
    dfdxi = torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
    dfdeta_internal = (-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h
    dfdeta_low = (-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
    dfdeta_up = (11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
    dfdeta = torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
    dfdx = Jinv*(dfdxi*dydeta-dfdeta*dydxi)
    return dfdx

# df/dy
def dfdy(f,dxdxi,dxdeta,Jinv):
    dfdxi_internal = (-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h
    dfdxi_left = (-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
    dfdxi_right = (11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
    dfdxi = torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
    dfdeta_internal = (-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h
    dfdeta_low = (-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
    dfdeta_up = (11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
    dfdeta = torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
    dfdy = Jinv*(dfdeta*dxdxi-dfdxi*dxdeta)
    return dfdy

# df/dt
def dfdt(T1,T0):
    dfdt = (T1-T0)/time_step
    return dfdt

'''
# df/dt
def dfdt(f,i):
    # 3次精度の左側片側差分
    if i == 0 or i == 1:
        dfdt = (-11*f[:,i,:,:,:]+18*f[:,i+1,:,:,:]-9*f[:,i+2,:,:,:]+2*f[:,i+3,:,:,:])/(6*time_step)
    # 3次精度の右側片側差分
    elif i == sequence-2 or i == sequence-1:
        dfdt = (11*f[:,i,:,:,:]-18*f[:,i-1,:,:,:]+9*f[:,i-2,:,:,:]-2*f[:,i-3,:,:,:])/(6*time_step)
    # 4次精度の中心差分
    else:
        dfdt = (-f[:,i+2,:,:,:]+8*f[:,i+1,:,:,:]-8*f[:,i-1,:,:,:]+f[:,i-2,:,:,:])/(12*time_step)
    return dfdt
'''

# train関数
def train(epoch, outputV):
    startTime = time.time()
    Res = 0
    eV_n = 0
    for iteration, batch in enumerate(training_data_loader):
        [T0, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta] = to4DTensor(batch)

        '''
        # 入力を可変にする場合
        # T1'~T9'がないなら境界条件の最小値(0)~最大値(1)でランダムに作る(torch.tensor型)
        if not outputV:
            outputV = torch.rand(1,sequence-1,1,ny,nx).to('cuda')
        # T1'~T9'とT0を結合する
        stack_list = [T0]
        for i in range(sequence-1):
            output = outputV[:,i,:,:,:]
            stack_list.append(output)
        '''

        stack_list = [T0 for _ in range(sequence)]
        T_sequence = torch.stack(stack_list, dim=1)

        optimizer.zero_grad()
        output=model(T_sequence)
        outputV=udfpad(output)

        for i in range(batchSize):
            for j in range(sequence):
                outputV[i,j,0,-padSingleSide:,padSingleSide:-padSingleSide] = 0
                outputV[i,j,0,:padSingleSide,padSingleSide:-padSingleSide] = ParaList[i]
                outputV[i,j,0,padSingleSide:-padSingleSide,-padSingleSide:] = ParaList[i]
                outputV[i,j,0,padSingleSide:-padSingleSide,:padSingleSide] = ParaList[i]
                outputV[i,j,0,0,0] = 0.5*(outputV[i,j,0,0,1] + outputV[i,j,0,1,0])
                outputV[i,j,0,0,-1] = 0.5*(outputV[i,j,0,0,-2] + outputV[i,j,0,1,-1])

        continuity = 0
        for i in range(sequence):
            dvdx_i = dfdx(outputV[:,i,:,:,:],dydeta,dydxi,Jinv)
            d2vdx2_i = dfdx(dvdx_i,dydeta,dydxi,Jinv)
            dvdy_i = dfdy(outputV[:,i,:,:,:],dxdxi,dxdeta,Jinv)
            d2vdy2_i = dfdy(dvdy_i,dxdxi,dxdeta,Jinv)
            if i == 0:
                dvdt_i = dfdt(outputV[:,i,:,:,:], T0)
            else:
                dvdt_i = dfdt(outputV[:,i,:,:,:], outputV[:,i-1,:,:,:])
            # 4次精度の時間微分
            # dvdt_i = dfdt(outputV,i)
            continuity_i = torch.abs(dvdt_i-k*(d2vdy2_i+d2vdx2_i))
            continuity += continuity_i

        loss = criterion(continuity, continuity * 0)
        loss.backward()
        optimizer.step()

        loss_mass = criterion(continuity, continuity * 0)
        Res += loss_mass.item()

        # eV_nは未定 今はeV_n=0
        # CNNVNumpy = outputV[0,-1,0,:,:].cpu().detach().numpy()

    # EarlyStoppingのcallメソッドを呼び出す
    earlystopping(Res, model)

    print('Epoch is ', epoch)
    print("mRes Loss is", (Res/len(training_data_loader)))
    print("eV_n Loss is", (eV_n/len(training_data_loader)))

    # train関数終わり
    return (Res/len(training_data_loader)),(eV_n/len(training_data_loader))

outputV = []
MRes=[]
EV_n=[]
TotalstartTime=time.time()

# 学習パート
for epoch in range(1,nEpochs+1):
    mres,ev_n=train(epoch,outputV)
    MRes.append(mres)
    EV_n.append(ev_n)
    if earlystopping.early_stop:
        print('Early Stopping!')
        break

TimeSpent=time.time()-TotalstartTime
print('学習にかかった時間は{}秒です'.format(TimeSpent))

plt.figure()
plt.plot(MRes,'-*',label='Equation Residual')
plt.xlabel('Epoch')
plt.ylabel('Residual')
plt.legend()
plt.yscale('log')
plt.savefig('convergence.pdf',bbox_inches='tight')
tikzplotlib.save('convergence.tikz')

plt.figure()
plt.plot(EV_n,'-x',label=r'$ev_n$')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.yscale('log')
plt.savefig('n_error.pdf',bbox_inches='tight')
tikzplotlib.save('n_error.tikz')

EV=np.asarray(EV_n)
MRes=np.asarray(MRes)
np.savetxt('EV.txt',EV)
np.savetxt('MRes.txt',MRes)