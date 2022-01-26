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
from dataset import VaryGeoDataset, FixGeoDataset
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh,setAxisLabel,np2cuda,to4DTensor
from model import USCNN
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

# dξ,dηのスケール
h = 0.01

myMesh = hcubeMesh(leftX,leftY,rightX,rightY,lowX,lowY,upX,upY,h,True,True,tolMesh=1e-10,tolJoint=1)


# 様々な変数を定義
NvarInput = 1
NvarOutput = 1
nEpochs = 100000
lr = 0.001
Ns = 1
nu = 0.01
# 熱伝導係数(OpenFOAMのtransportProperties)
k = 1
# sequence数
sequence = 30

model = torch.load('./checkpoint_model.pth')
model = model.cuda()
# model = model.cpu()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)


# idxをOpenFOAMのファイル名と一致させる関数
def decimal_normalize(f):
    # 指数表記を10進数表記に変える関数
    def _remove_exponent(d):
        return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()
    a = Decimal.normalize(Decimal(str(f)))
    b = _remove_exponent(a)
    return str(b)

# ファイル名を正しく読み取るためにDecimal関数を用いて10進法の0.01を得る
time_step = Decimal(str(0.01))

# 温度場を取得する
ParaList=[1,2,3,4,5,6,7]
caseName=['TemplateCase0','TemplateCase1','TemplateCase2','TemplateCase3','TemplateCase4','TemplateCase5','TemplateCase6']

T_practice = []

# 7つのTemplateCaseフォルダに1つずつアクセスする
for name in caseName:
    # TemplateCaseの中にある全てのファイル(初期条件1つ+sequence数)にアクセスしてリストに入れる
    filelist = []
    for i in range(sequence+1):
        idx = decimal_normalize(i * time_step)
        OFPic = convertOFMeshToImage_StructuredMesh(nx, ny, name+'/0/C', [name+'/' + idx + '/T'],[0, 1, 0, 1], 0.0, False)
        filelist.append(OFPic[:, :, 2])
    T_practice.append(filelist)


# テスト用のデータセットを作る
test_set = FixGeoDataset(myMesh,T_practice)
test_data_loader = DataLoader(dataset=test_set, batch_size=1)


# test_data_loader　→　batchsize(=1)×7
for iteration, batch in enumerate(test_data_loader):
    [T0, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta] = to4DTensor(batch)

    stack_list = [T0 for _ in range(sequence)]
    T_sequence = torch.stack(stack_list, dim=1)

    # モデルを呼び出しておく
    if iteration == 0:
        _ = model(T_sequence)

    optimizer.zero_grad()
    time_start = time.time()
    output = model(T_sequence)
    outputV = udfpad(output)

    for i in range(sequence):
        outputV[0, i, 0, -padSingleSide:, padSingleSide:-padSingleSide] = 0
        outputV[0, i, 0, :padSingleSide, padSingleSide:-padSingleSide] = ParaList[iteration]
        outputV[0, i, 0, padSingleSide:-padSingleSide, -padSingleSide:] = ParaList[iteration]
        outputV[0, i, 0, padSingleSide:-padSingleSide, :padSingleSide] = ParaList[iteration]
        outputV[0, i, 0, 0, 0] = 0.5 * (outputV[0, i, 0, 0, 1] + outputV[0, i, 0, 1, 0])
        outputV[0, i, 0, 0, -1] = 0.5 * (outputV[0, i, 0, 0, -2] + outputV[0, i, 0, 1, -1])

    print('全体の出力時間は'+str(time.time()-time_start)+'です')

    # ①横軸(1~10),縦軸(予測したTとOpenFOAMのエラー)のグラフを作る
    time_list = [time_step * (i + 1) for i in range(sequence)]
    eV_list = []
    for i in range(sequence):
        CNNVNumpy = outputV[0, i, 0, :, :].cpu().detach().numpy()
        eV = np.sqrt(calMSE(T_practice[iteration][i + 1], CNNVNumpy) / calMSE(T_practice[iteration][i + 1], T_practice[iteration][i + 1] * 0))
        eV_list.append(eV)

    plt.figure()
    plt.plot(time_list, eV_list, label=r'$e_v$')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('./TemplateCase'+str(iteration)+'_result/'+'error_shift.pdf', bbox_inches='tight')

    # ②outputV(PhyGeoNetの出力)を画像(jpg)に変換して保存する
    # T0
    fig = plt.figure()

    ax1 = plt.subplot(1, 2, 1)
    visualize2D(ax1, myMesh.x[1:-1, 1:-1], myMesh.y[1:-1, 1:-1], T_practice[iteration][0][1:-1, 1:-1], 'horizontal',[0, ParaList[iteration]])
    setAxisLabel(ax1, 'p')
    ax1.set_title('CNN ' + r'$T$' + ' t=' + str(0 * time_step))
    ax1.set_aspect('equal')

    ax2 = plt.subplot(1, 2, 2)
    visualize2D(ax2, myMesh.x[1:-1, 1:-1], myMesh.y[1:-1, 1:-1], T_practice[iteration][0][1:-1, 1:-1], 'horizontal',[0, ParaList[iteration]])
    setAxisLabel(ax2, 'p')
    ax2.set_title('FV ' + r'$T$' + ' t=' + str(0 * time_step))
    ax2.set_aspect('equal')

    fig.tight_layout(pad=1)
    fig.savefig('./TemplateCase'+str(iteration)+'_result/'+'0T.jpg', bbox_inches='tight')
    plt.close(fig)

    # T1~T10まで
    for i in range(sequence):
        fig = plt.figure()

        ax1 = plt.subplot(1, 2, 1)
        visualize2D(ax1, myMesh.x[1:-1, 1:-1], myMesh.y[1:-1, 1:-1],
                    outputV[0, i, 0, 1:-1, 1:-1].cpu().detach().numpy(), 'horizontal', [0, ParaList[iteration]])
        setAxisLabel(ax1, 'p')
        ax1.set_title('CNN ' + r'$T$' + ' t=' + str((i + 1) * time_step))
        ax1.set_aspect('equal')

        ax2 = plt.subplot(1, 2, 2)
        visualize2D(ax2, myMesh.x[1:-1, 1:-1], myMesh.y[1:-1, 1:-1], T_practice[iteration][i + 1][1:-1, 1:-1], 'horizontal', [0, ParaList[iteration]])
        setAxisLabel(ax2, 'p')
        ax2.set_title('FV ' + r'$T$' + ' t=' + str((i + 1) * time_step))
        ax2.set_aspect('equal')

        fig.tight_layout(pad=1)
        fig.savefig('./TemplateCase'+str(iteration)+'_result/'+str(i + 1) + 'T.jpg', bbox_inches='tight')
        plt.close(fig)

    # ③outputV-OpenFOAMをグレースケール画像にする
    gray_list = [np.zeros((ny,nx))]
    for i in range(sequence):
        gray_list.append(np.abs((outputV[0, i, 0, :, :].cpu().detach().numpy()-T_practice[iteration][i+1])/ParaList[iteration]))

    for i in range(sequence+1):
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        visualize2D(ax, myMesh.x[1:-1, 1:-1], myMesh.y[1:-1, 1:-1], gray_list[i], 'horizontal', [0, 1], cmap='gray_r')
        setAxisLabel(ax, 'p')
        ax.set_title('Gray Scale' + ' t=' + str(i * time_step))
        ax.set_aspect('equal')
        fig.tight_layout(pad=1)
        fig.savefig('./TemplateCase'+str(iteration)+'_result/'+str(i) + 'T_gray.jpg', bbox_inches='tight')
        plt.close(fig)

    # ④保存したOpenFOAMとPhyGeoNet出力の比較画像(jpg)を動画にする
    im = cv2.imread('./TemplateCase'+str(iteration)+'_result/'+'0T.jpg')
    size = (im.shape[1], im.shape[0])
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    save = cv2.VideoWriter('./TemplateCase'+str(iteration)+'_result/'+'video1.mp4', fourcc, 4, size)

    for i in range(len(glob.glob('./TemplateCase'+str(iteration)+'_result/'+'*T.jpg'))):
        img = cv2.imread('./TemplateCase'+str(iteration)+'_result/'+str(i) + 'T.jpg')
        save.write(img)

    save.release()

    # ⑤保存したT_listの要素-PhyGeoNetのグレースケール画像(jpg)を動画にする
    im = cv2.imread('./TemplateCase' + str(iteration) + '_result/' + '0T_gray.jpg')
    size = (im.shape[1], im.shape[0])
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    save = cv2.VideoWriter('./TemplateCase'+str(iteration)+'_result/'+'video2.mp4', fourcc, 4, size)

    for i in range(len(glob.glob('./TemplateCase'+str(iteration)+'_result/'+'*T_gray.jpg'))):
        img = cv2.imread('./TemplateCase'+str(iteration)+'_result/'+str(i) + 'T_gray.jpg')
        save.write(img)