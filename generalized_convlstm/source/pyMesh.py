import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch

# 何度も使う文字列を変数として定義
arrow = '====>'
errMessageJoint = 'The geometry is not closed!'
errMessageParallel = 'The parallel sides do not have the same number of node!'
errMessageXYShape = 'The x y shapes do not have match each other!'
errMessageDomainType = 'domainType can only be physical domain or reference domain!'
clow = 'green'
cup = 'blue'
cright = 'red'
cleft = 'orange'
cinternal = 'black'

# 入力したリストのアイテムをgpuに載せて返す関数
def np2cuda(myList):
    MyList = []
    for item in myList:
        MyList.append(item.to('cuda'))
        # MyList.append(item.to('cpu'))
    return MyList


# mylistの中身を4次元に揃える関数
def to4DTensor(myList):
    MyList = []
    for item in myList:
        if len(item.shape) == 3:
            item = torch.tensor(item.reshape([item.shape[0], 1, item.shape[1], item.shape[2]]))
            MyList.append(item.float().to('cuda'))
            # MyList.append(item.float().to('cpu'))
        else:
            item = torch.tensor(item)
            MyList.append(item.float().to('cuda'))
            # MyList.append(item.float().to('cpu'))
    return MyList


# 正しい四角形であることを確認する関数
def checkGeo(leftX, leftY, rightX, rightY, lowX, lowY, upX, upY, tolJoint):
    # 開始
    print(arrow + 'Check bc nodes!')

    # 座標データが1次元であることを確認
    assert len(leftX.shape) == len(leftY.shape) == len(rightX.shape) == len(rightY.shape) == len(lowX.shape) == \
           len(lowY.shape) == len(upX.shape) == len(upY.shape) == 1, 'all left(right)X(Y) must be 1d vector!'

    # 領域の角(joint)が一致していることを確認
    assert np.abs(leftX[0] - lowX[0]) < tolJoint, errMessageJoint
    assert np.abs(leftX[-1] - upX[0]) < tolJoint, errMessageJoint
    assert np.abs(rightX[0] - lowX[-1]) < tolJoint, errMessageJoint
    assert np.abs(rightX[-1] - upX[-1]) < tolJoint, errMessageJoint
    assert np.abs(leftY[0] - lowY[0]) < tolJoint, errMessageJoint
    assert np.abs(leftY[-1] - upY[0]) < tolJoint, errMessageJoint
    assert np.abs(rightY[0] - lowY[-1]) < tolJoint, errMessageJoint
    assert np.abs(rightY[-1] - upY[-1]) < tolJoint, errMessageJoint

    # 四角形の対辺が同じ格子点数であることを確認
    assert leftX.shape == leftY.shape == rightX.shape == rightY.shape, errMessageParallel

    # 終了
    print(arrow + 'BC nodes pass!')


# 境界に色を付けてプロット
def plotBC(ax, x, y):
    ax.plot(x[:, 0], y[:, 0], '-o', color=cleft)
    ax.plot(x[:, -1], y[:, -1], '-o', color=cright)
    ax.plot(x[0, :], y[0, :], '-o', color=clow)
    ax.plot(x[-1, :], y[-1, :], '-o', color=cup)
    return ax


# axにメッシュの境界を作る
def plotMesh(ax, x, y, width=0.05):
    [ny, nx] = x.shape
    for j in range(0, nx):
        ax.plot(x[:, j], y[:, j], color=cinternal, linewidth=width)
    for i in range(0, ny):
        ax.plot(x[i, :], y[i, :], color=cinternal, linewidth=width)
    return ax


# axに軸のラベルを作る
def setAxisLabel(ax, type):
    if type == 'p':
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
    elif type == 'r':
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\eta$')
    else:
        raise ValueError('The axis type only can be reference or physical')


# ラプラス方程式を解いてメッシュを切る
def ellipticMap(x, y, h, tol):
    assert x.shape == y.shape, errMessageXYShape
    [ny, nx] = x.shape
    A = np.ones([ny - 2, nx - 2])
    B = A
    C = A
    eps = 2.2e-16
    ite = 1

    while True:
        X = (A * (x[2:, 1:-1] + x[0:-2, 1:-1]) + C * (x[1:-1, 2:] + x[1:-1, 0:-2]) - B / 2 * (
                    x[2:, 2:] + x[0:-2, 0:-2] - x[2:, 0:-2] - x[0:-2, 2:])) / 2 / (A + C)
        Y = (A * (y[2:, 1:-1] + y[0:-2, 1:-1]) + C * (y[1:-1, 2:] + y[1:-1, 0:-2]) - B / 2 * (
                    y[2:, 2:] + y[0:-2, 0:-2] - y[2:, 0:-2] - y[0:-2, 2:])) / 2 / (A + C)
        err = np.max(np.max(np.abs(x[1:-1, 1:-1] - X))) + np.max(np.max(np.abs(y[1:-1, 1:-1] - Y)))

        x[1:-1, 1:-1] = X
        y[1:-1, 1:-1] = Y
        A = ((x[1:-1, 2:] - x[1:-1, 0:-2]) / 2 / h) ** 2 + ((y[1:-1, 2:] - y[1:-1, 0:-2]) / 2 / h) ** 2 + eps
        B = (x[2:, 1:-1] - x[0:-2, 1:-1]) / 2 / h * (x[1:-1, 2:] - x[1:-1, 0:-2]) / 2 / h + (
                    y[2:, 1:-1] - y[0:-2, 1:-1]) / 2 / h * (y[1:-1, 2:] - y[1:-1, 0:-2]) / 2 / h + eps
        C = ((x[2:, 1:-1] - x[0:-2, 1:-1]) / 2 / h) ** 2 + ((y[2:, 1:-1] - y[0:-2, 1:-1]) / 2 / h) ** 2 + eps

        if err < tol:
            print('The mesh generation reaches covergence!')
            break
        if ite > 50000:
            print('The mesh generation not reaches covergence ' + 'within 50000 iterations! The current resdiual is ')
            print(err)
            break
        ite = ite + 1
    return x, y


def gen_e2vcg(x):
    nelem = (x.shape[0] - 1) * (x.shape[1] - 1)
    nelemx = x.shape[1] - 1
    nelemy = x.shape[0] - 1
    nnx = x.shape[1]
    nny = x.shape[0]
    e2vcg0 = np.zeros([4, nelem])
    e2vcg = np.zeros([4, nelem])
    for j in range(nelemy):
        for i in range(nelemx):
            e2vcg[:, j * nelemx + i] = np.asarray(
                [j * nnx + i, j * nnx + i + 1, (j + 1) * nnx + i, (j + 1) * nnx + i + 1])
    return e2vcg.astype('int')


# axとcbarを返す
def visualize2D(ax, x, y, u, colorbarPosition='vertical', colorlimit=None, cmap=matplotlib.cm.coolwarm):
    xdg0 = np.vstack([x.flatten(order='C'), y.flatten(order='C')])
    udg0 = u.flatten(order='C')
    idx = np.asarray([0, 1, 3, 2])
    nelem = (x.shape[0] - 1) * (x.shape[1] - 1)
    nelemx = x.shape[1] - 1;
    nelemy = x.shape[0] - 1;
    nelem = nelemx * nelemy
    nnx = x.shape[1];
    nny = x.shape[0]
    e2vcg0 = gen_e2vcg(x)
    udg_ref = udg0[e2vcg0]
    polygon_list = []
    for i in range(nelem):
        polygon_ = Polygon(xdg0[:, e2vcg0[idx, i]].T)
        polygon_list.append(polygon_)
    polygon_ensemble = PatchCollection(polygon_list, cmap=cmap, alpha=1)
    polygon_ensemble.set_edgecolor('face')
    polygon_ensemble.set_array(np.mean(udg_ref, axis=0))
    if colorlimit is None:
        pass
    else:
        polygon_ensemble.set_clim(colorlimit)
    ax.add_collection(polygon_ensemble)
    ax.set_xlim(np.min(xdg0[0, :]), np.max(xdg0[0, :]))
    # ax.set_xticks([np.min(xdg0[0,:]),np.max(xdg0[0,:])])
    ax.set_ylim(np.min(xdg0[1, :]), np.max(xdg0[1, :]))
    # ax.set_yticks([np.min(xdg0[1,:]),np.max(xdg0[1,:])])
    # ax.set_aspect('equal')
    cbar = plt.colorbar(polygon_ensemble, orientation=colorbarPosition)
    return ax, cbar


class hcubeMesh(object):
    def __init__(self, leftX, leftY, rightX, rightY, lowX, lowY, upX, upY, h,
                 plotFlag=False, saveFlag=False, saveDir='./mesh.pdf', tolMesh=1e-8, tolJoint=1e-6):
        self.h = h
        self.tolMesh = tolMesh
        self.tolJoint = tolJoint
        self.plotFlag = plotFlag
        self.saveFlag = saveFlag

        # 正しい四角形であることを確認
        checkGeo(leftX, leftY, rightX, rightY, lowX, lowY, upX, upY, tolJoint)

        # 格子点数を取得
        self.ny = leftX.shape[0]
        self.nx = upX.shape[0]

        # 内部の座標マップを作る(self.x,y,xi,etaはphygeonetの名残でインスタンス変数表記になっているが他の部分で呼び出されることはない)
        self.x = np.zeros([self.ny, self.nx])
        self.y = np.zeros([self.ny, self.nx])
        self.x[:, 0] = leftX
        self.x[:, -1] = rightX
        self.x[0, :] = lowX
        self.x[-1, :] = upX
        self.y[:, 0] = leftY
        self.y[:, -1] = rightY
        self.y[0, :] = lowY
        self.y[-1, :] = upY
        self.x, self.y = ellipticMap(self.x, self.y, self.h, self.tolMesh)

        # dξ,dηの生成
        eta, xi = np.meshgrid(np.linspace(0, self.ny - 1, self.ny), np.linspace(0, self.nx - 1, self.nx), indexing='ij')
        self.xi = xi * h
        self.eta = eta * h

        # 座標マップの作成
        # (x,y)
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        plotBC(ax, self.x, self.y)
        plotMesh(ax, self.x, self.y)
        setAxisLabel(ax, 'p')
        ax.set_aspect('equal')
        ax.set_title('Physics Domain Mesh')
        # (ξ,η)
        ax = fig.add_subplot(1, 2, 2)
        plotBC(ax, self.xi, self.eta)
        plotMesh(ax, self.xi, self.eta)
        setAxisLabel(ax, 'r')
        ax.set_aspect('equal')
        ax.set_title('Reference Domain Mesh')

        # 図のレイアウトを指定
        fig.tight_layout(pad=1)

        # saveDirに図を保存するか
        if saveFlag:
            plt.savefig(saveDir, bbox_inches='tight')
        # 図を表示するか
        if plotFlag:
            plt.show()

        plt.close(fig)

        # 4次の中心差分(上から順に内部のdx/dξ,dy/dξ,dx/dη,dy/dη)
        dxdxi_ho_internal = (-self.x[:, 4:] + 8 * self.x[:, 3:-1] - \
                             8 * self.x[:, 1:-3] + self.x[:, 0:-4]) / 12 / self.h
        dydxi_ho_internal = (-self.y[:, 4:] + 8 * self.y[:, 3:-1] - \
                             8 * self.y[:, 1:-3] + self.y[:, 0:-4]) / 12 / self.h
        dxdeta_ho_internal = (-self.x[4:, :] + 8 * self.x[3:-1, :] - \
                              8 * self.x[1:-3, :] + self.x[0:-4, :]) / 12 / self.h
        dydeta_ho_internal = (-self.y[4:, :] + 8 * self.y[3:-1, :] - \
                              8 * self.y[1:-3, :] + self.y[0:-4, :]) / 12 / self.h

        # 4次の片側差分(上から順に左境界のdx/dξ,右境界のdx/dξ,左境界のdy/dξ,右境界のdy/dξ)
        dxdxi_ho_left = (-11 * self.x[:, 0:-3] + 18 * self.x[:, 1:-2] - \
                         9 * self.x[:, 2:-1] + 2 * self.x[:, 3:]) / 6 / self.h
        dxdxi_ho_right = (11 * self.x[:, 3:] - 18 * self.x[:, 2:-1] + \
                          9 * self.x[:, 1:-2] - 2 * self.x[:, 0:-3]) / 6 / self.h
        dydxi_ho_left = (-11 * self.y[:, 0:-3] + 18 * self.y[:, 1:-2] - \
                         9 * self.y[:, 2:-1] + 2 * self.y[:, 3:]) / 6 / self.h
        dydxi_ho_right = (11 * self.y[:, 3:] - 18 * self.y[:, 2:-1] + \
                          9 * self.y[:, 1:-2] - 2 * self.y[:, 0:-3]) / 6 / self.h

        # 4次の片側差分(上から順に下境界のdx/dη,上境界のdx/dη,下境界のdy/dη,上境界のdy/dη)
        dxdeta_ho_low = (-11 * self.x[0:-3, :] + 18 * self.x[1:-2, :] - \
                         9 * self.x[2:-1, :] + 2 * self.x[3:, :]) / 6 / self.h
        dxdeta_ho_up = (11 * self.x[3:, :] - 18 * self.x[2:-1, :] + \
                        9 * self.x[1:-2, :] - 2 * self.x[0:-3, :]) / 6 / self.h
        dydeta_ho_low = (-11 * self.y[0:-3, :] + 18 * self.y[1:-2, :] - \
                         9 * self.y[2:-1, :] + 2 * self.y[3:, :]) / 6 / self.h
        dydeta_ho_up = (11 * self.y[3:, :] - 18 * self.y[2:-1, :] + \
                        9 * self.y[1:-2, :] - 2 * self.y[0:-3, :]) / 6 / self.h

        # 4次のdx/dξ
        self.dxdxi_ho = np.zeros(self.x.shape)
        self.dxdxi_ho[:, 2:-2] = dxdxi_ho_internal
        self.dxdxi_ho[:, 0:2] = dxdxi_ho_left[:, 0:2]
        self.dxdxi_ho[:, -2:] = dxdxi_ho_right[:, -2:]

        # 4次のdy/dξ
        self.dydxi_ho = np.zeros(self.y.shape)
        self.dydxi_ho[:, 2:-2] = dydxi_ho_internal
        self.dydxi_ho[:, 0:2] = dydxi_ho_left[:, 0:2]
        self.dydxi_ho[:, -2:] = dydxi_ho_right[:, -2:]

        # 4次のdx/dη
        self.dxdeta_ho = np.zeros(self.x.shape)
        self.dxdeta_ho[2:-2, :] = dxdeta_ho_internal
        self.dxdeta_ho[0:2, :] = dxdeta_ho_low[0:2, :]
        self.dxdeta_ho[-2:, :] = dxdeta_ho_up[-2:, :]

        # 4次のdy/dη
        self.dydeta_ho = np.zeros(self.y.shape)
        self.dydeta_ho[2:-2, :] = dydeta_ho_internal
        self.dydeta_ho[0:2, :] = dydeta_ho_low[0:2, :]
        self.dydeta_ho[-2:, :] = dydeta_ho_up[-2:, :]

        # 行列式
        self.J_ho = self.dxdxi_ho * self.dydeta_ho - self.dxdeta_ho * self.dydxi_ho
        self.Jinv_ho = 1 / self.J_ho
