from torch.utils.data import Dataset, DataLoader
import pdb
import numpy as np

# case0
class VaryGeoDataset(Dataset):
    def __init__(self, MeshList, T):
        self.MeshList = MeshList
        self.T = T

    def __len__(self):
        return len(self.MeshList)

    def __getitem__(self, idx):
        # メッシュリストに入っているi番目のhcubeMeshインスタンスのインスタンス変数(T0,J,Jinv,dxdxi,dydxi,dxdeta,dydeta)を返す
        mesh = self.MeshList[idx]

        # 初期条件の温度場を取得
        T0 = self.T[0]

        J = mesh.J_ho
        Jinv = mesh.Jinv_ho
        InvariantInput = np.zeros([2, J.shape[0], J.shape[1]])  # 2,84,19の0行列
        InvariantInput[0, :, :] = J
        InvariantInput[1, :, :] = Jinv  # 1要素目にヤコビアンの行列式J、2要素目にJの逆数

        x = mesh.x
        y = mesh.y
        xi = mesh.xi
        eta = mesh.eta
        coord = np.zeros([2, x.shape[0], x.shape[1]])  # 2,84,19の0行列
        coord[0, :, :] = x;
        coord[1, :, :] = y  # 1要素目にx(形状は物理領域=長方形じゃない)、2要素目にy(形状は物理領域)

        dxdxi = mesh.dxdxi_ho
        dydxi = mesh.dydxi_ho
        dxdeta = mesh.dxdeta_ho
        dydeta = mesh.dydeta_ho
        return [T0, InvariantInput, coord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta]

# case2
class FixGeoDataset(Dataset):
    def __init__(self, mesh, T_practice):
        self.T_practice = T_practice
        self.mesh = mesh

    def __len__(self):
        return len(self.T_practice)

    def __getitem__(self, idx):
        mesh = self.mesh

        x = mesh.x
        y = mesh.y
        xi = mesh.xi
        eta = mesh.eta
        cord = np.zeros([2, x.shape[0], x.shape[1]])
        cord[0, :, :] = x;
        cord[1, :, :] = y

        J = mesh.J_ho
        Jinv = mesh.Jinv_ho

        dxdxi = mesh.dxdxi_ho
        dydxi = mesh.dydxi_ho
        dxdeta = mesh.dxdeta_ho
        dydeta = mesh.dydeta_ho

        return [self.T_practice[idx][0], cord, xi, eta, J, Jinv, dxdxi, dydxi, dxdeta, dydeta]