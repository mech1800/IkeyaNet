from torch.utils.data import Dataset, DataLoader
import pdb
import numpy as np

class VaryGeoDataset(Dataset):
    def __init__(self, MeshList):
        self.MeshList = MeshList

    def __len__(self):
        return len(self.MeshList)

    def __getitem__(self, idx):
        # メッシュリストに入っているi番目のhcubeMeshインスタンスのインスタンス変数(T0,J,Jinv,dxdxi,dydxi,dxdeta,dydeta)を返す
        mesh = self.MeshList[idx]
        T0 = mesh.T0

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
        return [T0,InvariantInput,coord,xi,eta,J,Jinv,dxdxi,dydxi,dxdeta,dydeta]