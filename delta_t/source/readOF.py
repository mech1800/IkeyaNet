from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
import pdb
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from torch.utils.data import Dataset, DataLoader
from foamFileOperation import readVectorFromFile,readScalarFromFile

def convertOFMeshToImage_StructuredMesh(nx,ny,MeshFile,FileName,ext,mriLevel=0,plotFlag=True):
    title = ['x', 'y']
    OFVector = None
    OFScalar = None

    # FileNameのファイル名からファイルが保持する物理量を判別し抽出する
    for i in range(len(FileName)):
        if FileName[i][-1] == 'U':
            OFVector = readVectorFromFile(FileName[i])
            title.append('u')
            title.append('v')
        elif FileName[i][-1] == 'p':
            OFScalar = readScalarFromFile(FileName[i])
            title.append('p')
        elif FileName[i][-1] == 'T':
            OFScalar = readScalarFromFile(FileName[i])
            title.append('T')
        elif FileName[i][-1] == 'f':
            OFScalar = readScalarFromFile(FileName[i])
            title.append('f')
        else:
            print('Variable name is not clear')
            exit()

    # (x,y,無し)のような座標データを作成
    OFMesh = readVectorFromFile(MeshFile)

    # (x,y,物理量)のような座標データ+物理量を作成
    Ng = OFMesh.shape[0]
    nVar = len(title)
    OFCase = np.zeros([Ng, nVar])
    # (x,y,物理量)の(x,y)を代入
    OFCase[:, 0:2] = np.copy(OFMesh[:, 0:2])
    # (x,y,物理量)の(物理量)を代入
    if OFVector is not None and OFScalar is not None:
        if mriLevel > 1e-16:
            OFVector = foamFileAddNoise.addMRINoise(OFVector, mriLevel)
        OFCase[:, 2:4] = np.copy(OFVector[:, 0:2])
        OFCase[:, 4] = np.copy(OFScalar)
    elif OFScalar is not None:
        OFCase[:, 2] = np.copy(OFScalar)

    # OFCase(x,y,物理量)を座標単位のリストに変形する
    OFPic = np.reshape(OFCase, (ny, nx, nVar), order='F')

    # 本来はplotFlag==Trueで図を表示したかったはず
    if plotFlag:
        pass

    return OFPic