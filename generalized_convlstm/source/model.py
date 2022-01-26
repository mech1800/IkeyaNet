import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pdb
torch.manual_seed(123)

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=10, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)  #ベストモデルのstate_dictを指定したpathに保存
        torch.save(model, self.path)  # ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class USCNN(nn.Module):
    # k:カーネルサイズ,s:ストライド,p:パディング
    def __init__(self, sequence, h, nx, ny, nVarIn=1, nVarOut=1, initWay=None, k=5, s=1, p=2):
        super(USCNN, self).__init__()
        self.initWay = initWay
        self.nVarIn = nVarIn
        self.nVarOut = nVarOut
        self.k = k
        self.s = s
        self.p = p
        self.deltaX = h
        self.nx = nx
        self.ny = ny
        self.sequence = sequence

        # ネットワーク層や活性化関数を定義
        self.relu = nn.ReLU()
        self.US = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode='bicubic')
        # self.conv1 = nn.Conv2d(self.nVarIn, 16, kernel_size=k, stride=s, padding=p)
        self.convlstm = ConvLSTM(input_dim=self.nVarIn, hidden_dim=16, kernel_size=(k,k), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=k, stride=s, padding=p)
        self.convA = nn.Conv2d(32, 64, kernel_size=k, stride=s, padding=p)
        self.convB = nn.Conv2d(64, 32, kernel_size=k, stride=s, padding=p)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=k, stride=s, padding=p)
        self.conv4 = nn.Conv2d(16, self.nVarOut, kernel_size=k, stride=s, padding=p)
        self.pixel_shuffle = nn.PixelShuffle(1)

        # initWayの指定する方法で重みを初期化(l44)
        if self.initWay is not None:
            self._initialize_weights()

    # forwardを定義 lstmのinputは(B=2,T=sequence,C=nVarIn,H,W),outputは((B,T,C,H,W),(h,c))
    def forward(self, x):
        # upsample
        x = ([self.US(x[:,i,:,:,:]) for i in range(self.sequence)])
        x = torch.stack(x,dim=1)

        # convLSTMからconv2,3,pixel_shuffle
        x, _ = self.convlstm(x)
        x = self.relu(x[0])
        x_list = []
        for i in range(self.sequence):
            x_t = self.relu(self.conv2(x[:,i,:,:,:]))
            x_t = self.relu(self.convA(x_t))
            x_t = self.relu(self.convB(x_t))
            x_t = self.relu(self.conv3(x_t))
            x_t = self.pixel_shuffle(self.conv4(x_t))
            x_list.append(x_t)
        x = torch.stack(x_list, dim=1)
        return x

    def _initialize_weights(self):
        if self.initWay == 'kaiming':
            init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
            init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
            init.kaiming_normal_(self.conv4.weight)
        elif self.initWay == 'ortho':
            init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv4.weight)
        else:
            print('Only Kaiming or Orthogonal initializer can be used!')
            exit()