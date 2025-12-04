import os
import torch
from TransGAT.models import GRU, LSTM, Transformer, DLinear, Informer, PatchTST, iTransformer, TimeXer, AGSTNet, \
    AGSTNetWoGraphTemporal, TCN, BiGRU, BiLSTM, AGSTNetWoTemporal, AGSTNetWoTurbineFea, AGSTNetWoFeaFea, AGSTNetTrial, \
    STGRU, STLSTM, STDLinear, STiTransformer, AGSTNetWiWeight, AGSTNetGAT, STGCN, DCRNN, AGCRN, AGSTNetWaveletDecomp, \
    AGSTNetMultiDecomp


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'GRU': GRU,
            'LSTM': LSTM,
            'BiGRU': BiGRU,
            'BiLSTM': BiLSTM,
            'TCN': TCN,
            'Transformer': Transformer,
            'DLinear': DLinear,
            'Informer': Informer,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'STiTransformer': STiTransformer,
            'TimeXer': TimeXer,
            'AGSTNet': AGSTNet,
            'AGSTNetWiWeight': AGSTNetWiWeight,
            'AGSTNetGAT': AGSTNetGAT,
            'AGSTNetWoGraphTemporal': AGSTNetWoGraphTemporal,
            'AGSTNetWoTemporal': AGSTNetWoTemporal,
            'AGSTNetWoTurbineFea': AGSTNetWoTurbineFea,
            'AGSTNetWoFeaFea': AGSTNetWoFeaFea,
            'AGSTNetWaveletDecomp': AGSTNetWaveletDecomp,
            'AGSTNetMultiDecomp':AGSTNetMultiDecomp,
            'AGSTNetTrial': AGSTNetTrial,
            'STGRU': STGRU,
            'STLSTM': STLSTM,
            'STDLinear': STDLinear,
            'STGCN': STGCN,
            'DCRNN': DCRNN,
            'AGCRN': AGCRN

        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
