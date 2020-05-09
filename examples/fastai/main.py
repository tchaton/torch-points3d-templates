from utils import load_torch_points3d
load_torch_points3d()
import os
import sys
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_points3d.applications.pointnet2 import PointNet2
from torch_points3d.applications.kpconv import KPConv
from torch_points3d.applications.rsconv import RSConv
from torch_points3d.datasets.segmentation.shapenet import ShapeNetDataset
from torch_points3d.core.common_modules.base_modules import Seq
from torch_points3d.core.common_modules.dense_modules import Conv1D
from torch_points3d.core.data_transform import AddOnes

from fastai.basics import *

"""
Conclusion: FastAI Framework can be used, but seems pretty static in its implementation, creating a strong limitation for more complex workflow.
"""

AVAILABLE_MODELS = ["pointnet2", "kpconv", "rsconv"]

class DatasetWrapper():

    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, idx):
        data = next(iter(self._dataset))
        out = {"pos":data.pos}
        if data.batch is not None: out["batch"] = data.batch
        return out, data.y

    def __len__(self):
        return len(self._dataset)


class Net(nn.Module):

    def __init__(self, params, num_classes):
        super(Net, self).__init__()
        self._model = nn.ModuleDict()
        self._model["backbone"], self._model_opt, self._backbone_name = self.build_backbone(params)
        self._model["classifier"] = Seq()
        self._model["classifier"].append(Conv1D(self._model_opt.output_nc, num_classes, activation=None, bias=True, bn=False))

    def forward(self, input, *args):
        if self._backbone_name == "kpconv":
            data = Data(pos=input["pos"].squeeze(), batch=input["batch"].squeeze(), x=torch.ones((input["pos"].shape[1], 1)))
        else:
            data = Data(pos=input["pos"].squeeze(), x=input["pos"].squeeze())
        data_out = self._model["backbone"](data)
        if data_out.x.dim() == 2: data_out.x = data_out.x.permute(1, 0).unsqueeze(dim=0)
        out =  F.log_softmax(self._model["classifier"](data_out.x).permute(0, 2, 1), dim=-1)
        return out

    @staticmethod
    def build_backbone(params):
        model = nn.ModuleDict()
        backbone_name = sys.argv[1] if len(sys.argv) == 2 else "pointnet2"
        assert backbone_name in AVAILABLE_MODELS, "model_name should within {}".format(AVAILABLE_MODELS)
        model_opt = getattr(params.models, backbone_name)
        backbone_builder = globals().copy()[model_opt.class_name]
        backbone_builder = globals().copy()[model_opt.class_name]
        return backbone_builder(architecture="unet", 
                                input_nc=model_opt.input_nc, 
                                in_feat=model_opt.in_feat,
                                num_layers=model_opt.num_layers,
                                output_nc=model_opt.output_nc,
                                multiscale=True), model_opt, backbone_name

class CustomLoss:

    def __init__(self):
        self._loss = nn.NLLLoss() 

    def __call__(self, pred, target):
        return self._loss(pred.view((-1, pred.shape[-1])), target.flatten())

def main(params):

    dataset = ShapeNetDataset(params.data)
    model = Net(params, dataset.num_classes) 
    dataset.create_dataloaders(model._model_opt, params.training.batch_size, True, 0, False)
    data = DataBunch.create(DatasetWrapper(dataset._train_loader), DatasetWrapper(dataset._train_loader), bs=1)
    model = Net(params, dataset.num_classes) 
    learner = Learner(data, model, opt_func=torch.optim.Adam, loss_func=CustomLoss())
    learner.fit(1)

if __name__ == '__main__':
    params = OmegaConf.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "main.yaml"))
    main(params)
