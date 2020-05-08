from utils import load_torch_points3d
load_torch_points3d()
import os
import sys
import torch
from torch import nn
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch_points3d.core.data_transform import AddOnes
from torch_points3d.applications.pointnet2 import PointNet2
from torch_points3d.applications.kpconv import KPConv
from torch_points3d.applications.rsconv import RSConv
from torch_points3d.datasets.classification.modelnet import ModelNetDataset
from torch_points3d.core.common_modules.base_modules import Seq
from torch_points3d.core.common_modules.dense_modules import Conv1D
import pytorch_lightning as pl

class Model(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self._params = params
        self._build_backbone()
        self._model["classifier"] = Seq()
        self._model["classifier"].append(Conv1D(self._model_opt.output_nc, self._params.data.number, activation=None, bias=True, bn=False))

    def _build_backbone(self):
        self._model = nn.ModuleDict()
        self._model_name = sys.argv[1] if len(sys.argv) == 2 else "pointnet2"
        assert self._model_name in ["pointnet2", "kpconv", "rsconv"], "model_name should within ['pointnet2', 'pointnet', 'kpconv', 'rsconv']"
        self._model_opt = getattr(self._params.models, self._model_name)
        model_builder = globals().copy()[self._model_opt.class_name]
        self._model["backbone"] = model_builder(architecture="encoder", 
                                                input_nc=self._model_opt.input_nc, 
                                                in_feat=self._model_opt.in_feat,
                                                num_layers=self._model_opt.num_layers,
                                                output_nc=self._model_opt.output_nc,
                                                multiscale=True)

    def forward(self, data):
        if self._model_name == "kpconv": # KPConv needs added one for its x features
            data = AddOnes()(data)
            data.x = data.ones
        data_out = self._model["backbone"](data)
        return F.log_softmax(self._model["classifier"](data_out.x).squeeze(), dim=-1)

    def training_step(self, data, *args):
        y_hat = self(data)
        return {'loss': F.nll_loss(y_hat, data.y.squeeze()), "acc": self.compute_acc(y_hat, data.y.squeeze())}

    def test_step(self, data, *args):
        y_hat = self(data)
        return {'test_loss': F.nll_loss(y_hat, data.y.squeeze()), "test_acc": self.compute_acc(y_hat, data.y.squeeze())}

    def test_epoch_end(self, outputs):
        return {'test_loss': torch.stack([x['test_loss'] for x in outputs]).mean(),
                'test_acc': torch.stack([x['test_acc'] for x in outputs]).mean()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):
        self._dataset = ModelNetDataset(self._params.data)
        self._dataset.create_dataloaders(self._model_opt, self._params.training.batch_size, True, self._params.training.num_workers, False)

    def train_dataloader(self):
        return self._dataset._train_loader

    def test_dataloader(self):
        return self._dataset._test_loader

    @staticmethod
    def compute_acc(y_hat, y):
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        return torch.tensor(acc)

def main(params):

    model = Model(params=params)
    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()), progress_bar_refresh_rate=10, max_epochs=10)
    trainer.fit(model)   

if __name__ == '__main__':
    params = OmegaConf.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "main.yaml"))
    main(params)
