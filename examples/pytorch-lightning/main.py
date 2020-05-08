from utils import load_torch_points3d
load_torch_points3d()
import os
import torch
from torch import nn
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch_points3d.datasets.classification.modelnet import ModelNetDataset
from torch_points3d.core.common_modules.base_modules import Seq
from torch_points3d.core.common_modules.dense_modules import Conv1D
from torch_points3d.applications.pointnet2 import PointNet2
import pytorch_lightning as pl
from pytorch_lightning import Trainer


class Model(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        
        self._model_opt = params.model
        self._dataset_opt = params.data
        self._training_opt = params.training
        self._input_nc = eval(self._model_opt.input_nc)
        self._output_nc = eval(self._model_opt.output_nc)
        self._model = nn.ModuleDict()
        self._model["encoder"] = PointNet2(input_nc=self._input_nc, 
                                  output_nc=self._output_nc,
                                  in_feat=self._model_opt.in_feat, 
                                  multiscale=self._model_opt.multiscale, 
                                  num_layers=self._model_opt.num_layers, architecture="encoder")
        
        self._model["classifier"] = Seq()
        self._model["classifier"].append(Conv1D(self._output_nc, self._dataset_opt.number, activation=None, bias=True, bn=False))

    def forward(self, data):
        data_out = self._model["encoder"](data)
        return self._model["classifier"](data_out.x)

    def training_step(self, data, *args):
        y_hat = self(data)
        loss = F.nll_loss(y_hat, data.y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, data, *args):
        y_hat = self(data)
        return {'test_loss': F.nll_loss(y_hat, data.y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):
        self._dataset = ModelNetDataset(self._dataset_opt)
        self._dataset.create_dataloaders(
            self._model_opt,
            self._training_opt.batch_size,
            True,
            self._training_opt.num_workers,
            False,
        )

    def train_dataloader(self):
        return self._dataset._train_loader

    def test_dataloader(self):
        return self._dataset._test_loader

def main(params):

    model = Model(params=params)

    # most basic trainer, uses good defaults
    trainer = Trainer(gpus=0, progress_bar_refresh_rate=10, max_epochs=10)
    trainer.fit(model)   

if __name__ == '__main__':

    params = """
    data:
        dataroot: data
        num_points: 1024
        number: 10
        use_normal: False
    model:
        conv_type: "dense"
        input_nc: 3 * (1 * ${data.use_normal})
        in_feat: 16
        output_nc: 16 * ${model.in_feat}
        num_layers: 3
        multiscale: True
    training:
        batch_size: 8
        num_workers: 4
    """

    params = OmegaConf.create(params)

    main(params)
