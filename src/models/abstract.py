import pytorch_lightning as pl

class AbstractModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass
    def training_step(self, batch, batch_idx):
        pass
    def validation_step(self, batch, batch_idx):
        pass
    def test_step(self, batch, batch_idx):
        pass
    def configure_optimizers(self):
        pass
    def on_train_epoch_end(self):
        pass
    def on_validation_epoch_end(self):
        pass
    def on_test_epoch_end(self):
        pass
    def on_train_epoch_start(self):
        pass
    def on_validation_epoch_start(self):
        pass
    def on_test_epoch_start(self):
        pass
    def encode(self, x):
        pass
    def decode(self, z):
        pass
