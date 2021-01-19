#_*_coding:utf-8_*_
'''
@project: ExudingTao
@author: exudingtao
@time: 2021/1/19 4:55 下午
'''

'''
pytorch-lightning 的核心设计哲学是将 深度学习项目中的 研究代码(定义模型) 和 工程代码 (训练模型) 相互分离。
用户只需专注于研究代码(pl.LightningModule)的实现，而工程代码借助训练工具类(pl.Trainer)统一实现。

更详细地说，深度学习项目代码可以分成如下4部分：
    研究代码 (Research code)，用户继承LightningModule实现。
    工程代码 (Engineering code)，用户无需关注通过调用Trainer实现。
    非必要代码 （Non-essential research code，logging, etc...），用户通过调用Callbacks实现。
    数据 (Data)，用户通过torch.utils.data.DataLoader实现。
'''

import pytorch_lightning as pl
import torch
from torch import nn
import torchvision
from torchvision import transforms
import datetime


transform = transforms.Compose([transforms.ToTensor()])
ds_train = torchvision.datasets.MNIST(root="./minist/", train=True, download=True, transform=transform)
ds_valid = torchvision.datasets.MNIST(root="./minist/", train=False, download=True, transform=transform)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=128, shuffle=False, num_workers=4)

print(len(ds_train))
print(len(ds_valid))


#定义模型
class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    # 定义loss,以及可选的各种metrics
    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = nn.CrossEntropyLoss()(prediction, y)
        return loss

    # 定义optimizer,以及可选的lr_scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return {"test_loss": loss}

##训练模型
pl.seed_everything(1234)
model = Model()


ckpt_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    save_top_k=1,
    mode='min'
)

# gpus=0 则使用cpu训练，gpus=1则使用1个gpu训练，gpus=2则使用2个gpu训练，gpus=-1则使用所有gpu训练，
# gpus=[0,1]则指定使用0号和1号gpu训练， gpus="0,1,2,3"则使用0,1,2,3号gpu训练
# tpus=1 则使用1个tpu训练

trainer = pl.Trainer(max_epochs=5, gpus=0, callbacks=[ckpt_callback])

#断点续训
#trainer = pl.Trainer(resume_from_checkpoint='./lightning_logs/version_31/checkpoints/epoch=02-val_loss=0.05.ckpt')

trainer.fit(model, dl_train, dl_valid)

##评估模型
result = trainer.test(model, test_dataloaders=dl_valid)
print(result)

##使用模型
data, label = next(iter(dl_valid))
model.eval()
prediction = model(data)
print(prediction)

## 保存模型
print(trainer.checkpoint_callback.best_model_path)
print(trainer.checkpoint_callback.best_model_score)
model_clone = Model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
trainer_clone = pl.Trainer(max_epochs=3)
result = trainer_clone.test(model_clone, dl_valid)
print(result)



