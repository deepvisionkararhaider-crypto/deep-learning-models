"""Self-contained PyTorch architecture definitions (no dataset loaders).

Generated from model_zoo.py so inference apps stay lightweight and offline.
"""
"""Twenty compact PyTorch architectures with real-dataset training pipelines."""
import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(30,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,x): return self.net(x)
class CNN(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.fc=nn.Linear(32,10)
    def forward(self,x): return self.fc(self.net(x).flatten(1))
class RNN(nn.Module):
    def __init__(self): super().__init__(); self.rnn=nn.RNN(6,32,batch_first=True); self.fc=nn.Linear(32,6)
    def forward(self,x): return self.fc(self.rnn(x)[0][:,-1])
class LSTM(nn.Module):
    def __init__(self): super().__init__(); self.rnn=nn.LSTM(6,32,batch_first=True); self.fc=nn.Linear(32,6)
    def forward(self,x): return self.fc(self.rnn(x)[0][:,-1])
class GRU(nn.Module):
    def __init__(self): super().__init__(); self.rnn=nn.GRU(6,32,batch_first=True); self.fc=nn.Linear(32,6)
    def forward(self,x): return self.fc(self.rnn(x)[0][:,-1])
class Autoencoder(nn.Module):
    def __init__(self): super().__init__(); self.enc=nn.Sequential(nn.Flatten(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,12)); self.dec=nn.Sequential(nn.Linear(12,32),nn.ReLU(),nn.Linear(32,64),nn.Sigmoid())
    def forward(self,x): return self.dec(self.enc(x)).view(-1,1,8,8)
class VAE(nn.Module):
    def __init__(self): super().__init__(); self.h=nn.Linear(64,32); self.mu=nn.Linear(32,12); self.lv=nn.Linear(32,12); self.dec=nn.Sequential(nn.Linear(12,32),nn.ReLU(),nn.Linear(32,64),nn.Sigmoid())
    def forward(self,x):
        h=F.relu(self.h(x.flatten(1))); mu,lv=self.mu(h),self.lv(h); z=mu+torch.randn_like(mu)*torch.exp(.5*lv); return self.dec(z).view(-1,1,8,8),mu,lv
class GANGenerator(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Linear(64,64),nn.Sigmoid())
    def forward(self,z): return self.net(z).view(-1,1,8,8)
class GANDiscriminator(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(64,64),nn.LeakyReLU(.2),nn.Linear(64,1))
    def forward(self,x): return self.net(x)
class Transformer(nn.Module):
    def __init__(self): super().__init__(); self.emb=nn.Embedding(1024,32); layer=nn.TransformerEncoderLayer(32,4,64,batch_first=True); self.enc=nn.TransformerEncoder(layer,2); self.fc=nn.Linear(32,4)
    def forward(self,x): return self.fc(self.enc(self.emb(x)).mean(1))
class TinyBERT(Transformer): pass
class TinyGPT(nn.Module):
    def __init__(self): super().__init__(); self.emb=nn.Embedding(1024,32); layer=nn.TransformerEncoderLayer(32,4,64,batch_first=True); self.tr=nn.TransformerEncoder(layer,2); self.fc=nn.Linear(32,1024)
    def forward(self,x):
        n=x.size(1); mask=torch.triu(torch.ones(n,n,device=x.device),1).bool(); return self.fc(self.tr(self.emb(x),mask=mask))
class Seq2Seq(TinyGPT): pass
class ResNet(nn.Module):
    def __init__(self): super().__init__(); self.stem=nn.Conv2d(1,16,3,padding=1); self.a=nn.Conv2d(16,16,3,padding=1); self.b=nn.Conv2d(16,16,3,padding=1); self.fc=nn.Linear(16,10)
    def forward(self,x): z=F.relu(self.stem(x)); z=F.relu(self.b(F.relu(self.a(z)))+z); return self.fc(F.adaptive_avg_pool2d(z,1).flatten(1))
class DenseNet(nn.Module):
    def __init__(self): super().__init__(); self.a=nn.Conv2d(1,12,3,padding=1); self.b=nn.Conv2d(13,12,3,padding=1); self.c=nn.Conv2d(25,12,3,padding=1); self.fc=nn.Linear(36,10)
    def forward(self,x): a=F.relu(self.a(x)); b=F.relu(self.b(torch.cat([x,a],1))); c=F.relu(self.c(torch.cat([x,a,b],1))); return self.fc(F.adaptive_avg_pool2d(torch.cat([a,b,c],1),1).flatten(1))
class ViT(nn.Module):
    def __init__(self): super().__init__(); self.patch=nn.Conv2d(1,32,2,2); layer=nn.TransformerEncoderLayer(32,4,64,batch_first=True); self.tr=nn.TransformerEncoder(layer,2); self.fc=nn.Linear(32,10)
    def forward(self,x): return self.fc(self.tr(self.patch(x).flatten(2).transpose(1,2)).mean(1))
class UNet(nn.Module):
    def __init__(self): super().__init__(); self.e1=nn.Conv2d(1,8,3,padding=1); self.e2=nn.Conv2d(8,16,3,padding=1); self.out=nn.Conv2d(16,1,1)
    def forward(self,x): z=F.max_pool2d(F.relu(self.e1(x)),2); return self.out(F.interpolate(F.relu(self.e2(z)),size=x.shape[-2:]))
class YOLOLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.back=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                nn.Conv2d(16,32,3,padding=1),nn.ReLU(),
                                nn.Conv2d(32,32,3,padding=1),nn.ReLU(),
                                nn.AdaptiveAvgPool2d((8,8)))
        self.head=nn.Conv2d(32,8,1)          # per-cell [obj,x,y,w,h,c0,c1,c2]
    def forward(self,x): return self.head(self.back(x)).permute(0,2,3,1)
class Siamese(nn.Module):
    def __init__(self): super().__init__(); self.enc=nn.Sequential(nn.Flatten(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,16))
    def forward(self,a,b): return self.enc(a),self.enc(b)
class DiffusionMLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(31,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,30))
    def forward(self,x,t): return self.net(torch.cat([x,t],1))
class GNN(nn.Module):
    def __init__(self): super().__init__(); self.w1=nn.Linear(1433,64); self.w2=nn.Linear(64,64); self.fc=nn.Linear(64,7)
    def forward(self,x,a): h=F.relu(self.w1(a@x)); return self.fc(F.relu(self.w2(a@h)))
MODEL_SPECS=[(1,'ANN / MLP',MLP,'tabular','Breast Cancer Wisconsin: malignant vs benign'),(2,'CNN',CNN,'image','UCI handwritten digits: 10-class image classification'),(3,'RNN',RNN,'sequence','UCI smartphone HAR: 6-class activity recognition'),(4,'LSTM',LSTM,'sequence','UCI smartphone HAR: 6-class activity recognition'),(5,'GRU',GRU,'sequence','UCI smartphone HAR: 6-class activity recognition'),(6,'Autoencoder',Autoencoder,'image','UCI handwritten digits: image reconstruction'),(7,'VAE',VAE,'image','UCI handwritten digits: variational representation'),(8,'GAN',GANDiscriminator,'image','UCI handwritten digits: adversarial discriminator'),(9,'Transformer',Transformer,'text','20 Newsgroups: 4-topic text classification'),(10,'BERT-style',TinyBERT,'text','20 Newsgroups: bidirectional topic classification'),(11,'GPT-style',TinyGPT,'text','20 Newsgroups: causal next-token language modeling'),(12,'ResNet',ResNet,'image','UCI handwritten digits: residual classification'),(13,'DenseNet',DenseNet,'image','UCI handwritten digits: dense classification'),(14,'ViT',ViT,'image','UCI handwritten digits: vision classification'),(15,'U-Net',UNet,'segmentation','UCI handwritten digits: foreground segmentation'),(16,'YOLO-style',YOLOLite,'detection','UCI handwritten digits: localization + class'),(17,'Siamese Network',Siamese,'pairs','UCI handwritten digits: same/different metric learning'),(18,'Seq2Seq',Seq2Seq,'text','20 Newsgroups: noisy text reconstruction'),(19,'Diffusion MLP',DiffusionMLP,'generative','Breast Cancer Wisconsin: tabular denoising'),(20,'GNN',GNN,'graph','Cora: citation-node classification')]
SPEC_BY_ID={x[0]:x for x in MODEL_SPECS}
