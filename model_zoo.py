"""Lightweight, CPU-friendly trainable deep-learning model zoo for the Streamlit lab."""
import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,x): return self.net(x)
class CNN(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.fc=nn.Linear(32,2)
    def forward(self,x): return self.fc(self.net(x).flatten(1))
class RNN(nn.Module):
    def __init__(self): super().__init__(); self.rnn=nn.RNN(8,32,batch_first=True); self.fc=nn.Linear(32,2)
    def forward(self,x): y,_=self.rnn(x); return self.fc(y[:,-1])
class LSTM(nn.Module):
    def __init__(self): super().__init__(); self.rnn=nn.LSTM(8,32,batch_first=True); self.fc=nn.Linear(32,2)
    def forward(self,x): y,_=self.rnn(x); return self.fc(y[:,-1])
class GRU(nn.Module):
    def __init__(self): super().__init__(); self.rnn=nn.GRU(8,32,batch_first=True); self.fc=nn.Linear(32,2)
    def forward(self,x): y,_=self.rnn(x); return self.fc(y[:,-1])
class Autoencoder(nn.Module):
    def __init__(self): super().__init__(); self.enc=nn.Sequential(nn.Linear(20,12),nn.ReLU(),nn.Linear(12,4)); self.fc=nn.Linear(4,2)
    def forward(self,x): return self.fc(self.enc(x))
class VAE(nn.Module):
    def __init__(self): super().__init__(); self.h=nn.Linear(20,16); self.mu=nn.Linear(16,4); self.logvar=nn.Linear(16,4); self.fc=nn.Linear(4,2)
    def forward(self,x):
        h=F.relu(self.h(x)); mu,lv=self.mu(h),self.logvar(h); z=mu+torch.randn_like(mu)*torch.exp(.5*lv); return self.fc(z)
class GANDiscriminator(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(20,64),nn.LeakyReLU(.2),nn.Linear(64,32),nn.LeakyReLU(.2),nn.Linear(32,2))
    def forward(self,x): return self.net(x)
class Transformer(nn.Module):
    def __init__(self): super().__init__(); self.proj=nn.Linear(8,32); layer=nn.TransformerEncoderLayer(32,4,64,batch_first=True); self.enc=nn.TransformerEncoder(layer,2); self.fc=nn.Linear(32,2)
    def forward(self,x): return self.fc(self.enc(self.proj(x)).mean(1))
class TinyBERT(nn.Module):
    def __init__(self): super().__init__(); self.emb=nn.Linear(8,32); layer=nn.TransformerEncoderLayer(32,4,64,batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(layer,2); self.fc=nn.Linear(32,2)
    def forward(self,x): return self.fc(self.enc(self.emb(x)).mean(1))
class TinyGPT(nn.Module):
    def __init__(self): super().__init__(); self.emb=nn.Linear(8,32); layer=nn.TransformerEncoderLayer(32,4,64,batch_first=True); self.tr=nn.TransformerEncoder(layer,2); self.fc=nn.Linear(32,2)
    def forward(self,x):
        n=x.size(1); mask=torch.triu(torch.ones(n,n,device=x.device),1).bool(); return self.fc(self.tr(self.emb(x),mask=mask)[:,-1])
class ResidualBlock(nn.Module):
    def __init__(self,c): super().__init__(); self.a=nn.Conv2d(c,c,3,padding=1); self.b=nn.Conv2d(c,c,3,padding=1)
    def forward(self,x): return F.relu(self.b(F.relu(self.a(x)))+x)
class ResNet(nn.Module):
    def __init__(self): super().__init__(); self.stem=nn.Conv2d(1,16,3,padding=1); self.block=ResidualBlock(16); self.fc=nn.Linear(16,2)
    def forward(self,x): return self.fc(F.adaptive_avg_pool2d(self.block(F.relu(self.stem(x))),1).flatten(1))
class DenseNet(nn.Module):
    def __init__(self): super().__init__(); self.a=nn.Conv2d(1,12,3,padding=1); self.b=nn.Conv2d(13,12,3,padding=1); self.c=nn.Conv2d(25,12,3,padding=1); self.fc=nn.Linear(36,2)
    def forward(self,x): a=F.relu(self.a(x)); b=F.relu(self.b(torch.cat([x,a],1))); c=F.relu(self.c(torch.cat([x,a,b],1))); return self.fc(F.adaptive_avg_pool2d(torch.cat([a,b,c],1),1).flatten(1))
class ViT(nn.Module):
    def __init__(self): super().__init__(); self.patch=nn.Conv2d(1,32,7,7); layer=nn.TransformerEncoderLayer(32,4,64,batch_first=True); self.tr=nn.TransformerEncoder(layer,2); self.fc=nn.Linear(32,2)
    def forward(self,x): return self.fc(self.tr(self.patch(x).flatten(2).transpose(1,2)).mean(1))
class UNet(nn.Module):
    def __init__(self): super().__init__(); self.e1=nn.Conv2d(1,8,3,padding=1); self.e2=nn.Conv2d(8,16,3,padding=1); self.out=nn.Conv2d(16,2,1)
    def forward(self,x): z=F.max_pool2d(F.relu(self.e1(x)),2); z=F.interpolate(F.relu(self.e2(z)),size=x.shape[-2:]); return self.out(z).mean((2,3))
class YOLOLite(nn.Module):
    def __init__(self): super().__init__(); self.back=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.head=nn.Linear(32,6)
    def forward(self,x): return self.head(self.back(x).flatten(1))[:,:2]
class Siamese(nn.Module):
    def __init__(self): super().__init__(); self.enc=nn.Sequential(nn.Linear(20,32),nn.ReLU(),nn.Linear(32,8)); self.fc=nn.Linear(8,2)
    def forward(self,x):
        half=x.shape[0]//2; d=torch.abs(self.enc(x[:half])-self.enc(x[half:2*half])); return self.fc(d)
class Seq2Seq(nn.Module):
    def __init__(self): super().__init__(); self.enc=nn.GRU(8,32,batch_first=True); self.dec=nn.GRU(8,32,batch_first=True); self.fc=nn.Linear(32,2)
    def forward(self,x): h=self.enc(x)[1]; y,_=self.dec(x,h); return self.fc(y[:,-1])
class DiffusionMLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(21,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,20)); self.fc=nn.Linear(20,2)
    def forward(self,x):
        t=torch.rand(x.size(0),1,device=x.device); return self.fc(self.net(torch.cat([x,t],1)))
class GNN(nn.Module):
    def __init__(self): super().__init__(); self.w1=nn.Linear(4,16); self.w2=nn.Linear(16,16); self.fc=nn.Linear(16,2)
    def forward(self,x):
        n=x.size(1); a=torch.eye(n,device=x.device)
        for i in range(n): a[i,(i-1)%n]=1; a[i,(i+1)%n]=1
        h=F.relu(self.w1(torch.matmul(a,x))); h=F.relu(self.w2(torch.matmul(a,h))); return self.fc(h.mean(1))

MODEL_SPECS=[
 (1,'ANN / MLP',MLP,'tabular','Multi-layer perceptron'),(2,'CNN',CNN,'image','Convolutional neural network'),(3,'RNN',RNN,'sequence','Vanilla recurrent network'),(4,'LSTM',LSTM,'sequence','Long short-term memory'),(5,'GRU',GRU,'sequence','Gated recurrent unit'),(6,'Autoencoder',Autoencoder,'tabular','Compact encoder classifier'),(7,'VAE',VAE,'tabular','Variational latent representation'),(8,'GAN',GANDiscriminator,'tabular','Adversarial discriminator'),(9,'Transformer',Transformer,'sequence','Self-attention encoder'),(10,'BERT-style',TinyBERT,'sequence','Bidirectional transformer encoder'),(11,'GPT-style',TinyGPT,'sequence','Causal transformer decoder'),(12,'ResNet',ResNet,'image','Residual CNN'),(13,'DenseNet',DenseNet,'image','Dense connectivity CNN'),(14,'ViT',ViT,'image','Vision Transformer'),(15,'U-Net',UNet,'image','Encoder-decoder segmentation backbone'),(16,'YOLO-style',YOLOLite,'image','Single-stage detection head'),(17,'Siamese Network',Siamese,'pairs','Metric-learning twin encoder'),(18,'Seq2Seq',Seq2Seq,'sequence','Encoder-decoder recurrent model'),(19,'Diffusion MLP',DiffusionMLP,'tabular','Noise-conditioned denoiser'),(20,'GNN',GNN,'graph','Message-passing graph network')]
SPEC_BY_ID={x[0]:x for x in MODEL_SPECS}

def make_batch(kind,n=256):
    g=torch.Generator().manual_seed(42)
    if kind=='tabular':
        x=torch.randn(n,20,generator=g); return x,(x[:,:5].sum(1)>0).long()
    if kind=='image':
        x=torch.randn(n,1,28,28,generator=g); return x,(x.mean((1,2,3))>0).long()
    if kind=='sequence':
        x=torch.randn(n,12,8,generator=g); return x,(x[:,:,:2].mean((1,2))>0).long()
    if kind=='pairs':
        x=torch.randn(n,20,generator=g); x2=x+0.25*torch.randn(n,20,generator=g); return torch.cat([x,x2],0),(x[:,:5].mean(1)>0).long()
    if kind=='graph':
        x=torch.randn(n,6,4,generator=g); return x,(x.mean((1,2))>0).long()
    raise ValueError(kind)

def train_one(model_id,epochs=8,lr=1e-3):
    _,_,cls,kind,_=SPEC_BY_ID[model_id]; torch.manual_seed(7); model=cls(); x,y=make_batch(kind); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
    for _ in range(epochs):
        opt.zero_grad(); logits=model(x); target=y[:logits.size(0)] if model_id==17 else y; loss=F.cross_entropy(logits,target); loss.backward(); opt.step(); losses.append(float(loss.detach()))
    with torch.no_grad(): logits=model(x); target=y[:logits.size(0)] if model_id==17 else y; acc=float((logits.argmax(1)==target).float().mean())
    return model,losses,acc
