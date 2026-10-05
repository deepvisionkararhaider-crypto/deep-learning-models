"""Twenty compact PyTorch architectures with real-dataset training pipelines."""
import torch
from torch import nn
import torch.nn.functional as F
from real_datasets import breast_cancer, digits, har, text_sequences, cora

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
    def __init__(self): super().__init__(); self.back=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.head=nn.Linear(32,14)
    def forward(self,x): return self.head(self.back(x).flatten(1))
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
def _sample(x,y,n=512):
    idx=torch.arange(min(n,len(x))); return x[idx],y[idx]
def _bbox(x):
    out=[]
    for im in x:
        ys,xs=(im[0]>.2).nonzero(as_tuple=True); out.append([xs.min()/7,ys.min()/7,xs.max()/7,ys.max()/7] if len(xs) else [0,0,1,1])
    return torch.tensor(out,dtype=torch.float32)
def train_one(model_id,epochs=3,lr=1e-3):
    torch.manual_seed(7); _,_,cls,_,_=SPEC_BY_ID[model_id]
    if model_id==1:
        x,y,_,_=breast_cancer(); model=cls(); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
        for _ in range(epochs): opt.zero_grad(); loss=F.cross_entropy(model(x),y); loss.backward(); opt.step(); losses.append(float(loss))
        return model,losses,float((model(x).argmax(1)==y).float().mean())
    if model_id in (2,12,13,14):
        x,y,_=digits(); x,y=_sample(x,y); model=cls(); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
        for _ in range(epochs): opt.zero_grad(); loss=F.cross_entropy(model(x),y); loss.backward(); opt.step(); losses.append(float(loss))
        return model,losses,float((model(x).argmax(1)==y).float().mean())
    if model_id in (3,4,5):
        x,y,_=har(); x,y=_sample(x,y); model=cls(); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
        for _ in range(epochs): opt.zero_grad(); loss=F.cross_entropy(model(x),y); loss.backward(); opt.step(); losses.append(float(loss))
        return model,losses,float((model(x).argmax(1)==y).float().mean())
    if model_id==6:
        x,_,_=digits(); x,_=_sample(x,x); model=cls(); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
        for _ in range(epochs): opt.zero_grad(); loss=F.mse_loss(model(x),x); loss.backward(); opt.step(); losses.append(float(loss))
        return model,losses,float(1/(1+losses[-1]))
    if model_id==7:
        x,_,_=digits(); x,_=_sample(x,x); model=cls(); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
        for _ in range(epochs): opt.zero_grad(); recon,mu,lv=model(x); loss=F.binary_cross_entropy(recon,x)+.0001*torch.mean(mu.pow(2)+lv.exp()-lv-1); loss.backward(); opt.step(); losses.append(float(loss))
        return model,losses,float(1/(1+losses[-1]))
    if model_id==8:
        real,_,_=digits(); real,_=_sample(real,real); G=GANGenerator(); D=cls(); og=torch.optim.Adam(G.parameters(),lr=lr); od=torch.optim.Adam(D.parameters(),lr=lr); losses=[]
        for _ in range(epochs):
            z=torch.randn(len(real),32); fake=G(z).detach(); od.zero_grad(); ld=F.binary_cross_entropy_with_logits(D(real),torch.ones(len(real),1))+F.binary_cross_entropy_with_logits(D(fake),torch.zeros(len(real),1)); ld.backward(); od.step(); og.zero_grad(); fake=G(z); lg=F.binary_cross_entropy_with_logits(D(fake),torch.ones(len(real),1)); lg.backward(); og.step(); losses.append(float(lg))
        return D,losses,float(torch.sigmoid(D(real)).mean())
    if model_id in (9,10):
        x,y,_,_,_=text_sequences(); x,y=_sample(x,y); model=cls(); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
        for _ in range(epochs): opt.zero_grad(); loss=F.cross_entropy(model(x),y); loss.backward(); opt.step(); losses.append(float(loss))
        return model,losses,float((model(x).argmax(1)==y).float().mean())
    if model_id in (11,18):
        x,_,_,_,_=text_sequences(); x,_=_sample(x,x); model=cls(); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
        for _ in range(epochs): opt.zero_grad(); out=model(x[:,:-1]); loss=F.cross_entropy(out.reshape(-1,1024),x[:,1:].reshape(-1)); loss.backward(); opt.step(); losses.append(float(loss))
        return model,losses,float(1/(1+losses[-1]))
    if model_id==15:
        x,_,_=digits(); x,_=_sample(x,x); target=(x>.15).float(); model=cls(); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
        for _ in range(epochs): opt.zero_grad(); loss=F.binary_cross_entropy_with_logits(model(x),target); loss.backward(); opt.step(); losses.append(float(loss))
        return model,losses,float((torch.sigmoid(model(x))>.5).eq(target).float().mean())
    if model_id==16:
        x,y,_=digits(); x,y=_sample(x,y); box=_bbox(x); model=cls(); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
        for _ in range(epochs): opt.zero_grad(); out=model(x); loss=F.mse_loss(torch.sigmoid(out[:,:4]),box)+F.cross_entropy(out[:,4:],y); loss.backward(); opt.step(); losses.append(float(loss))
        return model,losses,float((model(x)[:,4:].argmax(1)==y).float().mean())
    if model_id==17:
        x,y,_=digits(); x,y=_sample(x,y,256); b=torch.roll(x,1,0); same=(y==torch.roll(y,1,0)).float(); model=cls(); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
        for _ in range(epochs):
            opt.zero_grad(); za,zb=model(x,b); dist=(za-zb).pow(2).sum(1).sqrt(); loss=(same*dist.pow(2)+(1-same)*F.relu(1-dist).pow(2)).mean(); loss.backward(); opt.step(); losses.append(float(loss))
        return model,losses,float((same==(dist<.5).float()).float().mean())
    if model_id==19:
        x,_,_,_=breast_cancer(); model=cls(); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
        for _ in range(epochs):
            t=torch.rand(len(x),1); noise=torch.randn_like(x)*t; opt.zero_grad(); loss=F.mse_loss(model(x+noise,t),noise); loss.backward(); opt.step(); losses.append(float(loss))
        return model,losses,float(1/(1+losses[-1]))
    if model_id==20:
        x,y,a=cora(); model=cls(); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
        for _ in range(epochs): opt.zero_grad(); loss=F.cross_entropy(model(x,a),y); loss.backward(); opt.step(); losses.append(float(loss))
        return model,losses,float((model(x,a).argmax(1)==y).float().mean())
    raise ValueError(model_id)
