"""Real public benchmark dataset loaders used by the 20-model Streamlit lab."""
from functools import lru_cache
from pathlib import Path
import hashlib,pickle,urllib.request,zipfile
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer,load_digits,fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
CACHE=Path.home()/'.deep_learning_models'; CACHE.mkdir(parents=True,exist_ok=True)
@lru_cache(maxsize=1)
def breast_cancer():
    d=load_breast_cancer(); x=StandardScaler().fit_transform(d.data).astype('float32'); y=d.target.astype('int64'); return torch.from_numpy(x),torch.from_numpy(y),list(d.feature_names),list(d.target_names)
@lru_cache(maxsize=1)
def digits():
    d=load_digits(); x=(d.images/16.0).astype('float32'); y=d.target.astype('int64'); return torch.from_numpy(x[:,None,:,:]),torch.from_numpy(y),list(map(str,range(10)))
HAR_URL='https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
@lru_cache(maxsize=1)
def har():
    root=CACHE/'UCI_HAR_Dataset'
    if not root.exists():
        z=CACHE/'har.zip'
        if not z.exists(): urllib.request.urlretrieve(HAR_URL,z)
        with zipfile.ZipFile(z) as f: f.extractall(CACHE)
    base=root/'train'; x=np.loadtxt(base/'X_train.txt',dtype='float32'); y=np.loadtxt(base/'y_train.txt',dtype='int64')-1; return torch.from_numpy(x[:,:558].reshape(-1,93,6)),torch.from_numpy(y),['walking','upstairs','downstairs','sitting','standing','laying']
TEXT_CATEGORIES=['comp.graphics','sci.space','rec.sport.baseball','talk.politics.misc']
@lru_cache(maxsize=1)
def newsgroups():
    tr=fetch_20newsgroups(subset='train',categories=TEXT_CATEGORIES,remove=('headers','footers','quotes'),random_state=42); te=fetch_20newsgroups(subset='test',categories=TEXT_CATEGORIES,remove=('headers','footers','quotes'),random_state=42); return tr.data,tr.target.astype('int64'),te.data,te.target.astype('int64'),TEXT_CATEGORIES
def _tokenize(text,length=32,vocab=1024):
    words=text.lower().split()[:length]; ids=[1+int(hashlib.md5(w.encode('utf-8','ignore')).hexdigest(),16)%(vocab-2) for w in words]; return ids+[0]*(length-len(ids))
@lru_cache(maxsize=1)
def text_sequences():
    tr,y,te,yt,cats=newsgroups(); X=torch.tensor([_tokenize(t) for t in tr[:4000]],dtype=torch.long); Y=torch.tensor(y[:4000]); Xt=torch.tensor([_tokenize(t) for t in te[:1000]],dtype=torch.long); Yt=torch.tensor(yt[:1000]); return X,Y,Xt,Yt,cats
@lru_cache(maxsize=1)
def cora():
    urls={k:f'https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.cora.{k}' for k in ['x','y','tx','ty','allx','ally','graph']}; paths={}
    for k,u in urls.items():
        p=CACHE/f'cora_{k}'
        if not p.exists(): urllib.request.urlretrieve(u,p)
        paths[k]=p
    from scipy import sparse
    def read(p):
        with open(p,'rb') as f: return pickle.load(f,encoding='latin1')
    allx,ally,tx,ty,graph=read(paths['allx']),read(paths['ally']),read(paths['tx']),read(paths['ty']),read(paths['graph']); x=sparse.vstack((allx,tx)).toarray().astype('float32'); y=np.vstack((ally,ty)).argmax(1).astype('int64'); n=len(x); A=np.eye(n,dtype='float32')
    for i,ns in graph.items():
        for j in ns: A[i,j]=1; A[j,i]=1
    A/=np.maximum(A.sum(1,keepdims=True),1); return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(A)
DATASET_CATALOG={1:('Breast Cancer Wisconsin (Diagnostic)','UCI / scikit-learn','binary malignant vs benign classification'),2:('Optical Recognition of Handwritten Digits','UCI / scikit-learn','even vs odd digit image classification'),3:('Human Activity Recognition Using Smartphones','UCI','walking vs non-walking sensor-sequence classification'),4:('Human Activity Recognition Using Smartphones','UCI','walking vs non-walking sensor-sequence classification'),5:('Human Activity Recognition Using Smartphones','UCI','walking vs non-walking sensor-sequence classification'),6:('Optical Recognition of Handwritten Digits','UCI / scikit-learn','even vs odd representation classifier'),7:('Optical Recognition of Handwritten Digits','UCI / scikit-learn','even vs odd variational classifier'),8:('Optical Recognition of Handwritten Digits','UCI / scikit-learn','digit discriminator / even-vs-odd classifier'),9:('20 Newsgroups','scikit-learn','topics 0–1 vs topics 2–3 text classification'),10:('20 Newsgroups','scikit-learn','bidirectional topics 0–1 vs 2–3 classification'),11:('20 Newsgroups','scikit-learn','causal text topic-group classification'),12:('Optical Recognition of Handwritten Digits','UCI / scikit-learn','even vs odd residual image classification'),13:('Optical Recognition of Handwritten Digits','UCI / scikit-learn','even vs odd dense image classification'),14:('Optical Recognition of Handwritten Digits','UCI / scikit-learn','even vs odd vision-transformer classification'),15:('Optical Recognition of Handwritten Digits','UCI / scikit-learn','digit foreground / parity classification'),16:('Optical Recognition of Handwritten Digits','UCI / scikit-learn','digit localization backbone / parity classification'),17:('Optical Recognition of Handwritten Digits','UCI / scikit-learn','same-vs-different parity metric learning'),18:('20 Newsgroups','scikit-learn','text sequence topic-group reconstruction classifier'),19:('Breast Cancer Wisconsin (Diagnostic)','UCI / scikit-learn','noise-conditioned tabular malignant-vs-benign classifier'),20:('Cora','Planetoid citation benchmark','binary node-group classification from real citation features')}
