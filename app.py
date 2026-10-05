from pathlib import Path
import io
import numpy as np
import streamlit as st
import torch
from PIL import Image
from sklearn.datasets import load_breast_cancer
from model_zoo import MODEL_SPECS, SPEC_BY_ID
from real_datasets import DATASET_CATALOG, _tokenize
from pretrained_weights import MODEL_WEIGHTS, MODEL_METRICS, load_state
GITHUB='https://github.com/deepvisionkararhaider-crypto/deep-learning-models'
st.set_page_config(page_title='Deep Learning Models | Pretrained Lab',page_icon='🧠',layout='wide')
torch.set_num_threads(2)
@st.cache_resource(show_spinner=False)
def get_model(model_id):
    _,name,cls,_,_=SPEC_BY_ID[model_id]; model=cls(); model.load_state_dict(load_state(model_id),strict=True); model.eval(); return model
def numeric_vector(text,size):
    vals=[float(v.strip()) for v in text.replace('\n',',').split(',') if v.strip()]
    if len(vals)!=size: raise ValueError(f'Expected exactly {size} numbers; received {len(vals)}.')
    return torch.tensor([vals],dtype=torch.float32)
def image_tensor(upload):
    img=Image.open(upload).convert('L').resize((28,28)); arr=np.asarray(img,dtype=np.float32)/255.0; return torch.from_numpy(arr)[None,None,:,:],img
def sequence_tensor(upload):
    import pandas as pd
    df=pd.read_csv(io.BytesIO(upload.getvalue())); values=df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32).reshape(-1)
    if values.size<96: raise ValueError('CSV needs at least 96 numeric values.')
    return torch.from_numpy(values[:96].reshape(12,8))[None,:,:]
def text_tensor(text):
    ids=_tokenize(text,length=32,vocab=1024); return torch.tensor(ids,dtype=torch.float32).view(1,4,8)/1024.0
def predict(model_id,x):
    with torch.no_grad(): out=get_model(model_id)(x)
    probs=torch.softmax(out,dim=1); cls=int(probs.argmax(1).item()); return cls,float(probs[0,cls].item()),probs[0].cpu().numpy()
def overview():
    st.title('🧠 Deep Learning Models — Pretrained Prediction Lab'); st.caption('20 architectures are trained once on real public datasets and shipped with the app.')
    a,b,c,d=st.columns(4); a.metric('Models','20'); b.metric('Pretrained','20 / 20'); c.metric('Retraining for visitors','Not required'); d.metric('Runtime','CPU')
    st.success('✅ Every visitor uses the committed pretrained weights. Opening the public link does NOT start a new training job.')
    st.divider(); st.subheader('Model → real dataset → prediction task')
    for start in range(0,20,4):
        cols=st.columns(4)
        for col,(n,name,_,_,_) in zip(cols,MODEL_SPECS[start:start+4]):
            ds,_,task=DATASET_CATALOG[n]
            with col: st.markdown(f'### {n:02d} · {name}'); st.write(task); st.caption(ds)
def prediction():
    st.title('🔮 Instant Prediction Studio'); st.info('The model is already trained. Enter/upload data and click Predict — there is no training step.')
    n=st.selectbox('Choose model',range(1,21),format_func=lambda i:f'{i:02d} · {SPEC_BY_ID[i][1]}'); ds,_,task=DATASET_CATALOG[n]; st.markdown(f'**Dataset:** {ds}  \n**Task:** {task}'); st.caption(f'Pretrained build metric: {MODEL_METRICS.get(n,0):.1%}')
    try:
        if n==1:
            st.subheader('Breast-cancer tabular input'); st.caption('First 20 standardized features used by the bundled ANN.'); raw=st.text_area('20 comma-separated standardized features',','.join(['0']*20),height=100)
            if st.button('🔮 Predict',type='primary'):
                cls,conf,probs=predict(n,numeric_vector(raw,20)); st.success(f"Prediction: {'benign' if cls==1 else 'malignant'}"); st.metric('Confidence',f'{conf:.1%}'); st.bar_chart({'malignant':float(probs[0]),'benign':float(probs[1])})
            st.caption('Educational demonstration only — not a medical diagnostic tool.')
        elif n in (2,12,13,14,15,16):
            up=st.file_uploader('Upload an image',type=['png','jpg','jpeg','webp'])
            if up:
                x,img=image_tensor(up); st.image(img,width=180)
                if st.button('🔮 Predict',type='primary'):
                    cls,conf,probs=predict(n,x); st.success(f"Prediction: {'odd digit' if cls else 'even digit'}"); st.metric('Confidence',f'{conf:.1%}'); st.bar_chart({'even':float(probs[0]),'odd':float(probs[1])})
        elif n in (3,4,5):
            up=st.file_uploader('Upload HAR CSV',type=['csv']); st.caption('Provide at least 96 numeric values; the first 96 become a 12×8 sensor window.')
            if up and st.button('🔮 Predict',type='primary'):
                cls,conf,probs=predict(n,sequence_tensor(up)); st.success(f"Prediction: {'walking' if cls==0 else 'non-walking'}"); st.metric('Confidence',f'{conf:.1%}'); st.bar_chart({'walking':float(probs[0]),'non-walking':float(probs[1])})
        elif n in (9,10,11,18):
            text=st.text_area('Text input',height=180,placeholder='Paste a short news-style text...')
            if st.button('🔮 Predict',type='primary') and text.strip():
                cls,conf,probs=predict(n,text_tensor(text)); st.success(f"Prediction: {'topics 0–1' if cls==0 else 'topics 2–3'}"); st.metric('Confidence',f'{conf:.1%}'); st.bar_chart({'topic-group-0':float(probs[0]),'topic-group-1':float(probs[1])})
        elif n==17:
            a=st.text_area('Vector A — 20 comma-separated values',','.join(['0']*20)); b=st.text_area('Vector B — 20 comma-separated values',','.join(['0']*20))
            if st.button('🔮 Predict',type='primary'):
                cls,conf,_=predict(n,torch.cat([numeric_vector(a,20),numeric_vector(b,20)],0)); st.success(f"Prediction: {'same parity class' if cls else 'different parity class'}"); st.metric('Confidence',f'{conf:.1%}')
        elif n==20:
            raw=st.text_area('6×4 graph features (24 comma-separated values)',','.join(['0']*24))
            if st.button('🔮 Predict',type='primary'):
                cls,conf,_=predict(n,numeric_vector(raw,24).reshape(1,6,4)); st.success(f'Prediction: Cora binary node-group {cls}'); st.metric('Confidence',f'{conf:.1%}')
        else:
            raw=st.text_area('20 comma-separated features',','.join(['0']*20))
            if st.button('🔮 Predict',type='primary'):
                cls,conf,_=predict(n,numeric_vector(raw,20)); st.success(f'Prediction class: {cls}'); st.metric('Confidence',f'{conf:.1%}')
    except Exception as e: st.error(str(e))
def training():
    st.title('🎛️ Training / Fine-tuning Lab'); st.warning('Optional only. Prediction Studio never trains. Training here is a session experiment and does not replace the committed pretrained weights.'); st.info('The production inference path is the pretrained model shipped in pretrained_weights.py.')
def catalog():
    st.title('📦 20-Model Catalog')
    for n,name,cls,_,desc in MODEL_SPECS:
        ds,_,task=DATASET_CATALOG[n]
        with st.expander(f'{n:02d} · {name}'): st.write(f'**Dataset:** {ds}'); st.write(f'**Task:** {task}'); st.write(f'**Architecture:** {desc}'); st.caption(f'Build metric: {MODEL_METRICS.get(n,0):.1%}')
with st.sidebar:
    st.markdown('# 🧠 Deep Learning Lab'); page=st.radio('Navigate',['Overview','🔮 Prediction Studio','🎛️ Training / Fine-tuning','📦 20-Model Catalog']); st.divider(); st.link_button('⭐ GitHub',GITHUB,use_container_width=True)
if not MODEL_WEIGHTS: st.error('Pretrained weights are still being built. Refresh after GitHub Actions finishes.')
elif page=='Overview': overview()
elif page=='🔮 Prediction Studio': prediction()
elif page=='🎛️ Training / Fine-tuning': training()
else: catalog()
st.divider(); st.caption('20 pretrained deep-learning architectures · real public datasets · persistent bundled inference weights')
