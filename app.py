from pathlib import Path
import sys
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model_zoo import MODEL_SPECS, SPEC_BY_ID, train_one

st.set_page_config(page_title='Deep Learning Models | Interactive Lab', page_icon='🧠', layout='wide')
GITHUB = 'https://github.com/deepvisionkararhaider-crypto/deep-learning-models'

@st.cache_data
def cached_train(model_id, epochs, lr):
    _, losses, acc = train_one(model_id, epochs=epochs, lr=lr)
    return losses, acc

def render_overview():
    st.title('🧠 Deep Learning Models — Interactive Lab')
    st.caption('20 trainable deep-learning architectures, organized as one professional GitHub repository.')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Trainable models','20'); c2.metric('Model families','20'); c3.metric('Core tasks','5+'); c4.metric('Framework','PyTorch + Streamlit')
    st.divider()
    st.subheader('What is included?')
    for start in range(0, len(MODEL_SPECS), 4):
        cols=st.columns(4)
        for col,(n,name,_,kind,desc) in zip(cols, MODEL_SPECS[start:start+4]):
            with col:
                st.markdown(f'### {n:02d} · {name}')
                st.caption(kind.title())
                st.write(desc)
                st.code(f'{n:02d}_{name.lower().replace(" / ","_").replace(" ","_").replace("-", "_")}', language='text')
    st.info('The numbered folders contain the full educational/research implementations. The Interactive Training Lab uses lightweight CPU-friendly reference implementations so every architecture can be trained directly in the browser.')

def render_catalog():
    st.title('📦 20-Model Catalog')
    folders={1:'ann',2:'cnn',3:'rnn',4:'lstm',5:'gru',6:'autoencoder',7:'vae',8:'gan',9:'transformer',10:'bert',11:'gpt',12:'resnet',13:'densenet',14:'vit',15:'unet',16:'yolo',17:'siamese',18:'seq2seq',19:'diffusion',20:'gnn'}
    for n,name,cls,kind,desc in MODEL_SPECS:
        with st.expander(f'{n:02d} · {name} — {kind.title()}'):
            st.write(desc)
            st.write(f'**Python class:** `{cls.__name__}`')
            st.write(f'**Repository folder:** `{n:02d}_{folders[n]}/`')
            st.link_button('Open source', f'{GITHUB}/tree/main/{n:02d}_{folders[n]}', use_container_width=True)

def render_training():
    st.title('🎛️ Interactive Training Lab')
    st.write('Select any of the 20 architectures, choose the training settings, and train it live on a deterministic synthetic dataset. This demonstrates real forward passes, backpropagation, optimizer steps, loss curves, and accuracy.')
    names={n:f'{n:02d} · {name}' for n,name,_,_,_ in MODEL_SPECS}
    selected=st.selectbox('Model', list(names.keys()), format_func=lambda n:names[n])
    _,name,_,kind,desc=SPEC_BY_ID[selected]
    st.caption(desc)
    a,b=st.columns(2)
    epochs=a.slider('Epochs',1,30,8)
    lr=b.select_slider('Learning rate',[0.0001,0.0003,0.001,0.003,0.01],value=0.001,format_func=lambda x:f'{x:g}')
    if st.button('🚀 Train selected model', type='primary', use_container_width=True):
        with st.spinner(f'Training {name}...'):
            losses,acc=cached_train(selected,epochs,lr)
        st.success(f'{name} trained successfully.')
        m1,m2=st.columns(2); m1.metric('Final loss',f'{losses[-1]:.4f}'); m2.metric('Training accuracy',f'{acc:.1%}')
        st.line_chart({'loss':losses})
        st.caption('Training uses real PyTorch autograd and Adam optimization. The dataset is generated locally in the app and is not a benchmark claim.')

def render_playground():
    st.title('🖼️ Image Playground')
    uploaded=st.file_uploader('Upload PNG/JPG/JPEG',type=['png','jpg','jpeg'])
    if not uploaded:
        st.info('Upload an image to inspect model-ready preprocessing.')
        return
    image=Image.open(uploaded).convert('RGB'); size=st.slider('Resize',64,512,224,32); processed=ImageOps.fit(image,(size,size))
    a,b=st.columns(2); a.image(image,caption='Original',use_container_width=True); b.image(processed,caption=f'{size}×{size}',use_container_width=True)
    arr=np.asarray(processed,dtype=np.float32)/255.0
    st.json({'shape':list(arr.shape),'min':float(arr.min()),'max':float(arr.max()),'mean':float(arr.mean())})

def render_about():
    st.title('📖 About')
    st.markdown('This repository is a single deep-learning portfolio containing 20 model families. Each numbered model has its own subfolder and training implementation. The Streamlit application adds a browser-based catalog, preprocessing playground, and an interactive CPU-friendly training lab.')
    st.code('''deep-learning-models/\n├── app.py\n├── model_zoo.py\n├── requirements-streamlit.txt\n├── render.yaml\n├── 01_ann/ ... 16_yolo/\n├── 17_siamese/\n├── 18_seq2seq/\n├── 19_diffusion/\n└── 20_gnn/''')
    st.link_button('GitHub repository',GITHUB)

def main():
    with st.sidebar:
        st.markdown('# 🧠 Deep Learning Lab')
        page=st.radio('Navigate',['Overview','20-Model Catalog','Interactive Training','Image Playground','About'])
        st.divider(); st.link_button('⭐ GitHub',GITHUB,use_container_width=True)
    if page=='Overview': render_overview()
    elif page=='20-Model Catalog': render_catalog()
    elif page=='Interactive Training': render_training()
    elif page=='Image Playground': render_playground()
    else: render_about()
    st.divider(); st.caption('Deep Learning Models · 20 trainable architectures · Streamlit')

if __name__=='__main__': main()
