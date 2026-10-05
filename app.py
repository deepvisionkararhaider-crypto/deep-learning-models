from pathlib import Path
import sys
import streamlit as st
from PIL import Image

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from model_zoo import MODEL_SPECS,SPEC_BY_ID,train_one
from real_datasets import DATASET_CATALOG

st.set_page_config(page_title='Deep Learning Models | Real Data Lab',page_icon='🧠',layout='wide')
GITHUB='https://github.com/deepvisionkararhaider-crypto/deep-learning-models'

@st.cache_data(show_spinner=False)
def cached_train(model_id,epochs,lr):
    _,losses,metric=train_one(model_id,epochs=epochs,lr=lr)
    return losses,metric

def overview():
    st.title('🧠 Deep Learning Models — Real Data Lab')
    st.caption('20 trainable architectures, each mapped to a real public benchmark dataset and concrete learning task.')
    c1,c2,c3,c4=st.columns(4); c1.metric('Models','20'); c2.metric('Benchmark families','6'); c3.metric('Trainable','20 / 20'); c4.metric('Runtime','PyTorch + Streamlit')
    st.divider(); st.subheader('Real-data model map')
    for start in range(0,len(MODEL_SPECS),4):
        cols=st.columns(4)
        for col,(n,name,_,kind,desc) in zip(cols,MODEL_SPECS[start:start+4]):
            ds,source,task=DATASET_CATALOG[n]
            with col:
                st.markdown(f'### {n:02d} · {name}'); st.caption(task); st.write(desc); st.markdown(f'**Dataset:** {ds}'); st.caption(source)
    st.info('The app no longer trains on synthetic placeholder samples. It pulls a real benchmark dataset on first use, caches it, and trains a CPU-sized sample for the public demo.')

def catalog():
    st.title('📦 20-Model Catalog')
    folders={1:'ann',2:'cnn',3:'rnn',4:'lstm',5:'gru',6:'autoencoder',7:'vae',8:'gan',9:'transformer',10:'bert',11:'gpt',12:'resnet',13:'densenet',14:'vit',15:'unet',16:'yolo',17:'siamese',18:'seq2seq',19:'diffusion',20:'gnn'}
    for n,name,cls,kind,desc in MODEL_SPECS:
        ds,source,task=DATASET_CATALOG[n]
        with st.expander(f'{n:02d} · {name} — {task}'):
            st.write(desc); st.write(f'**Dataset:** {ds}'); st.write(f'**Source:** {source}'); st.write(f'**Task:** {task}'); st.write(f'**Python class:** `{cls.__name__}`')
            st.link_button('Open implementation',f'{GITHUB}/tree/main/{n:02d}_{folders[n]}',use_container_width=True)

def training():
    st.title('🎛️ Real Dataset Training Lab')
    names={n:f'{n:02d} · {name}' for n,name,_,_,_ in MODEL_SPECS}; selected=st.selectbox('Architecture',list(names),format_func=lambda n:names[n])
    _,name,_,kind,desc=SPEC_BY_ID[selected]; ds,source,task=DATASET_CATALOG[selected]
    a,b=st.columns(2)
    with a: st.markdown(f'**Dataset:** {ds}'); st.markdown(f'**Task:** {task}'); st.caption(source)
    with b: st.markdown(f'**Architecture:** {name}'); st.caption(desc)
    epochs=st.slider('Epochs',1,10,3); lr=st.select_slider('Learning rate',[0.0001,0.0003,0.001,0.003],value=0.001,format_func=lambda x:f'{x:g}')
    if st.button('🚀 Train on real data',type='primary',use_container_width=True):
        with st.spinner(f'Loading {ds} and training {name}...'):
            losses,metric=cached_train(selected,epochs,lr)
        st.success(f'{name} trained on {ds}.'); m1,m2=st.columns(2); m1.metric('Final loss',f'{losses[-1]:.4f}'); m2.metric('Demo metric',f'{metric:.2%}'); st.line_chart({'loss':losses})
        st.caption('The public demo uses a reproducible CPU-sized subset. Metrics are demo-training metrics, not leaderboard results.')

def prediction():
    st.title('🔮 Prediction Studio')
    st.write('The studio lets you supply model-family inputs and inspect the expected preprocessing contract; training remains in the Real Dataset Training page.')
    n=st.selectbox('Model',list(range(1,21)),format_func=lambda i:f'{i:02d} · {SPEC_BY_ID[i][1]}'); ds,source,task=DATASET_CATALOG[n]; st.info(f'**Dataset:** {ds} · **Task:** {task}')
    if n==1:
        st.write('Enter 30 standardized breast-cancer features.'); st.text_input('30 comma-separated features',value='0,'*29+'0'); st.caption('Educational classifier only; not a medical diagnosis tool.')
    elif n in (2,6,7,8,12,13,14,15,16,17):
        up=st.file_uploader('Upload a digit image (PNG/JPG)',type=['png','jpg','jpeg'])
        if up: st.image(Image.open(up).convert('L'),width=180,caption='Uploaded input'); st.info('The benchmark is the real UCI handwritten-digits dataset; this upload previews the expected image input path.')
    elif n in (3,4,5):
        st.info('HAR models expect a 93×6 sensor window from the UCI smartphone activity dataset.'); st.file_uploader('HAR CSV (558 numeric values)',type=['csv'])
    else:
        text=st.text_area('Text input',height=150,placeholder='Paste a short newsgroup-style text sample...');
        if text: st.write(f'Characters received: {len(text)}')
    st.warning('The studio is an educational interface and should not be used for medical, financial, or other high-stakes decisions.')

def about():
    st.title('📖 About the real-data upgrade')
    st.markdown('Every architecture is now mapped to a real public benchmark. Lightweight cached subsets keep the free Streamlit deployment responsive. Dataset attribution, task definitions, and access notes are documented in `DATASETS.md`.')
    st.link_button('GitHub repository',GITHUB)

with st.sidebar:
    st.markdown('# 🧠 Deep Learning Lab')
    page=st.radio('Navigate',['Overview','🔮 Prediction Studio','🎛️ Real Dataset Training','📦 20-Model Catalog','About'])
    st.divider(); st.link_button('⭐ GitHub',GITHUB,use_container_width=True)
if page=='Overview': overview()
elif page=='🔮 Prediction Studio': prediction()
elif page=='🎛️ Real Dataset Training': training()
elif page=='📦 20-Model Catalog': catalog()
else: about()
st.divider(); st.caption('20 trainable deep-learning architectures · real public benchmark datasets · Streamlit Community Cloud')
