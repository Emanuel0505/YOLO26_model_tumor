from ultralytics import YOLO
from PIL import Image
import streamlit as st
import numpy as np
import requests
import os

st.set_page_config(
    page_title="Analisador de Tumores Cerebrais",
    layout="centered"
)

st.title("Analizador de Tumor Cerebrais com YOLO")
st.markdown("""
    Faça o upload de uma imagem de **Ressonância Magnética** ou **Tomografia**.
    A IA irá segmentar e classificar o tipo de tumor presente, dos três tipos:
    * **Glioma**;
    * **Meningioma**;
    * **Pituitary**;
    """)
st.divider()


MODEL_URL = "https://github.com/Emanuel0505/YOLO26_model_tumor/releases/download/Arquivo-de-treinamento/best.pt"
MODEL_PATH = os.path.join("models", "best.pt")

def download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        with requests.get(MODEL_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    return MODEL_PATH

@st.cache_resource
def carregar_modelo():
    return YOLO(download_model())

try: 
    model = carregar_modelo()
except Exception as e:
    st.error(f"Erro ao carregar o modelo de IA. Erro{e}")


file_config = st.file_uploader(
    label="Escolha uma imagem de Exam e com formato (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if file_config is not None:
    imagem_original = Image.open(file_config)
    imagem_original = np.array(imagem_original)
    
    st.write("")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        botao_analisar = st.button("🚀 Iniciar Segmentação por IA", type="primary", use_container_width=True)
    st.divider()
    
    # Layout de duas colunas para o Antes e Depois
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Imagem Original")
        st.image(imagem_original, use_container_width=True)
        
    with col2:
        st.subheader("🔍 Segmentação da IA")
        
        if botao_analisar:
            with st.spinner("A IA está analisando o exame... Por favor, aguarde."):
                result = model.predict(source=imagem_original, conf=0.25)
                
                image_plot_bgr = result[0].plot()
                
                image_plot_rgb = image_plot_bgr[..., ::-1]
                
                st.image(image_plot_rgb, use_container_width=True)
        else:
            # Caso o upload tenha sido feito, mas o botão ainda não foi clicado
            st.info("💡 Clique no botão **'Iniciar Segmentação por IA'** acima para processar este exame.")
    if botao_analisar:
        st.write("")
        st.subheader("📋 Relatório Clínico da IA")
        
        class_name = result[0].names
        class_detected = result[0].boxes.cls
        
        if len(class_detected) == 0:
            st.success("✅ **Nenhum tumor detectado** com base no limiar de confiança atual (0.25).")
        else:
            names_found = [class_name[int(c)] for c in class_detected]
            names_unics = set(names_found)
            
            for tumor in names_unics:
                st.warning(f"⚠️ **ALERTA:** Massa tumoral detectada compatível com: **{tumor.upper()}**")
                
            st.markdown("### 📊 Nível de Confiança por Detecção")
            trusts = result[0].boxes.conf
            
            metric_cols = st.columns(len(class_detected))
            for idx, classe_id in enumerate(class_detected):
                nome_tumor = class_name[int(classe_id)].upper()
                value_conf = trusts[idx] * 100
                
                with metric_cols[idx]:
                    st.metric(label=f"Precisão ({nome_tumor})", value=f"{value_conf:.2f}%")