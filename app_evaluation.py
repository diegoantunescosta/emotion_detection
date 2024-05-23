import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import random
import base64
import requests

model = YOLO("deteccionModel.pt", "v8")

def carregar_imagem(url_imagem):
    if url_imagem.startswith('data:image'):
        imagem_bytes = base64.b64decode(url_imagem.split(",")[1])
        imagem_array = np.frombuffer(imagem_bytes, np.uint8)
        imagem = cv2.imdecode(imagem_array, cv2.IMREAD_COLOR)
    else:
        resposta = requests.get(url_imagem)
        imagem_bytes = np.frombuffer(resposta.content, np.uint8)
        imagem = cv2.imdecode(imagem_bytes, cv2.IMREAD_COLOR)
    return imagem

def detectar_imagem(imagem):
    detections = []
    detect_params = model.predict(source=[imagem], conf=0.30, save=False)
    DP = detect_params[0].cpu().numpy()
    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            box = detect_params[0].boxes[i]
            clsID = box.cls.cpu().numpy()[0]
            conf = box.conf.cpu().numpy()[0]
            class_name = model.names[int(clsID)]
            detections.append((class_name, conf))
    return detections

def main():
    st.title("Detecção de Objetos com YOLO")
    st.write("Faça o upload de uma imagem para detectar objetos.")

    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)



        with st.spinner('Detectando...'):
            detections = detectar_imagem(image)

        if detections:
            max_conf_detection = max(detections, key=lambda x: x[1])
            max_class_name, max_conf = max_conf_detection
            st.write(f"Classe detectada com maior confiança: {max_class_name}, confiança: {max_conf:.2f}")

            for class_name, conf in detections:
                if class_name == max_class_name:
                    font_scale = 1
                    font_thickness = 2
                else:
                    font_scale = 0.8
                    font_thickness = 1
                
                image = cv2.putText(image, f"{class_name}: {conf:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
            
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Imagem com Detecções")
            nome_arquivo_salvo = f"imagem_detectada_{class_name}.jpg"
            cv2.imwrite(nome_arquivo_salvo, image)
            st.success(f"Imagem salva como {nome_arquivo_salvo}.")
        else:
            st.write("Nenhum objeto detectado.")

if __name__ == "__main__":
    main()
