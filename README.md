# 🦴 Bone Fracture Detection with Deep Learning

Sistema de detecção automática de fraturas ósseas em imagens de raio-X utilizando Deep Learning (CNN + Transfer Learning).

Este projeto cobre todo o pipeline de Machine Learning:

✔ Exploração dos dados  
✔ Pré-processamento  
✔ Treinamento de modelos  
✔ Avaliação com métricas  
✔ Interpretabilidade (Grad-CAM)  
✔ Deploy com Streamlit  

---

# 📌 Objetivo

Auxiliar hospitais e profissionais de saúde na identificação automática de fraturas ósseas a partir de exames de imagem.

O modelo recebe uma imagem de raio-X e retorna:

👉 **Fratura** ou **Normal**

Buscando:
- reduzir tempo de triagem
- apoiar diagnósticos
- diminuir erros humanos

---

# 🧠 Tecnologias utilizadas

- Python 3.8+
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit
- Google Colab

---

# 📂 Dataset

Utilizado o:

Human Bone Fractures Multi-modal Image Dataset (HBFMID)

Estrutura esperada:

Bone Fractures Detection/
   ├── fracture/
   ├── normal/

No Google Colab:

/content/drive/MyDrive/AluraDrive/Bone Fractures Detection

---

# 🔍 Etapa 1 — Exploração dos Dados

Análises realizadas:

✅ contagem total de imagens  
✅ distribuição das classes  
✅ verificação de desbalanceamento  
✅ visualização de amostras (grids)  
✅ inspeção de qualidade e resolução  

Principais observações:

- classes visualmente semelhantes  
- ruído nas imagens  
- contraste baixo em alguns exames  
- possível desbalanceamento  

Esses fatores tornam a classificação mais desafiadora.

---

# ⚙️ Etapa 2 — Pré-processamento

Aplicado:

- redimensionamento → 224x224
- normalização (0–1)
- divisão treino/teste estratificada
- Data Augmentation:
  - rotação
  - zoom
  - flip horizontal
  - deslocamentos

Objetivo:
melhorar generalização e reduzir overfitting.

---

# 🤖 Etapa 3 — Modelagem

Foram testadas duas abordagens.

## 🔹 CNN do zero

Arquitetura simples:

Conv → Pool → Conv → Pool → Dense → Softmax

Vantagens:
- simples
- didática

Desvantagens:
- menor desempenho
- precisa de mais dados

---

## 🔹 Transfer Learning (MobileNetV2) ⭐

Modelo pré-treinado no ImageNet usado como extrator de características.

Vantagens:
- maior acurácia
- treino mais rápido
- melhor generalização

Foi a abordagem com melhor resultado.

---

# 📊 Avaliação

Métricas utilizadas:

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusão
- Curvas de aprendizado

Exemplo:

Accuracy: 94%  
F1-score: 0.93  

---

# 🔬 Interpretabilidade — Grad-CAM

Implementado Grad-CAM para visualizar:

👉 regiões da imagem mais importantes para a decisão do modelo

Benefícios:

- maior confiança clínica
- explicabilidade
- validação do comportamento do modelo

---

# 🚀 Etapa 4 — Aplicação Web (Streamlit)

Aplicação interativa permite:

✔ upload de imagem  
✔ processamento automático  
✔ previsão em tempo real  
✔ exibição do resultado  

---

# ▶ Como executar

## 1️⃣ Clonar repositório

git clone https://github.com/seu-usuario/bone-fracture-detection.git  
cd bone-fracture-detection

---

## 2️⃣ Instalar dependências

pip install -r requirements.txt

---

## 3️⃣ Treinar modelo (opcional)

python src/treino.py

---

## 4️⃣ Rodar aplicação

streamlit run app.py

---

# 📦 requirements.txt

tensorflow  
opencv-python  
numpy  
matplotlib  
seaborn  
scikit-learn  
streamlit  

---

# 🗂 Estrutura do projeto

bone-fracture-detection/
│
├── notebooks/
│   ├── exploracao.ipynb
│   ├── treino.ipynb
│
├── src/
│   ├── treino.py
│   ├── predicao.py
│   ├── gradcam.py
│
├── models/
│   ├── fracture_model.h5
│
├── app.py
├── requirements.txt
├── README.md

---

# 🔄 Pipeline completo

Dataset  
↓  
Exploração  
↓  
Pré-processamento  
↓  
Treinamento  
↓  
Avaliação  
↓  
GradCAM  
↓  
Salvar modelo  
↓  
Deploy Streamlit  

---

# 💡 Melhorias futuras

- mais classes de fratura
- fine tuning completo
- ensemble de modelos
- API REST
- deploy em nuvem (AWS/GCP)
- validação clínica real

---

# 👨‍💻 Autor

Wesley de Sousa Costa  
Projeto educacional — Deep Learning aplicado à saúde
