import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import plotly
import platform
import time
from PIL import Image

# Systemada PosixPath yo'qligi uchun
import pathlib

plt = platform.system()
if plt == "Linux":
  pathlib.WindowsPath = pathlib.PosixPath
else:
  pathlib.PosixPath = pathlib.WindowsPath


# # app title ni yo ikonkasini o'zgartirish
st.set_page_config(
    page_title="Klassifikatsiya Modeli!",
    page_icon="🧊",
    # page_icon="tada",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Qush, Baliq va Yirtqich hayvonlarni klassifikatsiya qiluvchi model.")

# Rasimni joylash
with st.spinner("Wait for it ..."):
  time.sleep(0.2)
  file = st.file_uploader("Rasm yuklash. Baliq yoki Qush yo Yirtqich hayvon rasmini kiriting!", type=['png', 'jpg', 'jpeg', 'gif', 'svg'])

if file:
  # st.image(file)

  # PIL convertatsiya
  img = PILImage.create(file)
  st.image(img, 255, 255)

  # model
  model = load_learner("fishs_birds_animals_model.pkl")

  # prediction
  pred, pred_id, probs = model.predict(img)

  # chop qilish
  if pred == "Bird":
    st.success(f"Bashorat: Qush")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

  elif pred == "Carnivore":
    st.success(f"Bashorat: Yirtqich")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

  elif pred == "Fish":
    st.success(f"Bashorat: Baliq")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

  # ploting
  fig = px.bar(x=probs*100, y=model.dls.vocab)
  st.plotly_chart(fig)


