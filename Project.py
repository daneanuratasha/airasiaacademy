import streamlit as st
import pandas as pd
import pickle

st.write("""
# Advertising Sales Prediction App

This app predicts the **Sales** based on Advertising
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 10.0, 300.0, 151.5)
    Radio = st.sidebar.slider('Radio', 10.0, 50.0, 41.3)
    Newspaper = st.sidebar.slider('Newspaper', 55.0, 90.0, 58.4)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}         
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("Advertising-model-LRR.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
