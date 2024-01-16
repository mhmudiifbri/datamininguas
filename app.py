import streamlit as streamlit
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import pickle

#import model
svm = pickle.load(open('SVC.pkl', 'rb'))

# load dataset
data = pd_read_csv('diabetes.csv')
data = data.drop(data.columns[0], axis=1)

st.title('Diabetes Prediction')

html_layout1 = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Diabetes Checkup</h2>
</div>
"""

st.markdown(html_layout1, unsafe_allow_html=True)
activities = ['SVM', 'Model Lain']
option = st.sidebar.selectBox('Pilihan', activities)
st.sidebar.header('Data Pasien')

if st.checkbox('Tentang Dataset'):
    html_layout2 = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Ini adalah Dataset PIMA Indian</h2>
    </div>
    """
    st.markdown(html_layout2, unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Deskripsi Dataset')
    st.write(data.describe())

sns.set_style('darkgrid')
if st.checkbox('EDA'):
    pr = ProfileReport(data, explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)

# train test split
X = data.drop(['Outcome'], axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader('y_train')
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.head())
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    kehamilan = st.sidebar.slider('Kehamilan', 0, 20, 1)
    glukosa = st.sidebar.slider('Glukosa', 0, 200, 108)
    bp = st.sidebar.slider('Tekanan Darah', 0, 140, 40)
    ketebalan = st.sidebar.slider('Ketebalan Kulit', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 1000, 120)
    bmi = st.sidebar.slider('BMI', 0, 80, 25)
    diabetespd = st.sidebar.slider('Diabetes Pedigree Function', 0.05, 2.6, 0.45)
    age = st.sidebar.slider('Umur', 21, 100, 24)

    user_report_data = {
        'Pregnancies': kehamilan,
        'Glucose': glukosa,
        'BloodPressure': bp,
        'SkinThickness': ketebalan,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetespd,
        'Age': age
    }

    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Data Pasien
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = svm.predict(user_data)
svc_score = accuracy_score(y_test, svm.predict(X_test))

# output
st.subheader('Hasilnya adalah: ')
output = ''
if user_result[0] == 0:
    output = 'Tidak Terkena Diabetes'
else:
    output = 'Terkena Diabetes'
st.title(output)
st.subheader('Model yang digunakan' + option)
st.subheader('Akurasi Model')
st.write(str(svc_score*100)+'%')

