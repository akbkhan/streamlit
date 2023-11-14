import pandas as pd
import streamlit
import streamlit as st
import io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from st_aggrid import AgGrid

from PIL import Image

# Set title

st.title('Fiber to Retail Analytics')

image = Image.open('tdslogo.png')
st.image(image, use_column_width=True)

st.write("""
    # A simple Data App With Streamlit
""")

st.write("""
    ### Let's Explore different classifiers and datasets
""")


df = pd.read_csv('Ecommerce Purchases')
st.write(df.head())
st.write(df.tail())
st.write(df.dtypes)
st.write(df.isnull().sum())
st.write(df.columns)
st.write(df.shape)

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.write(df['Purchase Price'].max())
st.write(df['Purchase Price'].min())
st.write(df['Purchase Price'].mean())
st.write(len(df[df['Language']=='fr']))
st.write(len(df[df['Job'].str.contains('engineer', case=False)]))