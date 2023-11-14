import pandas as pd
import streamlit
import streamlit as st
import io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from st_aggrid import AgGrid

from PIL import Image

# Set title

st.title('Fiber to Retail Analytics')

image = Image.open('./tdslogo.png')
st.image(image, use_column_width=True)

st.write("""
    # A simple Data App With Streamlit
""")

st.write("""
    ### Let's Explore different classifiers and datasets
""")

df: DataFrame = pd.read_excel('superstore_sales.xlsx')
# st.dataframe(df)
# st.write(df.shape)
# st.write(df.columns)
#
# buffer = io.StringIO()
# df.info(buf=buffer)
# s = buffer.getvalue()
# st.text(s)
#
# st.write(df.describe(include='all'))
#
# st.write(df.describe())
#
# st.write(df.isnull().sum())
#

df['month_year'] = df['order_date'].apply(lambda x: x.strftime('%Y-%m'))
# st.write(df['month_year'])
# st.dataframe(df)
df_trend = df.groupby(['month_year'])[['sales', 'quantity', 'discount']].sum()['sales'].reset_index()
# st.write(df_trend)
fig = plt.figure(figsize=(20, 6))
plt.plot(df_trend['month_year'], df_trend['sales'])
plt.xticks(rotation='vertical', size=8)
st.pyplot(fig)

prod_sales = pd.DataFrame(df.groupby(['product_name'])[['sales', 'quantity', 'discount']].sum()['sales'])
prod_sales = prod_sales.sort_values('sales', ascending=False)
st.write(prod_sales)


most_sell_prod = pd.DataFrame(df.groupby('product_name')[['sales', 'quantity', 'discount']].sum()['quantity'])
most_sell_prod = most_sell_prod.sort_values('quantity', ascending=False)
st.write(most_sell_prod)

fig2 = plt.figure(figsize=(20, 6))
sns.countplot(x=df['ship_mode'])
st.pyplot(fig2)


most_prof_cat_subcat = pd.DataFrame(df.groupby(['category', 'sub_category'])[['sales', 'profit', 'quantity', 'discount']].sum()['profit'])
most_prof_cat_subcat = most_prof_cat_subcat.sort_values(['category', 'profit'], ascending=False)
st.write(most_prof_cat_subcat)












st.write("""
    ### Let's Explore different classifiers and datasets
""")
