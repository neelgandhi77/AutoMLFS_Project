import streamlit as st
import seaborn as sn
from pycaret.regression import *
import shap
import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
import plotly.express as px
import sklearn.metrics as metrics

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://th.bing.com/th/id/OIP.Npd77_nhXMQePxy_VsOGnQHaEK?rs=1&pid=ImgDetMain")
    st.title("Regression")
    choice = st.radio("Navigation", ["Upload", "Visualization Filtered", "Visualization Static", "Modelling","Additional"])
    st.info("Let's build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df = df.dropna()
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        st.write(df.shape)
        st.dataframe(df.describe())


if choice == "Visualization Filtered": 
    st.title("Data Visuals")
    df_numeric = df.select_dtypes(include=np.number)
    chosen_option = st.selectbox('Choose the Criteria', ['Select','Countrywise (Year on Year)','Yearwise (Country on Country)'])

    if chosen_option == 'Select':
        st.write("Please Select Criteria")
    if chosen_option == 'Countrywise (Year on Year)':
        chosen_country = st.selectbox('Choose the Coutry', df['Country'].unique())
        chosen_feature = st.selectbox('Choose Feature', df_numeric.columns[1:])
        if st.button('Get Visual'):
            fig1 = px.line(df[df['Country']==chosen_country], x='Year', y=chosen_feature) 
            st.plotly_chart(fig1, use_container_width=True)

    if chosen_option == 'Yearwise (Country on Country)':
        chosen_year = st.selectbox('Choose the Year', df['Year'].unique())
        chosen_feature = st.selectbox('Choose Feature', df_numeric.columns[1:])
        if st.button('Get Visual'):
            fig2 = px.line(df[df['Year']==chosen_year], x='Country', y=chosen_feature) 
            st.plotly_chart(fig2, use_container_width=True)
        
            
if choice == "Visualization Static": 
    st.title("Data Visuals")
    df_numeric = df.select_dtypes(include=np.number)
    st.info("HeatMap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    st.write(fig)


if choice == "Modelling": 
    df_numeric = df.select_dtypes(include=np.number)
    
    chosen_target = st.selectbox('Choose the Target Column', df_numeric.columns[1:])
    model = st.selectbox('Choose Model',['Model Selection', LinearRegression(),Lasso()])
    if model == 'Model Selection':
        st.write("Please Select any One Model")
    else:
        button = st.button('Run Modelling')
        if button: 
            X = df_numeric.drop([chosen_target],axis=1)
            #st.write(X)
            y = df[chosen_target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            #st.info("Accuracy Achieved")
            score = round(model.score(X_test,y_test),2)*100

            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse) # or mse**(0.5)  
            r2 = metrics.r2_score(y_test,y_pred)

            st.write("Results")
            st.write("Score: " + str(score) + "%")
            st.write("MAE:",mae)
            st.write("MSE:", mse)
            st.write("RMSE:", rmse)
            st.write("R-Squared:", r2)
            fig = px.scatter([[y_test,y_pred]], x=y_test, y=y_pred,trendline='ols',trendline_color_override = 'red') 
            st.plotly_chart(fig, use_container_width=True)

        #explainer = shap.Explainer(lr.predict)
        #shap_values = explainer(X_test)
        #fig = shap.plots.waterfall(shap_values[0])
        
        #explainer = shap.KernelExplainer(lr.predict, X_test)
        #shap_values = explainer.shap_values(X_test)
        #fig = shap.summary_plot(shap_values, X_train, plot_type="bar")
        #st.write(fig)

if choice == "Additional":
    pass
    
