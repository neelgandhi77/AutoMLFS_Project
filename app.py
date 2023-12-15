import streamlit as st
import seaborn as sn
from pycaret.regression import *
from sklearn.ensemble import *
import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
import plotly.express as px
import sklearn.metrics as metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.font_manager
import h2o
from h2o.automl import H2OAutoML
import requests



plt.rcParams.update({
    'font.family':'sans-serif',
    'font.sans-serif':['Liberation Sans'],
    })


st.set_page_config(
        page_title="Regression FS",
)


def model_train_test_results(X,y,model):
    
    test_size = 0.25
    st.subheader("Data Split Configuration")
    split_mode = st.selectbox('Choose Mode',["Select", "Mannual","Auto (75-25)"])
  
    
    if split_mode == "Select":
        st.warning("Please Select any One")
    
    else:
        if split_mode=="Mannual":
            test_size = float(st.number_input("Enter Test Size Between 0 to 1"))
    
        if st.button("Run Modelling"):
            if (test_size > 0 and test_size <= 1):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
                model.fit(X_train, y_train)
                        
                y_pred = model.predict(X_test)
                #st.info("Accuracy Achieved")        
                score = round(model.score(X_test,y_test),2)*100
                mae = metrics.mean_absolute_error(y_test, y_pred)
                mse = metrics.mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse) # or mse**(0.5)  
                r2 = metrics.r2_score(y_test,y_pred)


                st.header("Results", divider='rainbow')  
                st.subheader("Score: " + str(score) + "%")
                st.write("MAE:",mae)           
                st.write("MSE:", mse)
                st.write("RMSE:", rmse)
                st.write("R-Squared:", r2)
                fig = px.scatter([[y_test,y_pred]], x=y_test, y=y_pred,trendline='ols',trendline_color_override = 'red') 
                st.plotly_chart(fig, use_container_width=True)
                
        
            
            else:
                st.warning("please Enter value in given range")

            

def selected_features_train(X,y):
    selected_features = st.multiselect("Please Select features",options=pd.Series(X.columns),default= pd.Series(X.columns))
    st.write(selected_features)
    model = st.selectbox('Choose Model',['Model Selection', 'LinearRegression','Lasso'])

    if model=="LinearRegression":
        model=LinearRegression()
    if model=="Lasso":
        model=Lasso()
    if model == 'Model Selection':
        st.warning("Please Select any One Model")
    else:
        
        X = X[selected_features]    
        #st.write(X)
     
        model_train_test_results(X[selected_features],y,model)
            
if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)


with st.sidebar: 
    st.image("https://th.bing.com/th/id/OIP.Npd77_nhXMQePxy_VsOGnQHaEK?rs=1&pid=ImgDetMain")
    st.title("Regression")
    choice = st.radio("Navigation", ["Upload or Fetch", "Visualization Filtered --Specific One", "Visualization Static", "Modelling","FS","Additional"])
    st.info("Let's build and explore your data.")


if choice == "Upload or Fetch":
    
    st.title("Data Collection")
    selected_option = st.selectbox('Choose the Criteria', ['Select','Upload','Covid 19 API Fetch Data'])

    if selected_option == 'Select':
        st.warning("Please Select Criteria")

    if selected_option == "Upload":
        file = st.file_uploader("Upload Your Dataset")
        with st.status("Upload...", expanded=True) as status:
            if file: 
                df = pd.read_csv(file, index_col=None)
                df = df.dropna()
                
                df.to_csv('dataset.csv', index=None)
                st.dataframe(df)
    
    if selected_option == "Covid 19 API Fetch Data":
        with st.status("Fetching data...", expanded=True) as status:
            url = "https://covid-193.p.rapidapi.com/statistics"

            headers = {
                "X-RapidAPI-Key": "baa241d513msh46120bce8d7ae00p17d4cdjsn3e4f60e0bed8",
                "X-RapidAPI-Host": "covid-193.p.rapidapi.com"
            }

            response = requests.get(url, headers=headers)
            data = response.json()
            #print(response.json())
            df = pd.json_normalize(data,'response')
            df.columns = df.columns.str.replace(".", "_", regex=True)
            df = df.fillna(0)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)
    

if choice == "Visualization Filtered --Specific One": 
    st.header("Data Visuals",divider="rainbow")
    df_numeric = df.select_dtypes(include=np.number)
    df_numeric[df_numeric <  0] = 0
    chosen_option = st.selectbox('Choose the Criteria', ['Select','Countrywise (Year on Year)','Yearwise (Country on Country)'])

    if chosen_option == 'Select':
        st.warning("Please Select Criteria")
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
    st.header("Modelling",divider="rainbow")
    df_numeric = df.select_dtypes(include=np.number)
    
    chosen_target = st.selectbox('Choose the Target Column', df_numeric.columns[1:])
    model = st.selectbox('Choose Model',['Model Selection', 'LinearRegression','Lasso'])
    X = df_numeric.drop([chosen_target],axis=1)
    y = df[chosen_target]
    
    if model=="LinearRegression":
        model=LinearRegression()
    if model=="Lasso":
        model=Lasso()
    if model == 'Model Selection':
        st.warning("Please Select any One Model")
    else:
        #button = st.button('Run Modelling')
        #if button==True: 
        model_train_test_results(X,y,model)
           


if choice == "FS":
    st.header("Feature Selection",divider="rainbow")
    df_numeric = df.select_dtypes(include=np.number)
    df_numeric[df_numeric <  0] = 0
    chosen_target = st.selectbox('Choose the Target Column', df_numeric.columns[1:])
    X = df_numeric.drop([chosen_target],axis=1)
    y = df[chosen_target].astype('int')
    

    chosen_FS_Method = st.selectbox("Choose FS Method",['Select Method','CHI Test (SelectKBest)','Extra Tree Classifier'])
    if chosen_FS_Method=='Select Method':
        st.warning("Please Select any One Method")

    if chosen_FS_Method=="CHI Test (SelectKBest)":
        st.subheader("Best feature with score")
        bestfeatures = SelectKBest(score_func=chi2, k=len(df_numeric.columns)-1)
        fit = bestfeatures.fit(X,y)            
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        featureScores.columns = ['features','Score'] 
        featureScores = featureScores.sort_values(by=['Score'],ascending=False)
        st.write(featureScores)
        fig = px.bar(featureScores,x='features', y='Score')
        st.write(fig)

    if chosen_FS_Method=="Extra Tree Classifier":
        try:
            st.subheader("Best feature with Importance")
            bestfeatures = ExtraTreesClassifier()
            fit = bestfeatures.fit(X,y)
            dfscores = pd.DataFrame(fit.feature_importances_)
            dfcolumns = pd.DataFrame(X.columns)
            featureScores = pd.concat([dfcolumns,dfscores],axis=1)
            featureScores.columns = ['features','Importance'] 
            featureScores = featureScores.sort_values(by=['Importance'],ascending=False)
            st.write(featureScores)
            
            fig = px.bar(featureScores,x='features', y='Importance')
            st.write(fig)
        except:
            st.warning("Unable to display, Streamlit came accross Error --typically a Resource Error/ Font Type Error")
      
    selected_features_train(X,y)
    
        

if choice == "Additional":
    st.header("Data Visuals - Add.",divider="rainbow")
    corrmat = df.corr()
    top_corr_features = corrmat.index
    fig, ax = plt.subplots(figsize=(26,20))
    sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn",ax=ax)
    st.write(fig)
    
