import streamlit as st
#from pycaret.regression import *
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
#import plotly.graph_objects as go

import requests
import shap
from shap import Explainer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



st.set_option('deprecation.showPyplotGlobalUse', False)

plt.rcParams.update({
    'font.family':'sans-serif',
    'font.sans-serif':['Liberation Sans'],
    })


st.set_page_config(
        page_title="AutoML FS",
)


def selected_feature(X_train,y_train,model,problem_type):
    st.header("Recommended Features")
            
    model.fit(X_train, y_train)

    if problem_type == "Regression":
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
        st.pyplot(shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, plot_type="bar"))
        #st.pyplot(shap.summary_plot(shap_values, X_train, feature_names= X_train.columns))
        feature_importance = pd.DataFrame(shap_values.values, columns=X_train.columns).abs().mean().sort_values(ascending=False)
            
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_for_processing)
        st.pyplot(shap.summary_plot(shap_values[1], X_train, plot_type="bar")) 
        #st.pyplot(shap.summary_plot(shap_values, X_train, feature_names= X_train.columns)
        feature_importance = pd.DataFrame(shap_values[1], columns=X_train.columns).abs().mean().sort_values(ascending=False)

    return feature_importance
                      
def model_train_test_results(X,y,model,tag):
    
    test_size = 0.25
    st.subheader("Data Split Configuration")
    split_mode = st.selectbox('Choose Mode',["Auto (75-25)", "Manual"])
  
    
    if split_mode == "Select":
        st.warning("Please Select any One")
    
    else:
        if split_mode=="Manual":
            test_size = float(st.number_input("Enter Test Size Between 0 to 1"))
    
        if st.button("Run Modelling"):
            if (test_size > 0 and test_size < 1):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
                model.fit(X_train, y_train)
                        
                y_pred = model.predict(X_test)

                if tag =="Regression":

                    #st.info("Accuracy Achieved")        
                    score = round(model.score(X_test,y_test),5) * 100
                    #mae = metrics.mean_absolute_error(y_test, y_pred)
                    #mse = metrics.mean_squared_error(y_test, y_pred)
                    #rmse = np.sqrt(mse) # or mse**(0.5)  
                    #r2 = metrics.r2_score(y_test,y_pred)


                    st.header("Results", divider='rainbow')  
                    st.subheader("Score: " + str(score) + "%")
                    #st.write("MAE:",mae)           
                    #st.write("MSE:", mse)
                    #st.write("RMSE:", rmse)
                    #st.write("R-Squared:", r2)
                    #fig = px.scatter([[y_test,y_pred]], x=y_test, y=y_pred,trendline="ols",trendline_color_override = 'red') 
                    #st.plotly_chart(fig, use_container_width=True)
                     
                else:

                    score = round(accuracy_score(y_test, y_pred),5) * 100
                    st.header("Results", divider='rainbow')  
                    st.subheader("Score: " + str(score) + "%")

                    #Generate the confusion matrix
                    #cf_matrix = confusion_matrix(y_test, y_pred)
                    #fig,ax = plt.subplots()
                    

                    #ax.set_title('Confusion Matrix\n\n')
                    #ax.set_xlabel('\nPredicted Values')
                    #ax.set_ylabel('Actual Values ')

                    #ax.xaxis.set_ticklabels(['False','True'])
                    #ax.yaxis.set_ticklabels(['False','True'])
                    #sns.heatmap(cf_matrix, annot=True, cmap='Blues',ax=ax)
                    #st.write(fig)

        
            
            else:
                st.warning("Please Enter value in given range...")

def check_problem(chosen_target):
  
    problem_type=""
    
    target_data_type = df[chosen_target].dtype

    # Count the number of unique values in the target column
    unique_values_count = df[chosen_target].nunique()

    # Determine if it's more suitable for classification or regression
    if target_data_type == 'object' or unique_values_count <= 10:
        problem_type = "Classification"
    else:
        problem_type = "Regression"

    return problem_type

def selected_features_train(X,y,chosen_target):
    selected_features = st.multiselect("Please Select features",options=pd.Series(X.columns),default= pd.Series(X.columns))
    st.write(selected_features)

    problem_type = check_problem(chosen_target)

    if problem_type == "Regression":
        model = st.selectbox('Choose Model',['Model Selection', 'LinearRegression','Lasso'])

        if model=="LinearRegression":
            model=LinearRegression()
        if model=="Lasso":
            model=Lasso()
        if model == 'Model Selection':
            st.warning("Please Select any One Model")

    else:
        
        model = st.selectbox('Choose Model',['Model Selection', 'Random Forest','Decision Tree'])

        if model=="Random Forest":
            model= RandomForestClassifier(n_estimators=100,random_state=1200)
        if model=="Decision Tree":
            model= DecisionTreeClassifier(criterion="gini",
                                      random_state=100)
        if model == 'Model Selection':
            st.warning("Please Select any One Model")
    
    if(model !="Model Selection"):
        
        X = X[selected_features]    
        #st.write(X)
     
        model_train_test_results(X[selected_features],y,model,problem_type)
   

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)


with st.sidebar: 
    st.image("https://th.bing.com/th/id/OIP.Npd77_nhXMQePxy_VsOGnQHaEK?rs=1&pid=ImgDetMain")
    st.title("AutoML")
    #choice = st.radio("Navigation", ["Upload or Fetch", "Visualization Filtered --Specific One", "Visualization Static", "Modelling","FS","SHAP"])
    choice = st.radio("Navigation", ["Data Access", "Train & Test","Deploy"])
    
    st.info("Let's build and explore your data.")


if choice == "Data Access":
    
    st.title("Data Access")
    selected_option = st.selectbox('Choose the Criteria', ['Select','Upload','Covid 19 API Fetch Data'])

    
    if selected_option == 'Select':
        st.warning("Please Select Criteria")
    
    st.image("Images/upload3.png")
    
    if selected_option == "Upload":
        file = st.file_uploader("Upload Your Dataset")
        with st.status("Upload...", expanded=True) as status:
            if file: 
                df = pd.read_csv(file, index_col=None)
                numeric_columns = df.select_dtypes(include=['number']).columns

                # Fill missing values with mean for numeric columns
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
                if "Pass/Fail" in df.columns:
                    df.loc[df["Pass/Fail"] == -1, "Pass/Fail"] = 0
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
    
    X = df_numeric.drop([chosen_target],axis=1)
    y = df[chosen_target]
    
    model=""
    problem_type = check_problem(chosen_target)

    if problem_type == "Regression":
        model = st.selectbox('Choose Model',['Model Selection', 'LinearRegression','Lasso'])

        if model=="LinearRegression":
            model=LinearRegression()
        if model=="Lasso":
            model=Lasso()
        if model == 'Model Selection':
            st.warning("Please Select any One Model")
       
        
    else:
        
        model = st.selectbox('Choose Model',['Model Selection', 'Random Forest','Decision Tree'])

        if model=="Random Forest":
            model= RandomForestClassifier(n_estimators=100,random_state=1200)
        if model=="Decision Tree":
            model= DecisionTreeClassifier(criterion="gini",
                                      random_state=100)
        if model == 'Model Selection':
            st.warning("Please Select any One Model")
        
    if(model != "Model Selection"):
        model_train_test_results(X,y,model,problem_type)


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
      
    selected_features_train(X,y,chosen_target)


if choice == "Train & Test":
   
    #process_counter = False
   
    st.header("SHAP",divider="rainbow")
    df_numeric = df.select_dtypes(include=np.number)
  
    
    chosen_target = st.selectbox('Choose the Target Column', df_numeric.columns[1:])
    problem_type= check_problem(chosen_target)

    X = df_numeric.drop([chosen_target],axis=1)
    y = df[chosen_target]
    X_for_processing = X
    y_for_processing = y

    X_train, X_test, y_train, y_test = train_test_split(X_for_processing, y_for_processing, test_size = 0.25)
    
    st.selectbox('Choose Process',['AutoML','Manual ML',])

    if problem_type == "Regression":
        model = st.selectbox('Choose Model',['Linear Regression','Lasso'])

        if model=="Linear Regression":
            model=LinearRegression()
        if model=="Lasso":
            model=Lasso()
        if model == 'Model Selection':
            st.warning("Please Select any One Model")
    else:
        
        model = st.selectbox('Choose Model',['Random Forest','Decision Tree'])

        if model=="Random Forest":
            model= RandomForestClassifier(n_estimators=100,random_state=1200)
        if model=="Decision Tree":
            model= DecisionTreeClassifier(criterion="gini",
                                      random_state=100)
        if model == 'Model Selection':
            st.warning("Please Select any One Model")
    

    if(model != "Model Selection"):
        
            top_features = selected_feature(X_train,y_train,model,problem_type)
            #top_features = feature_importance.index[:7]
            #st.write(top_features.index[:7])
            with st.form("my_form"):

                selected_features = st.multiselect("Please Select features",options=pd.Series(X.columns))
                st.write(selected_features)
                submitted = st.form_submit_button("Add Features",on_click=None)
            try:

                model_train_test_results(X[selected_features],y,model,tag=problem_type)
            except:
                st.warning("Please select at least one Feature")
            #if submitted:
              

if choice == "Deploy":
    st.header("Deploy",divider="rainbow")
    Data_name = st.text_input(
        "Trained Data",
        "myData",
        key="placeholder Data",)
    Interface_name = st.text_input(
        "Interface",
        "AYO IP123",
        key="placeholder Interface",)
    st.button("Submit")
