import pandas as pd 
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
import plotly
import warnings 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt



warnings.filterwarnings("ignore")
data = pd.read_csv('Dataset.csv')
continuous_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
categorical_cols = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
X = data.drop(columns=['output']) #Features
y = data['output'] 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

st.set_page_config(page_title='Heart', layout='wide', page_icon="❤️")
st.title("❤️ Heart Risk Analysis ❤️")

with st.sidebar:
    selected = option_menu(menu_title="Main Menu",options=["Introduction","Data", "Visualization", "KNN Model","Naive Bayes Model","Conclusion","Predictions"],icons=["newspaper","coin","basket","receipt","newspaper","receipt","check-circle"]
                       ,menu_icon="house",default_index=0)


       
if selected == "Introduction":
    st.divider()
    st.subheader("Objective")
    st.write("**In order to predict whether or not a person will be diagnosed with heart disease**")
    
    st.sidebar.divider()
    st.sidebar.info("The Introduction tab provides an overview of the Heart Risk Analysis dashboard.It includes key objectives and a detailed description of the dataset used in the analysis.")
    
    st.markdown("""
                ### Data Description:

                - **age:** Age of the patient
                - **sex:** Sex of the patient (1 = Male, 0 = Female)
                - **cp:** Type of chest pain experienced by the patient
                    - 0 = asymptomatic
                    - 1 = non-anginal
                    - 2 = non-typical
                    - 3 = typical
                - **trtbps:** Resting blood pressure of the patient (in mm Hg)
                - **chol:** Cholesterol level of the patient (in mg/dl)
                - **fbs:** Fasting blood sugar level of the patient
                    - 1 = fasting blood sugar > 120 mg/dl
                    - 0 = fasting blood sugar < 120 mg/dl
                - **restecg:** Resting ECG results of the patient
                    - 0 = normal
                    - 1 = ST-T wave abnormality
                    - 2 = left ventricular hypertrophy
                - **thalachh:** Maximum heart rate of the patient during exercise (in beats per minute)
                - **exng:** Exercise-induced angina (1 = yes, 0 = no)
                - **oldpeak:** ST depression induced by exercise relative to rest
                - **slp:** Slope of the peak exercise ST segment
                    - 1 = upsloping
                    - 2 = flat
                    - 3 = downsloping
                - **caa:** Number of major vessels (0-3)
                - **thall:** Thalassemia, a blood disorder
                    - 1 = normal
                    - 2 = fixed
                    - 3 = reversible
                - **output:** Presence or absence of heart disease
                    - 1 = has heart disease
                    - 0 = does not have heart disease
                """)
if selected == "Data":
    st.subheader("Description")
    col1, col2=st.columns([1,1])
    with col1:
        if st.button("Basic Statistics"):
            st.write(data.describe())
    with col2:
        if st.button("Data Types"):
             st.write(data.dtypes)
    st.divider()
    col3,col4=st.columns([1,1])
    with col3:
        if st.button("Missing Values"):
            st.write(data.isnull().sum())
    with col4:
        if st.button("Correlation Matrix"):
            df_corr = data[continuous_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(df_corr,annot=True)
            st.pyplot(fig)

if selected == "Visualization":
    if st.button("Boxplot"):
        fig, axes = plt.subplots(len(continuous_cols), 1, figsize=(10, 5 * len(continuous_cols)))  # Create subplots for each column
        for i, col in enumerate(continuous_cols):
            sns.boxplot(x=data[col], ax=axes[i], palette="Set2")
            axes[i].set_title(f'Boxplot of {col.capitalize()}')
            axes[i].set_xlabel(col.capitalize())
            axes[i].set_ylabel('Value')
        
        st.pyplot(fig)  # Display the plot in Streamlit
    if st.button("Countplots"):
        num_rows = (len(categorical_cols) - 1) // 2 + 1  # Calculate number of rows
        num_cols = 3  # Set number of columns for subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))  # Create the subplots grid
    
    # Flatten the axes for easy iteration
        axes = axes.flatten()

    # Iterate over categorical columns and create countplots
        for i, col in enumerate(categorical_cols):
            sns.countplot(x=col, data=data, palette="RdYlGn", ax=axes[i], edgecolor="black")
            axes[i].set_title(f'Countplot of {col.capitalize()}')
            axes[i].set_xlabel(col.capitalize())
            axes[i].set_ylabel('Count')
            
    # Remove any empty subplots (if the number of categorical columns is less than the total subplots)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        st.pyplot(fig)  # Display the figure in Streamlit
    if st.button("output variable"):
        fig, axes = plt.subplots()
        sns.countplot(x="output", data=data, palette="Greens",edgecolor="black")
        for container in axes.containers:
            axes.bar_label(container, fmt='%d', label_type='edge', fontsize=10)
        st.pyplot(fig)

if selected== "KNN Model":
    
    col5,col6=st.columns([1,1])
    with col5:
        if st.button("Hyper Parameter Tuning"):
            param_grid = {'n_neighbors': range(1, 21)}
        
        # Initialize KNN classifier
            knn = KNeighborsClassifier()

        # Perform grid search with 5-fold cross-validation
            grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
        
        # Fit grid search to the training data
            grid_search.fit(X_train, y_train)
        
        # Get the best k value
            best_k = grid_search.best_params_['n_neighbors']
        
        # Display the best k value in the Streamlit app
            st.success(f"Best n-neighbor (k) = {best_k}")

        # Optional: Display the best score from the grid search
            best_score = grid_search.best_score_
            st.write(f"Paramter which we provided are n_neighbors {param_grid['n_neighbors']}")
            st.write(f"Best accuracy score: {best_score:.4f}")
    with col6:
        if st.button("Fit model"):
            knn_model = KNeighborsClassifier(n_neighbors=18)
            knn_model.fit(X_train, y_train)
            pred=knn_model.predict(X_train)
            Acc=accuracy_score(y_train,pred)
            st.write(f"Accuracy on the training Data is **{Acc:.4f}**")
    st.divider()
    if st.button("Predictions"):
        knn_model = KNeighborsClassifier(n_neighbors=18)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy on the Testing Data is **{accuracy:.4f}**")
        
        target_names = ["Presence of Heart Disease", "Absence of Heart Disease"]
        st.text(classification_report(y_test, y_pred, target_names=target_names))

    st.info("K-Nearest Neighbors (KNN) Basic Idea: KNN is like finding friends in a crowd. To decide which group you belong to, you look at your closest neighbors.")
    st.write('''You have a set of data points (like different types of fruits). When you want to classify a new point (like a new fruit), KNN looks at the ‘K’ closest points in the data. It then sees which group (or category) these neighbors belong to and picks the majority category. Choosing K: The number 'K' is important. A small K can be noisy (too influenced by outliers), while a large K might smooth out important distinctions.''')

if selected== "Naive Bayes Model":
    
    if st.button("Fit model"):
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        pred1=nb_model.predict(X_train)
        Acc2=accuracy_score(y_train,pred1)
        st.write(f"Accuracy on the training Data is **{Acc2:.4f}**")
    st.divider()
    if st.button("Prediction"):
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        y_pred = nb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy on the training Data is **{accuracy:.4f}**")
        target_names = ["Presence of Heart Disease", "Absence of Heart Disease"]
        st.text(classification_report(y_test, y_pred, target_names=target_names))
    st.info("Naive Bayes is like making predictions based on past experiences. You assume certain features (or traits) independently contribute to the outcome.")
    st.write('''You have a set of features (like color, size, etc.) for each category (like types of fruits).For a new item, Naive Bayes calculates the probability of it belonging to each category based on those features.It uses Bayes’ theorem to update these probabilities, assuming that the features are independent (this is the "naive" part).Why It’s Useful: It’s fast and works well with large datasets, especially when the independence assumption is somewhat true.In summary, KNN looks at your closest examples to classify new data, while Naive Bayes uses probabilities based on features to make predictions''')

if selected == "Conclusion":
    col7,col8=st.columns([1,1])
    with col7:
        st.metric(label="Accuracy Of KNN Model",value="90%")
    with col8:
        st.metric(label="Accuracy of Naive Bayes",value="88%")
    st.subheader("KNN Model")
    st.write("The precision of 0.90 for presence of heart disease indicates that among all instances predicted as having heart disease, 90% were correctly classified. The recall of 0.90 implies that the model correctly identified 90% of all actual instances of heart disease. The F1-score of 0.90 is the harmonic mean of precision and recall, providing a balanced measure of the classifier's performance. With an accuracy of 0.90, the model correctly classified 90% of the instances in the dataset")

    st.subheader("Naive Bayes Model")
    st.write("The precision of 0.87 for presence of heart disease indicates that among all instances predicted as having heart disease, 87% were correctly classified. The recall of 0.90 implies that the model correctly identified 90% of all actual instances of heart disease. The F1-score of 0.88 is the harmonic mean of precision and recall, providing a balanced measure of the classifier's performance. With an accuracy of 0.89, the model correctly classified 89% of the instances in the dataset")
    st.divider()
    st.subheader("Thank you")
    
if selected == "Predictions":
    st.subheader("Predict Patient Risk for Heart Disease")
    
    # User input for Naive Bayes prediction
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", options=[0, 1])
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
    trtbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0)
    chol = st.number_input("Cholesterol Level (mg/dl)", min_value=0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
    thalachh = st.number_input("Max Heart Rate", min_value=0)
    exng = st.selectbox("Exercise Induced Angina", options=[0, 1])
    oldpeak = st.number_input("Oldpeak", min_value=0.0)
    slp = st.selectbox("Slope of Peak Exercise ST Segment", options=[0,1, 2, 3])
    caa = st.number_input("Number of Major Vessels (0-3)", min_value=1, max_value=3)
    thall = st.selectbox("Thalassemia", options=[1, 2, 3])

    if st.button("Predict Naive Bayes"):
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)  # Fit the model
        input_data = pd.DataFrame([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]], 
                                   columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall'])
        prediction = nb_model.predict(input_data)
        if prediction[0]==1:
            st.write("More likely to have heart dieases")
        else:
            st.write(" No Heart Disease")

