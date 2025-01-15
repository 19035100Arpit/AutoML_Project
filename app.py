import streamlit as st
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import os
import pickle
from pycaret.classification import models

# Initialize session state
if 'user_credentials' not in st.session_state:
    st.session_state['user_credentials'] = {}

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "login"

if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'file_name' not in st.session_state:
    st.session_state['file_name'] = None

if 'removed_features' not in st.session_state:
    st.session_state['removed_features'] = []

if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None

# Load user credentials from file
def load_credentials():
    if os.path.exists('credentials.pkl'):
        with open('credentials.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return {}

# Save user credentials to file
def save_credentials(credentials):
    with open('credentials.pkl', 'wb') as f:
        pickle.dump(credentials, f)

# Function to switch pages
def switch_page(page):
    st.session_state['current_page'] = page

# Sign-In page
def sign_in():
    st.title("Sign-In")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Sign-In"):
        if new_username and new_password:
            credentials = load_credentials()
            if new_username in credentials:
                st.warning("Username already exists. Please choose a different username.")
            else:
                credentials[new_username] = new_password
                save_credentials(credentials)
                st.success("Sign-In successful! You can now login.")
                switch_page("login")
        else:
            st.error("Please enter a username and password")

# Login page
def login():
    st.title("Log-In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        credentials = load_credentials()
        if username in credentials and credentials[username] == password:
            st.session_state['logged_in'] = True
            st.success("Login successful!")
            switch_page("main")
        else:
            st.error("Invalid username or password")

# Main application page
def main_app():
    # Create the datasets directory if it doesn't exist
    os.makedirs('datasets', exist_ok=True)

    # Load dataframe from session state or file if available
    if st.session_state['df'] is None and os.path.exists('./datasets/dataset.csv'):
        st.session_state['df'] = pd.read_csv('./datasets/dataset.csv', index_col=None)
        st.session_state['df_original'] = st.session_state['df'].copy()  # Keep original dataframe
        st.session_state['file_name'] = 'dataset.csv'

    df = st.session_state['df']
    df_original = st.session_state['df_original']
    file_name = st.session_state['file_name']
    removed_features = st.session_state['removed_features']

    with st.sidebar:
        st.title("Innovator's AutoML Project")
        st.title("Powered By-")
        st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQGqFH5N6bSl4dY4P9TGfHiyTqhQuo4X5Wy1A&s')
        
        choice = st.radio("Navigation", ["Data Ingestion", "Exploratory Data Analysis", "Data Transformation", "Modelling", "Download"])
        
    if choice == "Data Ingestion":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv('./datasets/dataset.csv', index=None)
            st.session_state['df'] = df  # Store dataframe in session state
            st.session_state['df_original'] = df.copy()  # Keep original dataframe
            st.session_state['file_name'] = file.name  # Store file name in session state
            st.success(f"Dataset {file.name} uploaded successfully!")
            st.dataframe(df)

        # Display the uploaded file name if it exists
        if file_name:
            st.write(f"Uploaded file: {file_name}")
            if df is not None:
                st.write(f"Data Dimensions : ", df.shape)
                #st.write(f"Number of rows: {df.shape[0]}")
                #st.write(f"Number of columns: {df.shape[1]}")

    if choice == "Exploratory Data Analysis":
        if df is not None:
            st.title("Exploratory Data Analysis")
            profile_df = df.profile_report()
            st_profile_report(profile_df)
        else:
            st.warning("Please upload a dataset first.")
            
    if choice == "Data Transformation":
        if df is not None:
            st.title("Data Transformation")
            # Multi-select for columns to remove
            selected_columns = st.multiselect("Select columns to ignore", df.columns)
            if selected_columns:
                st.session_state['removed_features'].extend(selected_columns)
                st.session_state['df'] = st.session_state['df'].drop(columns=selected_columns)
                st.success(f"You have selected {len(selected_columns)} column(s) to ignore: {', '.join(selected_columns)}")
                df = st.session_state['df']
                st.write("Remaining columns:", df.columns.tolist())
                st.write("Removed columns:", st.session_state['removed_features'])
            else:
                st.write("No columns selected")

            # Dropdown to add back removed features
            if st.session_state['removed_features']:
                add_feature = st.selectbox("Select a column to add back", st.session_state['removed_features'])
                if st.button("Add Feature"):
                    st.session_state['removed_features'].remove(add_feature)
                    st.session_state['df'][add_feature] = st.session_state['df_original'][add_feature]
                    st.write("Remaining columns:", st.session_state['df'].columns.tolist())
                    st.write("Removed columns:", st.session_state['removed_features'])
            st.write("Final Shape:" , df.shape)
        else:
            st.warning("Please upload a dataset first.")
            
    if choice == "Modelling":
        if df is not None:
            st.title("Auto Train Model")
            
            # Update dataframe based on removed features
            df = st.session_state['df']  # Ensure df reflects the most recent state
            
            chosen_target = st.selectbox('Choose the Target Column:', df.columns)
            train_size = st.number_input('Enter the Training Size:', min_value=0.0, max_value=1.0, value=0.7, step=0.05, format="%.2f")
            test_size = 1 - train_size
            formatted_test_size = f"{test_size:.2f}"
            st.write("Testing Size : ", formatted_test_size)
            #mdl = st.selectbox('Choose Modelling Type:', ['Classification', 'Regression'])
            
            if st.button('Run Modelling'):
                st.title("Classification Models: ")
                
                
                from pycaret.classification import setup, compare_models, pull, save_model
                
                # Make sure df is up-to-date
                df = st.session_state['df']  
                setup_df = setup(df, target=chosen_target,ignore_features=st.session_state['removed_features'], train_size=train_size)
                st.dataframe(pull())
                best_model_classification = compare_models()
                st.dataframe(pull())
                save_model(best_model_classification, 'best_model_classification')
                   
                st.title("Regression Models: ") 
                   
                from pycaret.regression import setup, compare_models, pull, save_model
                
                # Make sure df is up-to-date
                df = st.session_state['df']
                setup_df = setup(df, target=chosen_target,ignore_features=st.session_state['removed_features'], train_size=train_size)
                st.dataframe(pull())
                best_model_regression = compare_models()
                st.dataframe(pull())
                save_model(best_model_regression, 'best_model_regression')
                
        else:
            st.warning("Please upload a dataset first.")
        
    if choice == "Download":
        st.title("Hold Learnings: ")
        #Classification Model Download
        if os.path.exists('best_model_classification.pkl'):
            with open('best_model_classification.pkl', 'rb') as f:
                st.download_button('Download Best Classification Model', f, file_name="best_model_classification.pkl")
        else:
            st.warning("Please build a model first.")
            
        #Regression Model Download 
        if os.path.exists('best_model_regression.pkl'):
            with open('best_model_regression.pkl', 'rb') as f:
                st.download_button('Download Best Regression Model', f, file_name="best_model_regression.pkl")
        else:
            st.warning("Please build a model first.")
            
            
        st.title("Tune A Classification Model : ")
        classification_models = models()
        classification_models_list = classification_models.index.tolist()
        #st.write(classification_models)
        selected_Model = st.multiselect("Select Model for Tuning", classification_models)
        
    
        
    

# Determine which page to show
if st.session_state['logged_in']:
    main_app()
else:
    # Add a radio button for user to choose Sign-In or Login
    page_choice = st.radio("Choose action", ["Sign In", "Login"])
    if page_choice == "Sign In":
        sign_in()
    else:
        login()
