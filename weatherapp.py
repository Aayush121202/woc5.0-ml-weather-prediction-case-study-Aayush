#importing the libraries
import numpy as np
import pickle
import pandas as pd
import streamlit as st

#Load model from the disk
model = pickle.load(open("model.pkl", 'rb'))

# Creating the function which will make the prediction of the data the user inputs using the ML model we created
def predict_weather(precipitation, temp_max, temp_min, wind):
    prediction = model.predict([[precipitation, temp_max, temp_min, wind]])
    return prediction

html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Weather Prediction ML App </h2>
    </div>
    """

#Title and Subtitle
st.title("Weather Prediction")
st.write("This model predicts the weather using Multinomial Classification")

# Creating a sidebar header
st.sidebar.header('User Input Parameters')

# Creating a function to get the user input
def get_user_input():
    precipitation = st.sidebar.slider('Precipitation', 0.0, 60.0, 22.5)
    temp_min = st.sidebar.slider('Temp_min', -10.0, 28.0, 15.5)
    temp_max = st.sidebar.slider('Temp_max', 0.0, 50.0, 33.2)
    wind = st.sidebar.slider('Wind Speed', 0.0, 20.0, 5.5)
    user_data = {'Precipitation': precipitation,
                 'Temp_min': temp_min,
                 'Wind Speed': wind,
                 'Temp_max': temp_max}
    features = pd.DataFrame(user_data, index=[0])
    return features

# Storing the user input into a variable
user_input = get_user_input()

# Creating and storing the result
result = predict_weather(user_input['Precipitation'][0], user_input['Temp_max'][0], user_input['Temp_min'][0], user_input['Wind Speed'][0])
# Output the result and an image based on the result
if result == 'sun':
    st.success('The predicted weather is sunny')   
    st.image('sunny.png', width=300)
elif result == 'rain':
    st.success('The predicted weather is rain')   
    st.image('rainy.png', width=300)
elif result == 'drizzle':
    st.success('The predicted weather is drizzle')   
    st.image('drizzle.png', width=300)
elif result == 'snow':
    st.success('The predicted weather is snow')   
    st.image('snow.png',width=300)
elif result == 'fog':
    st.success('The predicted weather is fog')   
    st.image('fog.png',width=300)
if st.button("About"):
    st.text("Built with Streamlit")
    st.text("By Aayush Patel")











