import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# App title and description with HTML
st.set_page_config(page_title="Car Price Prediction App", page_icon="🚗", layout="centered")
st.markdown("""
    <div style="background-color:#4CAF50;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">🚗 Car Price Prediction App</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <p style="font-size:18px;text-align:center;">
    Welcome to the Car Price Prediction App! Enter the car details below, and we'll predict the estimated selling price based on the features you provide.
    </p>
""", unsafe_allow_html=True)

# Sidebar with inputs using HTML and CSS for styling
st.sidebar.markdown("""
    <div style="background-color:#f0f2f6;padding:10px;border-radius:10px">
        <h2 style="color:#4CAF50;text-align:center;">Input Car Details</h2>
        <p style="text-align:center;">Provide the details of the car you want to evaluate:</p>
    </div>
""", unsafe_allow_html=True)

# Input fields in the sidebar
Year = st.sidebar.number_input('Year of Purchase', 2000, 2023, step=1)
Present_Price = st.sidebar.number_input('Present Price (in lakhs)', 0.0, 50.0, step=0.1)
Kms_Driven = st.sidebar.number_input('Kilometers Driven', 0, 500000, step=1000)
Fuel_Type = st.sidebar.selectbox('Fuel Type', ('Petrol', 'Diesel', 'CNG'))
Seller_Type = st.sidebar.selectbox('Seller Type', ('Dealer', 'Individual'))
Transmission = st.sidebar.selectbox('Transmission Type', ('Manual', 'Automatic'))
Owner = st.sidebar.selectbox('Owner Type', ('First', 'Second', 'Third'))

# Process the categorical inputs to match the model training
Fuel_Type_encoded = 0 if Fuel_Type == 'Petrol' else 1 if Fuel_Type == 'Diesel' else 2
Seller_Type_encoded = 0 if Seller_Type == 'Dealer' else 1
Transmission_encoded = 0 if Transmission == 'Manual' else 1
Owner_encoded = 0 if Owner == 'First' else 1 if Owner == 'Second' else 2

# Convert the year to the car's age (assuming the dataset was trained with car age instead of year)
car_age = 2024 - Year  # Assuming the model was trained up to 2024

# Create a feature vector (ensure the order of columns is correct based on training data)
features = np.array([[car_age, Present_Price, Kms_Driven, Fuel_Type_encoded, Seller_Type_encoded, Transmission_encoded, Owner_encoded]])

# Scale the features (ensure scaler was used for the correct features during training)
scaled_features = scaler.transform(features)

# Predict the price and display the result with styled output
if st.sidebar.button('Predict Price'):
    predicted_price = model.predict(scaled_features)
    
    # Handle negative predictions
    predicted_price = max(predicted_price[0], 0)  # Set to 0 if negative

    st.markdown(f"""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-top:20px;">
            <h3 style="color:#4CAF50;text-align:center;">Prediction Result</h3>
            <p style="font-size:18px;text-align:center;">
                Based on the details you provided, the <strong>estimated selling price</strong> of the car is:
            </p>
            <h2 style="color:#4CAF50;text-align:center;">₹{predicted_price:.2f} lakhs</h2>
        </div>
    """, unsafe_allow_html=True)


# Footer with some styling
st.markdown("""
    <hr style="border:1px solid #f0f2f6;margin:40px 0;">
    <div style="text-align:center;">
        <p style="color:gray;">Created by <strong>[Uwais]</strong> | Data Science & Machine Learning Enthusiast</p>
    </div>
""", unsafe_allow_html=True)

# Add some custom CSS for buttons and overall layout
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        margin-top: 10px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stSidebar .stSelectbox, .stSidebar .stNumberInput {
        background-color: #fff;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)


