import numpy as np
import pickle
from pickle import load
import pandas as pd
import streamlit as st
import category_encoders
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://wallpapercave.com/wp/wp5055258.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()


def pred(df1):
    loaded_encoder = load(open('binary_encoder.pkl', 'rb'))
    loaded_rf_regressor = load(open('model.pkl', 'rb'))
    loaded_sc = load(open('stdscaler.pkl', 'rb'))
    var_transform = ['avg_cost_price', 'vehicle_age', 'km_driven', 'mileage', 'max_power', 'seats']
    df1 = loaded_encoder.transform(df1)
    df1[var_transform] = loaded_sc.transform(df1[var_transform])
    y = loaded_rf_regressor.predict(df1)
    selling_price = round(float(y))
    return selling_price


def main():
    df = pd.read_csv('cleaned_data.csv')
    brand_list = df['brand'].unique()
    st.title("PREDICT YOUR CAR PRICE")
    brand_option = st.selectbox('Please select the Brand', brand_list)
    model_list = df[df['brand'] == brand_option]['model'].unique()
    model_option = st.selectbox('Please select the Model', model_list)
    seller_type_list = df['seller_type'].unique()
    fuel_type_list = df['fuel_type'].unique()
    transmission_type_list = df['transmission_type'].unique()
    seats_list = df['seats'].unique()
    seller_type_option = st.selectbox('Please select the Seller Type', seller_type_list)
    fuel_type_option = st.selectbox('Please select the Fuel Type', fuel_type_list)
    transmission_type_option = st.selectbox('Please select the Transmission Type', transmission_type_list)
    seats_option = st.selectbox('Please select the number of Seats', seats_list)
    avg_cost_price = st.number_input("Enter the cost price in Rs.")
    vehicle_age = st.number_input("Enter the age of the Car")
    km_driven = st.number_input("Enter km driven")
    mileage = st.number_input("Enter Current Mileage of the Car in KmpL")
    max_power = st.number_input("Enter the maximum power of the Car Engine in bhp@rpm")
    input_data = {'brand': [brand_option], 'model': [model_option], 'seller_type': [seller_type_option],
                  'fuel_type': [fuel_type_option],'transmission_type': [transmission_type_option],
                  'avg_cost_price': [avg_cost_price],'vehicle_age': [vehicle_age],
                  'km_driven': [km_driven], 'mileage': [mileage],
                  'max_power': [max_power], 'seats': [seats_option]}
    df1 = pd.DataFrame(input_data)
    if st.button("Predict My Car Price"):
        if (avg_cost_price<=0 or mileage<=0 or km_driven <= 0 or max_power <= 0 ):
            st.error('Cost Price, km driven, Mileage,maximum power should be greater than 0 ', icon="🚨")
        else:
            selling_price = pred(df1)
            st.success("Your Car Should be Sold at about Rs. {}".format(selling_price))


if __name__ == '__main__':
    main()
