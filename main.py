import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import joblib
from prediction import predict_risk
from PIL import Image


def main():
    # load the image
    im = Image.open('icon.png')
    st.set_page_config(page_title='Heart Attack Risk Prediction', page_icon=im)

    # remove the default menu
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title('Predicting the risk of a heart attack')
    st.write(
        'This is a simple web app that predicts the risk of heart attack based on a few parameters.'
    )

    # Sex
    sex = st.radio('Sex', ['Female', 'Male'], horizontal=True)

    # General Health
    general_health = st.selectbox(
        'General Health', ['Very good', 'Good', 'Excellent', 'Fair', 'Poor']
    )

    # Last Checkup Time
    last_checkup_time = st.selectbox(
        'Last Checkup Time',
        [
            'Within past year (anytime less than 12 months ago)',
            'Within past 2 years (1 year but less than 2 years ago)',
            'Within past 5 years (2 years but less than 5 years ago)',
            '5 or more years ago',
        ],
    )

    # Physical Activities
    physical_activities = st.radio(
        'Physical Activities', ['Yes', 'No'], horizontal=True
    )

    # SleepHours
    sleep_hours = st.slider('Sleep Hours', 0, 24)

    # RemovedTeeth
    removed_teeth = st.selectbox(
        'Removed Teeth',
        [
            'None of them',
            '1 to 5',
            '6 or more, but not all',
            'All',
        ],
    )

    # HadAngina
    had_angina = st.radio('Had Angina', ['No', 'Yes'], horizontal=True)

    # HadArthritis
    had_arthritis = st.radio('Had Arthritis', ['No', 'Yes'], horizontal=True)

    # HadDiabetes
    had_diabetes = st.radio('Had Diabetes', ['No', 'Yes'], horizontal=True)

    # DifficultyWalking
    difficulty_walking = st.radio('Difficulty Walking', ['No', 'Yes'], horizontal=True)

    # SmokerStatus
    smoker_status = st.selectbox(
        'Smoker Status',
        [
            'Never Smoked',
            'Former Smoker',
            'Current smoker - now smokes every day',
            'Current smoker - now smokes some days',
        ],
    )

    # ECigaretteUsage
    e_cigarette_usage = st.selectbox(
        'E-Cigarette Usage',
        [
            'Never used e-cigarettes in my entire life',
            'Not at all (right now)',
            'Use them some days',
            'Use them every day',
        ],
    )

    # ChestScan
    chest_scan = st.radio('Chest Scan', ['No', 'Yes'], horizontal=True)

    # RaceEthnicityCategory
    race = st.selectbox(
        'RaceEthnicityCategory',
        [
            'White only, Non-Hispanic',
            'Hispanic',
            'Black only, Non-Hispanic',
            'Other race only, Non-Hispanic',
            'Multiracial, Non-Hispanic',
        ],
    )

    # Example of handling input for numerical data
    age_category = st.selectbox(
        'Age Category',
        [
            'Age 18 to 24',
            'Age 25 to 29',
            'Age 30 to 34',
            'Age 35 to 39',
            'Age 40 to 44',
            'Age 45 to 49',
            'Age 50 to 54',
            'Age 55 to 59',
            'Age 60 to 64',
            'Age 65 to 69',
            'Age 70 to 74',
            'Age 75 to 79',
            'Age 80 or older',
        ],
    )

    # AlcoholDrinkers
    alcohol_drinkers = st.radio('Alcohol Drinkers', ['No', 'Yes'], horizontal=True)

    # HIVTesting
    hiv_testing = st.radio('HIV Testing', ['No', 'Yes'], horizontal=True)

    # FluVaxLast12
    flu_vax_last_12 = st.radio('Flu Vax Last 12', ['No', 'Yes'], horizontal=True)

    # PneumoVaxEver
    pneumo_vax_ever = st.radio('Pneumo Vax Ever', ['No', 'Yes'], horizontal=True)

    # TetanusLast10Tdap
    tetanus_last_10_tdap = st.selectbox(
        'Tetanus Last 10 Tdap',
        [
            'No, did not receive any tetanus shot in the past 10 years',
            'Yes, received tetanus shot but not sure what type',
            'Yes, received Tdap',
            'Yes, received tetanus shot, but not Tdap',
        ],
    )

    # CovidPos
    covid_pos = st.radio('Covid Pos', ['No', 'Yes'], horizontal=True)

    # When all inputs are collected
    if st.button('Predict Risk'):
        data = pd.DataFrame(
            {
                'Sex': [sex],
                'GeneralHealth': [general_health],
                'LastCheckupTime': [last_checkup_time],
                'PhysicalActivities': [physical_activities],
                'SleepHours': [sleep_hours],
                'RemovedTeeth': [removed_teeth],
                'HadAngina': [had_angina],
                'HadArthritis': [had_arthritis],
                'HadDiabetes': [had_diabetes],
                'DifficultyWalking': [difficulty_walking],
                'SmokerStatus': [smoker_status],
                'ECigaretteUsage': [e_cigarette_usage],
                'ChestScan': [chest_scan],
                'RaceEthnicityCategory': [race],
                'AgeCategory': [age_category],
                'AlcoholDrinkers': [alcohol_drinkers],
                'HIVTesting': [hiv_testing],
                'FluVaxLast12': [flu_vax_last_12],
                'PneumoVaxEver': [pneumo_vax_ever],
                'TetanusLast10Tdap': [tetanus_last_10_tdap],
                'CovidPos': [covid_pos],
            }
        )
        result = predict_risk(data)
        st.write('The predicted risk of heart attack is:', result[0])


if __name__ == '__main__':
    main()
