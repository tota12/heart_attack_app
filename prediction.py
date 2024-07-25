import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


def predict_risk(data):
    # Encoding categorical variables
    le = LabelEncoder()

    categorical_cols = [
        'Sex',
        'GeneralHealth',
        'LastCheckupTime',
        'PhysicalActivities',
        'RemovedTeeth',
        'HadAngina',
        'HadArthritis',
        'HadDiabetes',
        'DifficultyWalking',
        'SmokerStatus',
        'ECigaretteUsage',
        'ChestScan',
        'RaceEthnicityCategory',
        'AgeCategory',
        'AlcoholDrinkers',
        'HIVTesting',
        'FluVaxLast12',
        'PneumoVaxEver',
        'TetanusLast10Tdap',
        'CovidPos',
    ]

    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    # Load the scaler
    scaler = joblib.load('scaler.pkl')

    # Scale the data
    data = scaler.transform(data)

    # Load the model
    model = joblib.load('rf_model1.pkl')

    return model.predict(data)
