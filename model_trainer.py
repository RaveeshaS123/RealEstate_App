import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib

def preprocess_data(data, property_types_to_filter):
    """
    Cleans and preprocesses the real estate data for a specific group of property types.
    """
    df_cleaned = data.copy()

    # Filter for the specific property type(s)
    df_cleaned = df_cleaned[df_cleaned['Property Type'].isin(property_types_to_filter)].copy()

    # Create 'Property Age' feature
    df_cleaned['Sale Date'] = pd.to_datetime(df_cleaned['Sale Date'], format='%d-%b-%y', errors='coerce')
    df_cleaned['Build/Sold Year'] = df_cleaned['Year Built'].fillna(df_cleaned['First Sold Year'])
    
    # Use a fallback for current_year if 'Sale Date' is all NaT
    if df_cleaned['Sale Date'].isnull().all():
        current_year = datetime.now().year
    else:
        current_year = df_cleaned['Sale Date'].dt.year

    df_cleaned['Property Age'] = current_year - df_cleaned['Build/Sold Year']
    # Impute missing 'Property Age' values suburb-wise
    df_cleaned['Property Age'] = df_cleaned.groupby('Location/Suburb')['Property Age'].transform(lambda x: x.fillna(x.median()))
    
    # Impute numerical columns with the median
    numerical_cols = ['Bedrooms', 'Bathrooms', 'Parking', 
                      'No of Primary Schools (within 1 km)', 'No of Secondary Schools (within 1 km)']
    for col in numerical_cols:
        if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
            median_val = df_cleaned[col].median()
            df_cleaned[col].fillna(median_val, inplace=True)

    # Impute categorical columns with a new category 'Missing'
    categorical_cols = ['Property Type', 'Location/Suburb', 'Agent']
    for col in categorical_cols:
        if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col].fillna('Missing', inplace=True)
            
    # Handle land size based on property type
    if 'Apartment' in property_types_to_filter:
        # This block only runs for apartments
        df_cleaned['Adjusted Land Size(m^2)'] = df_cleaned['Land Size(m^2)']
        apartment_defaults = {1: 50, 2: 65, 3: 85, 4: 110}
        
        for bedrooms, default_size in apartment_defaults.items():
            mask = (df_cleaned['Bedrooms'] == bedrooms) & (df_cleaned['Adjusted Land Size(m^2)'].isnull() | (df_cleaned['Adjusted Land Size(m^2)'] > 1000))
            df_cleaned.loc[mask, 'Adjusted Land Size(m^2)'] = default_size
            
        median_adjusted_land_size = df_cleaned['Adjusted Land Size(m^2)'].median()
        df_cleaned['Adjusted Land Size(m^2)'].fillna(median_adjusted_land_size, inplace=True)
        
        df_cleaned.drop(columns=['Land Size(m^2)'], inplace=True)
    else:
        # This block runs for houses and townhouses
        median_land_size = df_cleaned['Land Size(m^2)'].median()
        df_cleaned['Land Size(m^2)'].fillna(median_land_size, inplace=True)
        if 'Adjusted Land Size(m^2)' in df_cleaned.columns:
            df_cleaned.drop(columns=['Adjusted Land Size(m^2)'], inplace=True)

    # Drop original columns
    df_cleaned.drop(columns=['Year Built', 'First Sold Year', 'Build/Sold Year', 'Sale Date'], inplace=True)
    
    return df_cleaned

def train_and_save_model(df, model_name):
    """
    Trains a model for a specific group of property types and saves the artifacts.
    """
    print(f"Training model for {model_name}s...")

    # Numerical features list based on model_name
    if 'apartment' in model_name.lower():
        numerical_features = ['Bedrooms', 'Bathrooms', 'Parking', 'Adjusted Land Size(m^2)',
                              'No of Primary Schools (within 1 km)', 'No of Secondary Schools (within 1 km)',
                              'Property Age']
    else:
        numerical_features = ['Bedrooms', 'Bathrooms', 'Parking', 'Land Size(m^2)',
                              'No of Primary Schools (within 1 km)', 'No of Secondary Schools (within 1 km)',
                              'Property Age']

    # Separate features and target before encoding
    target = 'Sold Price'
    X = df.drop(columns=[target])
    y = df[target]

    # Train-test split (train + validation and test)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Categorical columns for encoding
    categorical_cols = ['Property Type', 'Location/Suburb', 'Agent']

    # One-hot encode categorical columns separately
    X_trainval = pd.get_dummies(X_trainval, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

    # Align columns of test set to train set
    X_test = X_test.reindex(columns=X_trainval.columns, fill_value=0)

    # Convert boolean to int
    X_trainval = X_trainval.astype(int)
    X_test = X_test.astype(int)

    # Initialize scaler and fit only on trainval numerical features
    scaler = StandardScaler()
    X_trainval[numerical_features] = scaler.fit_transform(X_trainval[numerical_features])

    # Transform test numerical features using the same scaler
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # Train the XGBoost Regressor on trainval
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                         random_state=42, objective='reg:squarederror')
    model.fit(X_trainval, y_trainval)

    # Save model, scaler, and features list
    model_features = X_trainval.columns.tolist()
    joblib.dump(model, f'{model_name}_model.pkl')
    joblib.dump(scaler, f'{model_name}_scaler.pkl')
    joblib.dump(model_features, f'{model_name}_features.pkl')

    print(f"Model, scaler, and features for {model_name}s have been trained and saved successfully.")



def main():
    """
    Main function to train and save both models.
    """
    print("Loading data...")
    try:
        df = pd.read_csv('realestate.csv')
    except FileNotFoundError:
        print("Error: The file 'realestate.csv' was not found.")
        return

    # Preprocess and train for Houses and Townhouses
    df_house_townhouse = preprocess_data(df, ['House', 'Townhouse'])
    train_and_save_model(df_house_townhouse, 'house_townhouse')

    # Preprocess and train for Apartments
    df_apartments = preprocess_data(df, ['Apartment'])
    train_and_save_model(df_apartments, 'apartment')

if __name__ == '__main__':
    main()