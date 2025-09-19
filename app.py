import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from datetime import datetime
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# --- Model Loading ---
try:
    house_townhouse_model = joblib.load('house_townhouse_model.pkl')
    house_townhouse_scaler = joblib.load('house_townhouse_scaler.pkl')
    house_townhouse_features = joblib.load('house_townhouse_features.pkl')

    apartment_model = joblib.load('apartment_model.pkl')
    apartment_scaler = joblib.load('apartment_scaler.pkl')
    apartment_features = joblib.load('apartment_features.pkl')

except FileNotFoundError:
    print("Error: One or more model files were not found.")
    print("Please run 'model_trainer.py' first to generate these files.")
    exit()

# Load the original dataframe to get dropdown options
try:
    df = pd.read_csv('realestate.csv')

    # Create 'Property Age' column for the original dataframe to be used for imputation
    df['Sale Date'] = pd.to_datetime(df['Sale Date'], format='%d-%b-%y', errors='coerce')
    df['Build/Sold Year'] = df['Year Built'].fillna(df['First Sold Year'])
    current_year = df['Sale Date'].dt.year.fillna(datetime.now().year)
    df['Property Age'] = current_year - df['Build/Sold Year']
   

    suburbs = sorted(df['Location/Suburb'].dropna().unique().tolist())
    property_types = sorted(df['Property Type'].dropna().unique().tolist())
    agents = sorted(df['Agent'].dropna().unique().tolist())
except FileNotFoundError:
    print("Error: 'realestate.csv' was not found. Cannot populate dropdowns.")
    suburbs = []
    property_types = []
    agents = []

# --- Preprocessing Functions ---

def preprocess_house_townhouse_input(input_data):
    """Preprocesses a single row of input data for the house/townhouse model."""
    try:
        new_df = pd.DataFrame([input_data])
        
        # Build/Sold Year logic
        new_df['Build/Sold Year'] = new_df['Year Built'].fillna(new_df['First Sold Year'])
        current_year = datetime.now().year
        new_df['Property Age'] = current_year - new_df['Build/Sold Year']

        # Impute missing Property Age
        if new_df['Property Age'].isnull().any():
            suburb = new_df['Location/Suburb'].iloc[0]
            median_age = df.groupby('Location/Suburb')['Property Age'].median().get(suburb, df['Property Age'].median())
            new_df['Property Age'].fillna(median_age, inplace=True)

        # Drop unused columns
        new_df.drop(columns=['Year Built', 'First Sold Year', 'Sale Date', 'Build/Sold Year'], errors='ignore', inplace=True)

        # One-hot encode categorical columns
        categorical_cols = ['Property Type', 'Location/Suburb', 'Agent']
        new_df_encoded = pd.get_dummies(new_df, columns=categorical_cols, drop_first=True)

        # Align columns to match training set
        final_df = new_df_encoded.reindex(columns=house_townhouse_features, fill_value=0)

        # Scale numerical features
        numerical_features = ['Bedrooms', 'Bathrooms', 'Parking', 'Land Size(m^2)',
                              'No of Primary Schools (within 1 km)', 'No of Secondary Schools (within 1 km)',
                              'Property Age']
        final_df[numerical_features] = house_townhouse_scaler.transform(final_df[numerical_features])

        return final_df.astype(float)

    except Exception as e:
        print(f"Error during house/townhouse preprocessing: {e}")
        return None
 
def preprocess_apartment_input(input_data):
    """Preprocesses a single row of input data for the apartment model."""
    try:
        new_df = pd.DataFrame([input_data])

        # Build/Sold Year logic
        new_df['Build/Sold Year'] = new_df['Year Built'].fillna(new_df['First Sold Year'])
        current_year = datetime.now().year
        new_df['Property Age'] = current_year - new_df['Build/Sold Year']

        # Impute missing Property Age
        if new_df['Property Age'].isnull().any():
            suburb = new_df['Location/Suburb'].iloc[0]
            median_age = df.groupby('Location/Suburb')['Property Age'].median().get(suburb, df['Property Age'].median())
            new_df['Property Age'].fillna(median_age, inplace=True)

        # Adjusted land size logic
        new_df['Adjusted Land Size(m^2)'] = new_df['Land Size(m^2)']
        apartment_defaults = {1: 50, 2: 65, 3: 85, 4: 110}

        for bedrooms, default_size in apartment_defaults.items():
            mask = (new_df['Bedrooms'] == bedrooms) & (new_df['Adjusted Land Size(m^2)'].isnull() | (new_df['Adjusted Land Size(m^2)'] > 1000))
            new_df.loc[mask, 'Adjusted Land Size(m^2)'] = default_size

        # Impute remaining Adjusted Land Size with overall median (if necessary)
        new_df['Adjusted Land Size(m^2)'].fillna(new_df['Adjusted Land Size(m^2)'].median(), inplace=True)

        new_df.drop(columns=['Land Size(m^2)', 'Year Built', 'First Sold Year', 'Sale Date', 'Build/Sold Year'], errors='ignore', inplace=True)

        # One-hot encode categorical columns
        categorical_cols = ['Property Type', 'Location/Suburb', 'Agent']
        new_df_encoded = pd.get_dummies(new_df, columns=categorical_cols, drop_first=True)

        # Align to training columns
        final_df = new_df_encoded.reindex(columns=apartment_features, fill_value=0)

        # Scale numerical features
        numerical_features = ['Bedrooms', 'Bathrooms', 'Parking', 'Adjusted Land Size(m^2)',
                              'No of Primary Schools (within 1 km)', 'No of Secondary Schools (within 1 km)',
                              'Property Age']
        final_df[numerical_features] = apartment_scaler.transform(final_df[numerical_features])

        return final_df.astype(float)

    except Exception as e:
        print(f"Error during apartment preprocessing: {e}")
        return None


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    if request.method == 'POST':
        try:
            # Safely get form data with fallbacks for potential empty or missing inputs
            bedrooms = request.form.get('bedrooms')
            bathrooms = request.form.get('bathrooms')
            parking = request.form.get('parking')
            land_size = request.form.get('land_size')
            primary_schools = request.form.get('primary_schools')
            secondary_schools = request.form.get('secondary_schools')
            year_built = request.form.get('year_built')
            first_sold_year = request.form.get('first_sold_year')
            sale_date = request.form.get('sale_date')

            input_data = {
                'Property Type': request.form['property_type'],
                'Location/Suburb': request.form['location'],
                'Bedrooms': float(bedrooms) if bedrooms and bedrooms.strip() else np.nan,
                'Bathrooms': float(bathrooms) if bathrooms and bathrooms.strip() else np.nan,
                'Parking': float(parking) if parking and parking.strip() else np.nan,
                'Land Size(m^2)': float(land_size) if land_size and land_size.strip() else np.nan,
                'No of Primary Schools (within 1 km)': float(primary_schools) if primary_schools and primary_schools.strip() else np.nan,
                'No of Secondary Schools (within 1 km)': float(secondary_schools) if secondary_schools and secondary_schools.strip() else np.nan,
                'Year Built': float(year_built) if year_built and year_built.strip() else np.nan,
                'First Sold Year': float(first_sold_year) if first_sold_year and first_sold_year.strip() else np.nan,
                'Sale Date': sale_date if sale_date and sale_date.strip() else datetime.now().strftime('%d-%b-%y'),
                'Agent': request.form['agent']
            }
            
            # Use an if/else to determine which model to use
            if input_data['Property Type'] in ['House', 'Townhouse']:
                processed_input = preprocess_house_townhouse_input(input_data)
                if processed_input is not None and not processed_input.empty:
                    prediction_value = house_townhouse_model.predict(processed_input)[0]
                    prediction_result = f"${prediction_value:,.2f}"
                else:
                    prediction_result = "Failed to process input for House/Townhouse. Check your inputs."
            elif input_data['Property Type'] == 'Apartment':
                processed_input = preprocess_apartment_input(input_data)
                if processed_input is not None and not processed_input.empty:
                    prediction_value = apartment_model.predict(processed_input)[0]
                    prediction_result = f"${prediction_value:,.2f}"
                else:
                    prediction_result = "Failed to process input for Apartment. Check your inputs."
            else:
                prediction_result = "Unsupported property type."

        except (ValueError, KeyError, IndexError) as e:
            print(f"Error during prediction: {e}")
            prediction_result = "Invalid input or processing error. Please check the values."

    return render_template('index.html',
                           prediction=prediction_result,
                           suburbs=suburbs,
                           property_types=property_types,
                           agents=agents)

if __name__ == '__main__':
    app.run(debug=True)