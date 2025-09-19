
````markdown
# RealEstate_App

A web application to predict real estate sold prices using machine learning. Provides models for Houses/Townhouses and Apartments, with preprocessing, feature engineering, and deployment via a Flask interface.

---

## Project Structure

| File / Folder | Description |
|---------------|-------------|
| `app.py` | Flask web app for serving predictions. |
| `model_trainer.py` | Script to preprocess data, train models (XGBoost) and save model artifacts. |
| `realestate.csv` | Original dataset used for training. |
| `requirements.txt` | Python packages/dependencies required. |
| `house_townhouse_model.pkl` / `apartment_model.pkl` | Trained model objects for House/Townhouse and Apartment predictions. |
| `house_townhouse_scaler.pkl` / `apartment_scaler.pkl` | Scalers used to standardize numerical features. |
| `house_townhouse_features.pkl` / `apartment_features.pkl` | Feature lists expected by each model. |
| `templates/` and `static/` | Flask templates and static assets (CSS/JS). |

---

## Installation & Setup

1. Clone the repo:

    ```bash
    git clone https://github.com/RaveeshaS123/RealEstate_App.git
    cd RealEstate_App
    ```

2. Create and activate a virtual environment (optional, but recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS / Linux
    venv\Scripts\activate     # Windows
    ```

3. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

---

## Training Models

You can retrain the models using:

```bash
python model_trainer.py
````

This script will:

* Preprocess the `realestate.csv` dataset (handle missing values, feature engineering, encoding, standardization).
* Train two models: one for Houses/Townhouses, another for Apartments.
* Save the model files, feature lists, and scalers (`*.pkl` files).

---

##  Deployment / Usage

After training or using the provided model artifacts, start the Flask app:

```bash
flask run
```

Alternatively, in production / hosting setups:

```bash
gunicorn app:app
```

Use the web UI to input property details (Bedrooms, Bathrooms, Property Type, Location/Suburb, etc.), and receive a predicted sold price.

---

## Prediction Flow

When  request a prediction:

1. Input data is collected via form.
2. Preprocessing is applied:

   * Compute new features (e.g., Property Age).
   * One-hot encode categorical features.
   * Align input features with those used during training.
   * Standardize numerical features using the appropriate scaler.
3. Model (House/Townhouse or Apartment) is selected based on `Property Type`.
4. Model makes prediction and result is returned to user.

---

## Key Features / Insights

* Models trained using **XGBoost Regressor** due to best performance on test data.
* Feature engineering includes correcting “Land Size” values for Apartments when listed for entire complex (unrealistic high values) using bedroom-based defaults.
* Used metrics like MAE, RMSE, and R² to compare models (Linear, Random Forest, XGBoost).
* SHAP-based interpretation shows features like `Property Type_House`, `Bedrooms`, `Adjusted Land Size`, etc., are among most influential.

---

## References

* Data source: `realestate.csv` (dataset used in training).
* Victorian apartment design standards and regional housing reports (used to justify default land sizes for apartments).
* SHAP for model interpretability.

---


```
