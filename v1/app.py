# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="Smart Crop & NPK Recommender",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------------------
# Load and preprocess dataset
# ----------------------
@st.cache_data
def load_data():
    """Loads, preprocesses the dataset, and fits the encoders."""
    try:
        df = pd.read_csv("dataset.csv")
    except FileNotFoundError:
        st.error("Dataset file 'dataset.csv' not found. Please ensure it's in the same directory.")
        st.stop()

    df.dropna(subset=['soil_texture', 'label'], inplace=True)
    
    soil_encoder = LabelEncoder()
    df['soil_texture_encoded'] = soil_encoder.fit_transform(df['soil_texture'])

    crop_encoder = LabelEncoder()
    df['crop_encoded'] = crop_encoder.fit_transform(df['label'])

    return df, soil_encoder, crop_encoder

df, soil_encoder, crop_encoder = load_data()

# ----------------------
# Train the Random Forest Model
# ----------------------
@st.cache_resource
def train_model():
    """Trains and caches the Random Forest model."""
    X = df[['temperature', 'humidity', 'ph', 'soil_texture_encoded']]
    y = df['crop_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf_model = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    test_acc = rf_model.score(X_test, y_test)
    return rf_model, test_acc

rf_model, test_acc = train_model()

# ----------------------
# NPK Lookup Dictionary
# ----------------------
@st.cache_data
def get_npk_lookup():
    """Creates a dictionary for average NPK values per crop."""
    return df.groupby("label")[["N", "P", "K"]].mean().round(0).to_dict(orient="index")

avg_npk = get_npk_lookup()

# ----------------------
# Streamlit UI
# ----------------------

# --- Sidebar ---
st.sidebar.title("Configuration")
st.sidebar.metric("Model Test Accuracy", f"{test_acc*100:.1f}%")
top_k = st.sidebar.slider("Show Top-K Recommendations", min_value=1, max_value=5, value=3)

st.sidebar.header("About")
st.sidebar.info(
    "This app recommends crops based on environmental conditions using a "
    "Random Forest model. It also suggests average NPK fertilizer values "
    "for the recommended crops based on the dataset."
)

# --- Main Page ---
st.title("🌱 Smart Crop & NPK Recommendation System")
st.write(
    "Enter your soil and environmental conditions below to get crop and NPK suggestions. "
)

# ---------------------------------------------------------
# INPUT SECTION: Horizontal Number Inputs
# ---------------------------------------------------------
st.header("Enter Environmental Conditions")

# Create 3 columns for horizontal layout
col1, col2, col3 = st.columns(3)

with col1:
    temp = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        format="%.1f",
        help="Average temperature in Celsius."
    )

with col2:
    hum = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0,
        format="%.1f",
        help="Relative humidity percentage."
    )

with col3:
    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5,
        step=0.1,
        format="%.1f",
        help="Acidity or alkalinity of the soil (0-14)."
    )

# Soil texture remains as a dropdown below the numeric inputs
soil_options = sorted(df['soil_texture'].dropna().unique().tolist())
soil_type = st.selectbox(
    "Soil Texture",
    soil_options,
    help="Select the texture of your soil."
)

# --- Prediction Logic ---
if st.button("Get Recommendations", type="primary"):
    # Encode user input
    try:
        soil_encoded = soil_encoder.transform([soil_type])[0]
        features = np.array([[temp, hum, ph, soil_encoded]])
        
        # Get probabilities for all crops
        probabilities = rf_model.predict_proba(features)[0]
        
        # Get top-k predictions
        top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_k_crops = crop_encoder.inverse_transform(top_k_indices)
        top_k_probs = probabilities[top_k_indices]
        
        st.divider()
        st.subheader(f"Top {top_k} Crop Recommendations")

        # English → Nepali crop mapping
        crop_map = {
            "rice": "धान", "maize": "मकै", "wheat": "गहुँ", "barley": "जौ", "millet": "कोदो",
            "chickpea": "चना", "kidneybeans": "राजमा", "pigeonpeas": "रहर", "mothbeans": "मोठ",
            "mungbean": "मुंग", "blackgram": "मास", "lentil": "मसुरो", "pomegranate": "अनार",
            "banana": "केरा", "mango": "आँप", "grapes": "अंगुर", "watermelon": "तरबुजा",
            "muskmelon": "खरबुजा", "apple": "स्याउ", "orange": "सुन्तला", "papaya": "मेवा",
            "coconut": "नरिवल", "cotton": "कपास", "jute": "पटसन (जुट)", "coffee": "कफी",
            "sugarcane": "उखु", "tea": "चिया", "potato": "आलु", "mustard": "तोरी"
        }

        for i, (crop, prob) in enumerate(zip(top_k_crops, top_k_probs)):
            nepali_name = crop_map.get(crop, "")
            
            # Create an expander for each result
            with st.expander(f"**{i+1}. {crop}** ({nepali_name}) — Confidence: {prob:.1%}", expanded=(i==0)):
                st.caption(f"Recommended NPK Fertilizer usage for {crop}:")
                
                # Get NPK values from our lookup dictionary
                npk_values = avg_npk.get(crop, None)
                
                if npk_values:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Nitrogen (N)", f"{npk_values['N']:.0f} kg/ha")
                    with c2:
                        st.metric("Phosphorus (P)", f"{npk_values['P']:.0f} kg/ha")
                    with c3:
                        st.metric("Potassium (K)", f"{npk_values['K']:.0f} kg/ha")
                else:
                    st.warning("NPK data not available for this crop.")
                    
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
