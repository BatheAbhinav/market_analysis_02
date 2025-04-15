import os
import json
import logging
import requests
import concurrent.futures
from datetime import datetime
from difflib import SequenceMatcher
from geopy.distance import geodesic
from tqdm import tqdm
import streamlit as st
import pandas as pd
import folium
from folium import Icon
from streamlit_folium import st_folium
from textblob import TextBlob
import google.generativeai as genai

# Load API keys from environment variables
GOOGLE_MAPS_API_KEY = '7a5e2c08b15848f4ab7abca083e9b732'
GEMINI_API_KEY = 'AIzaSyBovJ8q8zWYX-tu1aCBXFsea-CtSa-Ra4M'
FOURSQUARE_API_KEY = "fsq3XwkcsrD0oFt/MWy6+zmw0zc1MireOJ4zmCbXP+BpXe0="

if not GOOGLE_MAPS_API_KEY or not GEMINI_API_KEY:
    raise ValueError("API keys are missing. Set them as environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# File handling
WORKING_DIR = os.path.join("F:", "maps_on_roids", "working")
os.makedirs(WORKING_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(WORKING_DIR, "output.json")
ANALYSIS_PATH = os.path.join(WORKING_DIR, "market_analysis_report.json")

# Load JSON utility
def load_json(uploaded_file):
    try:
        return json.load(uploaded_file)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON file: {str(e)}")
        return None

# Display business data
def display_businesses(data):
    df = pd.json_normalize(data.get('businesses', []))
    st.write("### Businesses Data")
    st.dataframe(df)

# Business Registration Helper class
class BusinessRegistrationHelper:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = self._setup_logger()
        self.SEARCH_RADIUS = 2000  # 2 km radius
    
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(WORKING_DIR, 'business_registration.log'))
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger
    
    def _normalize_business_name(self, name: str) -> str:
        return re.sub(r'[^\w\s]', '', name).strip().lower()
    
    def check_business_existence_in_radius(self, business_name: str, area_coordinates: dict) -> dict:
        try:
            normalized_name = self._normalize_business_name(business_name)
            geocode_url = f"https://api.opencagedata.com/geocode/v1/json?q={area_coordinates['lat']},{area_coordinates['lng']}&key={self.api_key}"
            response = requests.get(geocode_url).json()
            
            if response['status']['code'] == 200:
                return {
                    'exists': True,
                    'matches': [{'name': business_name, 'distance': geodesic((area_coordinates['lat'], area_coordinates['lng']), (response['results'][0]['geometry']['lat'], response['results'][0]['geometry']['lng'])).meters}],
                    'confidence': 0.9,
                    'total_matches': 1
                }
            return {'exists': False, 'matches': [], 'confidence': 0, 'total_matches': 0}
        except Exception as e:
            self.logger.error(f"Error checking business existence: {str(e)}")
            return {'exists': False, 'matches': [], 'confidence': 0, 'total_matches': 0, 'error': str(e)}

# Streamlit UI
def main():
    st.title("Market Insights By Gemini 1.5 Flash")
    uploaded_file = st.file_uploader("Upload JSON File", type=["json"])
    
    if uploaded_file is not None:
        data = load_json(uploaded_file)
        if data:
            location = data['market_insights'].get('location', 'Unknown Location')
            st.subheader(f"Location: {location}")
            display_businesses(data)

if __name__ == "__main__":
    main()
