import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import json
import os
import streamlit as st
import tempfile
from tempfile import NamedTemporaryFile
import pandas as pd
import matplotlib.pyplot as plt
import time
import requests
import logging
from typing import Dict, List
from datetime import datetime
from geopy.distance import geodesic
import re

from difflib import SequenceMatcher
import concurrent.futures
from tqdm import tqdm

import folium
from folium import Icon
from streamlit_folium import st_folium
from textblob import TextBlob


google_maps_api_key = '7a5e2c08b15848f4ab7abca083e9b732' #OpenCage Api
genai.configure(api_key='AIzaSyBovJ8q8zWYX-tu1aCBXFsea-CtSa-Ra4M') #Gemini

def display_businesses(data):
    df = pd.json_normalize(data['businesses'])
    st.write("### Businesses Data")
    st.dataframe(df)

def display_category_distribution(data):
    categories = [business['category'] for business in data['businesses']]
    category_count = pd.Series(categories).value_counts()

    st.write("### Category Distribution")
    st.bar_chart(category_count)

def display_features_by_category(data, selected_category):
    filtered_data = [business for business in data['businesses'] if business['category'] == selected_category]
    features = []
    for business in filtered_data:
        features.extend(business['features'])

    st.write(f"### Features of {selected_category} Businesses")
    feature_count = pd.Series(features).value_counts()
    st.write(feature_count)

def upload_video_to_gemini(video_file):
    # Save the uploaded file to a temporary location
    with NamedTemporaryFile(suffix=".mp4", delete = False) as temp_file:
        temp_file.write(video_file.getvalue())
        temp_file.flush()
        
        video_file_id = genai.upload_file(path=temp_file.name, mime_type="video/mp4")
        return video_file_id
    
model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b-latest",
        system_instruction="Output format should be in JSON format"
    )
prompt = '''You are a computer vision expert specialized in analyzing street view footage. Please analyze this video and provide a detailed list of all visible businesses and shops. Respond ONLY in the following JSON format:

    {
        "businesses": [
            {
                "timestamp": "MM:SS",
                "name": "Business Name",
                "category": "Business Category",
                "address": "Detailed Address",
                "features": ["Feature 1", "Feature 2", "Feature 3"]
            }
        ],
        "analysis_summary": {
            "total_businesses": 0,
            "unique_categories": [],
            "analysis_time": "YYYY-MM-DD HH:MM:SS"
        }
    }

    Important instructions:
    1. Only include businesses with clearly visible signage and make sure that use complete signage  as some times single signage have multiple names
    2. Timestamp should be in MM:SS format
    3. Address must be as detailed and accurate as possible from the video
    4. Features should be a list of strings
    5. Ensure the response is valid JSON
    6. Do not include any explanatory text outside the JSON structure
    7. Do not repeat businesses that appear multiple times
    8. Extract full address information when visible in the video
    9. Make sure the Provided JSON content is accurate.
'''


def wait_for_file_active(file_id, timeout=300, interval=5):
    start_time = time.time()
    while True:
        file_info = genai.get_file(file_id.name)
        print("Checking file state:", file_info.state.name)  # or st.write
        if file_info.state.name == "ACTIVE":
            return file_info
        elif file_info.state.name == "FAILED":
            raise RuntimeError(f"Gemini file processing failed.")
        elif time.time() - start_time > timeout:
            raise TimeoutError("Waiting for file to become ACTIVE timed out.")
        time.sleep(interval)

vid = wait_for_file_active(video_file_id)
    #st.write(model.count_tokens([prompt, vid]))
response = model.generate_content([vid, prompt])
output_path = "working/output.json"  
working_dir = "working"