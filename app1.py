import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import json
import os
import streamlit as st
import tempfile
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

#video_file_name = "D:/Users/bathe/Downloads/street1-part2 (1).mp4"

google_maps_api_key = '7a5e2c08b15848f4ab7abca083e9b732' #OpenCage Api
genai.configure(api_key='AIzaSyBt4IFmbCIXC_j1HaFhjwxnLQEzWjSG87w') #Gemini

def process_part1(video_file_name):
    video_file = genai.upload_file(path = "D:/Users/bathe/Downloads/street1-part2 (1).mp4" ).uri
    vidname = video_file
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

    #while video_file.state.name == "PROCESSING":
    #    time.sleep(10)
    #    vid = genai.get_file(vidname)
    #    print(f"File Status: {video_file.state.name}")

    #if video_file.state.name == "FAILED":
    #    raise ValueError(video_file.state.name)
    vid = genai.get_file(vidname)
    response = model.generate_content([vid, prompt])
    output_path = "working/output.json"  
    working_dir = "working"

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    lines = response.text.splitlines()

    limited_lines = "\n".join(lines[:30])

    display(Markdown(limited_lines))

    response_cleaned = response.text.strip()
    if response_cleaned.startswith("```json"):
        response_cleaned = response_cleaned[7:]
    if response_cleaned.endswith("```"):
        response_cleaned = response_cleaned[:-3]

    if not os.path.exists(output_path):
        with open(output_path, "w") as json_file:
            json.dump({}, json_file)

    with open(output_path, "w") as json_file:
        json_file.write(response_cleaned)

    json_data = json.loads(response_cleaned)
    return json_data

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

## Streamlit UI-1
st.title("Business Data Visualization")
st.write("Upload a mp4 file to visualize the data.")

uploaded_file = st.file_uploader("Choose a Video file", type=["mp4"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete = False, suffix = ".mp4")
    tfile.write(uploaded_file.getbuffer())
    temp_video_path = tfile.name
    #temp_video_path = "temp_video.mp4"

    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File successfully saved as {temp_video_path}")

    #video_file = genai.upload_file(path = "temp_video.mp4")
    data = process_part1(temp_video_path)
    display_businesses(data)
    display_category_distribution(data)

    unique_categories = data['analysis_summary']['unique_categories']
    selected_category = st.selectbox("Select a category to view features", unique_categories)

    if selected_category:
        display_features_by_category(data, selected_category)

    st.write("### Analysis Summary")
    analysis_summary = data['analysis_summary']
    st.json(analysis_summary)
