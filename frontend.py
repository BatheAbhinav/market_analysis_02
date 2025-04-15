import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
# Function to load and display JSON data

from backend import (load_json, display_businesses, display_category_distribution, display_features_by_category)

uploaded_file = st.file_uploader("Choose a JSON file", type=["json"])

if uploaded_file is not None:
    data = load_json(uploaded_file)

    # Display businesses data as a table
    display_businesses(data)

    # Display category distribution
    display_category_distribution(data)

    # Show a selection for categories to filter features
    unique_categories = data['analysis_summary']['unique_categories']
    selected_category = st.selectbox("Select a category to view features", unique_categories)

    if selected_category:
        display_features_by_category(data, selected_category)

    # Display the analysis summary
    st.write("### Analysis Summary")
    analysis_summary = data['analysis_summary']
    st.json(analysis_summary)