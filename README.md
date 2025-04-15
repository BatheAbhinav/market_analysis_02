# Street View Business Intelligence Engine

## Overview

This application automatically analyzes street view videos to extract and present business data, providing valuable market insights. It leverages Generative AI and computer vision to identify businesses, their attributes, and overall market trends.

## Features

-   *AI-Powered Business Identification:* Utilizes the Gemini AI API to automatically identify businesses in street view videos.
-   *Data Extraction:* Extracts key business attributes, including name, category, address, and other features.
-   *Geospatial Validation:* Enhances data accuracy by integrating the OpenCage API for geocoding and cross-validation of location data.
-   *Market Analysis:* Provides analysis of category distribution, popular business features, and overall market trends.
-   *Interactive UI:* Offers an intuitive Streamlit-based user interface for video upload, data visualization, and interactive analysis.

## Technologies Used

-   Python
-   Streamlit
-   Gemini AI API
-   OpenCage API
-   Pandas
-   JSON

## Setup

1.  *Clone the repository:*
    
    git clone <repository_url>
    cd <repository_directory>
    
2.  *Install the dependencies:*
    
    pip install -r requirements.txt
    
3.  *Set up API Keys:*
    -   Obtain API keys from Google Gemini AI and OpenCage Data.
    -   Store your API keys securely using Streamlit's Secrets management. Create a .streamlit/secrets.toml file or use the Streamlit Cloud dashboard:
        
        OPENCAGE_API_KEY = "YOUR_OPENCAGE_API_KEY"
        GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
        
4.  *Run the application:*
    
    streamlit run temp_app.py
    

## Usage

1.  Upload a street view video in MP4 format.
2.  The application will automatically analyze the video and extract business data.
3.  Explore the interactive visualizations and analysis provided by the application.

## Notes

-   Ensure your API keys are correctly configured and have sufficient quota for the application to function properly.
-   The accuracy of business identification and data extraction depends on the quality of the video and the visibility of business signage.
-   For advanced use cases, consider fine-tuning the Gemini AI prompt or integrating additional data sources.
