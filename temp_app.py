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

#part-1 street analysis

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

st.title("Business Data Visualization")

st.markdown("### This tool helps you quickly understand the business environment in a specific area by analyzing street view videos. See what businesses are present, what types of shops are common, and even potential competitors.")

st.write("Upload a mp4 file to visualize the data.")
uploaded_file = st.file_uploader("Choose a Video file", type=["mp4"])

if uploaded_file is not None:
    video_file_id = upload_video_to_gemini(uploaded_file)
    #st.write(f"Video uploaded successfully: {video_file_id.uri}")

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
# 
    vidname = video_file_id.name

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

    data = json.loads(response_cleaned)

    #data = process_part1(temp_video_path)
    display_businesses(data)
    display_category_distribution(data)
    
    unique_categories = data['analysis_summary']['unique_categories']
    selected_category = st.selectbox("Select a category to view features", unique_categories)

    if selected_category:
        display_features_by_category(data, selected_category)

    st.write("### Analysis Summary")
    analysis_summary = data['analysis_summary']
    st.json(analysis_summary)

##part-2 cross-validation analysis

    class BusinessRegistrationHelper:
        def __init__(self, opencage_api_key: str):
            self.api_key = opencage_api_key
            self.logger = self._setup_logger()
            self.SEARCH_RADIUS = 2000  # 2 km radius

        def _setup_logger(self):
            logger = logging.getLogger(__name__)
            if not logger.handlers:
                handler = logging.FileHandler(
                    f'business_registration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                )
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
            return logger

        def _normalize_business_name(self, name: str) -> str:
            """
            Normalize business name by removing common words, punctuation, and converting to lowercase
            """
            # Remove common suffixes and prefixes
            name = re.sub(r'\b(the|shop|store|restaurant|cafe|mart)\b', '', name, flags=re.IGNORECASE)
            # Remove punctuation and extra whitespaces
            name = re.sub(r'[^\w\s]', '', name).strip().lower()
            return name

        def check_business_existence_in_radius(self, business_name: str, area_coordinates: Dict) -> Dict:
            """
            Comprehensively check business existence within a 2km radius using OpenCage API
            """
            try:
                # Normalize input business name
                normalized_input_name = self._normalize_business_name(business_name)

                # Geocode the center location
                geocode_url = f"https://api.opencagedata.com/geocode/v1/json?q={area_coordinates['lat']},{area_coordinates['lng']}&key={self.api_key}"
                response = requests.get(geocode_url).json()

                # Check if the geocode was successful
                if response['status']['code'] == 200:
                    matches = []
                    # Here, you would process and search for businesses using OpenCage's search functionality
                    # Currently, OpenCage only provides geocoding, so to search for businesses,
                    # you might need to use other data sources or external APIs.
                    # Assuming you could match data from other services, we'd proceed like this:

                    # Example logic for "matches" (you'd need to implement your own search logic or API calls)
                    matches.append({
                        'name': business_name,
                        'distance': geodesic((area_coordinates['lat'], area_coordinates['lng']), (response['results'][0]['geometry']['lat'], response['results'][0]['geometry']['lng'])).meters
                    })

                    return {
                        'exists': bool(matches),
                        'matches': matches,
                        'confidence': 0.9,  # Placeholder
                        'total_matches': len(matches)
                    }

                return {'exists': False, 'matches': [], 'confidence': 0, 'total_matches': 0}

            except Exception as e:
                self.logger.error(f"Error checking business existence: {str(e)}")
                return {
                    'exists': False,
                    'matches': [],
                    'confidence': 0,
                    'total_matches': 0,
                    'error': str(e)
                }

        def analyze_businesses_in_radius(
            self,
            businesses: List[Dict],
            area_address: str,
            max_workers: int = 4
        ) -> Dict:
            """
            Analyze businesses by checking their existence within a 2km radius
            """
            # Geocode the area address using OpenCage API
            geocode_url = f"https://api.opencagedata.com/geocode/v1/json?q={area_address}&key={self.api_key}"
            geocode_result = requests.get(geocode_url).json()

            if not geocode_result or geocode_result['status']['code'] != 200:
                raise ValueError(f"Could not geocode address: {area_address}")

            # Extract coordinates
            area_coordinates = geocode_result['results'][0]['geometry']

            analysis_results = {
                'registered_businesses': [],
                'unregistered_businesses': [],
                'potential_duplicates': [],
                'summary': {
                    'total_analyzed': len(businesses),
                    'registered_count': 0,
                    'unregistered_count': 0,
                    'duplicate_groups_count': 0,
                    'area_address': area_address,
                    'search_radius_meters': self.SEARCH_RADIUS
                }
            }

            # First, check for duplicates within input data
            business_groups = self._group_similar_businesses(businesses)

            # Process each business
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_business = {
                    executor.submit(
                        self._analyze_single_business,
                        business,
                        area_coordinates
                    ): business for business in businesses
                }

                for future in tqdm(
                    concurrent.futures.as_completed(future_to_business),
                    total=len(businesses),
                    desc="Analyzing businesses in 2km radius"
                ):
                    try:
                        result = future.result()
                        if result['exists']:
                            analysis_results['registered_businesses'].append({
                                'input_business': result['business'],
                                'matched_places': result['matches'],
                            })
                        else:
                            analysis_results['unregistered_businesses'].append(result['business'])
                    except Exception as e:
                        self.logger.error(f"Error in analysis: {str(e)}")

            # Add duplicate groups to results
            analysis_results['potential_duplicates'] = business_groups

            # Update summary
            analysis_results['summary'].update({
                'registered_count': len(analysis_results['registered_businesses']),
                'unregistered_count': len(analysis_results['unregistered_businesses']),
                'duplicate_groups_count': len(business_groups)
            })

            return analysis_results

        def _group_similar_businesses(self, businesses: List[Dict]) -> List[List[Dict]]:
            """Group similar businesses based on name similarity."""
            groups = []
            processed = set()

            for i, business1 in enumerate(businesses):
                if i in processed:
                    continue

                current_group = [business1]
                processed.add(i)

                normalized_name1 = self._normalize_business_name(business1['name'])

                for j, business2 in enumerate(businesses[i+1:], start=i+1):
                    if j not in processed:
                        # Normalize second business name
                        normalized_name2 = self._normalize_business_name(business2['name'])

                        similarity = self._calculate_similarity(
                            normalized_name1,
                            normalized_name2
                        )

                        if similarity > 0.8:  # Threshold for similarity
                            current_group.append(business2)
                            processed.add(j)

                if len(current_group) > 1:  # Only add groups with potential duplicates
                    groups.append(current_group)

            return groups

        def _analyze_single_business(self, business: Dict, area_coordinates: Dict) -> Dict:
            """
            Analyze a single business by checking its existence in the 2km radius
            """
            business_name = business.get('name', '')

            # Check business existence in the specified radius
            existence_check = self.check_business_existence_in_radius(
                business_name,
                area_coordinates
            )

            return {
                'business': business,
                'exists': existence_check['exists'],
                'matches': existence_check['matches']
            }

        from difflib import SequenceMatcher

        def _calculate_similarity(self, name1: str, name2: str) -> float:
            """
            Calculate the similarity between two business names using SequenceMatcher.
            Returns a float between 0 and 1, where 1 means identical strings.
            """
            return SequenceMatcher(None, name1, name2).ratio()

    helper = BusinessRegistrationHelper('7a5e2c08b15848f4ab7abca083e9b732')
    with open('working\output.json', 'r') as f:
        analyzed_data = json.load(f)
    #data = json.load('working\output.json')
    
    area_address = "Ramling Khind RdRaviwar Peth, Belagavi, Karnataka 590001"

    analysis_results = helper.analyze_businesses_in_radius(
        analyzed_data['businesses'],
        area_address
    )
    output_path = "working/cross_validation_analysis.json"

    with open(output_path, "w") as json_file:
        json.dump(analysis_results, json_file, indent=4)


    with open('working/cross_validation_analysis.json', 'r') as f:
        data = json.load(f)

    summary = data['summary']
    st.subheader('Summary')
    summary_data = {
        "Total Analyzed": summary['total_analyzed'],
        "Registered Businesses": summary['registered_count'],
        "Unregistered Businesses": summary['unregistered_count'],
        "Potential Duplicates": summary['duplicate_groups_count'],
        "Area Address": summary['area_address'],
        "Search Radius (meters)": summary['search_radius_meters']
    }
    st.write(summary_data)

    # Registered Businesses section
    st.subheader('Registered Businesses')
    registered_data = []

    for business in data['registered_businesses']:
        for place in business['matched_places']:
            registered_data.append({
                "Business Name": business['input_business']['name'],
                "Category": business['input_business']['category'],
                "Timestamp": business['input_business']['timestamp'],
                "Features": ", ".join(business['input_business']['features']),
                "Matched Place Name": place['name'],
            })

    # Display registered businesses in a table
    if registered_data:
        registered_df = pd.DataFrame(registered_data)
        st.dataframe(registered_df)
    else:
        st.write("No registered businesses data available.")

    # Unregistered Businesses section
    st.subheader('Unregistered Businesses')
    unregistered_data = []

    for business in data['unregistered_businesses']:
        unregistered_data.append({
            "Business Name": business['name'],
            "Category": business['category'],
            "Timestamp": business['timestamp'],
            "Features": ", ".join(business['features'])
        })

    # Display unregistered businesses in a table
    if unregistered_data:
        unregistered_df = pd.DataFrame(unregistered_data)
        st.dataframe(unregistered_df)
    else:
        st.write("No unregistered businesses data available.")

    # Potential duplicates section
    st.subheader('Potential Duplicates')
    if data['potential_duplicates']:
        potential_duplicates_data = []

        for group in data['potential_duplicates']:
            for business in group:
                potential_duplicates_data.append({
                    "Business Name": business['name'],
                    "Category": business['category'],
                    "Timestamp": business['timestamp'],
                    "Features": ", ".join(business['features'])
                })

        # Display potential duplicates in a table
        potential_duplicates_df = pd.DataFrame(potential_duplicates_data)
        st.dataframe(potential_duplicates_df)
    else:
        st.write("No potential duplicates found.")

#part-3 

    class EnhancedMarketAnalyzer:
        """Advanced market analyzer using OpenCage API for geocoding"""

        def __init__(self, opencage_api_key: str, gemini_api_key: str):
            if not opencage_api_key or not gemini_api_key:
                raise ValueError("Both OpenCage and Gemini API keys are required")

            self.opencage_api_key = opencage_api_key
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('models/gemini-1.5-flash')

            self.setup_logging()
            self._validate_api_keys()

        def setup_logging(self):
            """Set up logging to file and console"""
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = os.path.join(log_dir, f'market_analysis_{timestamp}.log')
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(log_filename)
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        def _validate_api_keys(self):
            """Validate API keys by making test requests"""

            # Validate OpenCage API Key
            test_location = "New York, USA"
            url = f"https://api.opencagedata.com/geocode/v1/json?q={test_location}&key={self.opencage_api_key}"
            response = requests.get(url)
            if response.status_code != 200:
                self.logger.error("Invalid OpenCage API Key")
                raise ValueError("Invalid OpenCage API Key")

            # Validate Gemini API Key
            try:
                prompt = "Test"
                response = self.model.generate_content(prompt)
                if not response or not response.text:
                    raise ValueError("Invalid Gemini API Key")
            except Exception:
                self.logger.error("Invalid Gemini API Key")
                raise ValueError("Invalid Gemini API Key")

        def analyze_opportunity(self, location: str, radius: int = 2000, output_json: str = "recommended_locations.json") -> Dict:
            """Main function to analyze market opportunities"""
            self.logger.info(f"Starting market analysis for location: {location}")

            lat_lng = self._get_lat_lng(location)
            if not lat_lng:
                raise ValueError(f"Could not geocode location: {location}")

            place_type = "13032"  # Example Foursquare category ID for restaurants (change as needed)
            competitors = self._get_competitors(lat_lng, radius, place_type)

            processed_data = self._process_competitor_data(competitors)
            recommendations = self._suggest_optimal_locations(competitors, lat_lng)
            self._save_recommendations_to_json(recommendations, output_json)

            insight = self._generate_gemini_insights(processed_data, location)

            report = {
                "market_insights": {
                    "location": location,
                    "competitor_analysis": processed_data,
                    "recommendations": recommendations,
                    "opportunities": insight
                }
            }

            print(json.dumps(report, indent=4))
            return report

        def _get_lat_lng(self, location: str) -> Dict:
            """Get latitude and longitude from OpenCage API"""
            url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={self.opencage_api_key}"
            response = requests.get(url).json()
            if response and response.get("results"):
                lat_lng = response["results"][0]["geometry"]
                return {"lat": lat_lng["lat"], "lng": lat_lng["lng"]}
            return {}

        def _get_competitors(self, lat_lng: Dict, radius: int, place_type: str) -> List[Dict]:
            """Fetch competitor data using Foursquare API"""
            self.logger.info(f"Fetching {place_type} competitors within {radius}m radius")

            foursquare_api_key = "fsq3XwkcsrD0oFt/MWy6+zmw0zc1MireOJ4zmCbXP+BpXe0="
            search_url = "https://api.foursquare.com/v3/places/search"

            headers = {
                "Accept": "application/json",
                "Authorization": foursquare_api_key
            }

            params = {
                "ll": f"{lat_lng['lat']},{lat_lng['lng']}",
                "radius": radius,
                "limit": 10,  # Adjust limit as needed
                "categories": place_type  # Place category instead of type
            }

            try:
                response = requests.get(search_url, headers=headers, params=params)

                if response.status_code != 200:
                    self.logger.error(f"Foursquare API Error: {response.status_code} - {response.text}")
                    return []

                places_data = response.json()
                if "results" not in places_data:
                    self.logger.warning("No competitors found in Foursquare API response")
                    return []

                competitors = []
                for place in places_data["results"]:
                    name = place.get("name", "Unknown")
                    category = place["categories"][0]["name"] if place.get("categories") else "Unknown"
                    distance = round(place.get("distance", 0) / 1000, 2)  # Convert meters to km
                    latitude = place["geocodes"]["main"]["latitude"] if "geocodes" in place else None
                    longitude = place["geocodes"]["main"]["longitude"] if "geocodes" in place else None
                    rating = place.get("rating", "N/A")  # Foursquare does not always provide ratings
                    reviews_count = place.get("stats", {}).get("total_ratings", 0)  # Some stats may be missing

                    # Process reviews (Foursquare does not provide full reviews, so we use placeholders)
                    reviews = ["Review data not available from Foursquare"]
                    sentiment, subjectivity = self._analyze_reviews(reviews)

                    competitors.append({
                        "name": name,
                        "category": category,
                        "rating": rating,
                        "reviews_count": reviews_count,
                        "price_level": "N/A",  # Foursquare does not provide price level
                        "distance": distance,
                        "latitude": latitude,
                        "longitude": longitude,
                        "reviews": reviews,
                        "sentiment_score": round(sentiment, 2),
                        "subjectivity_score": round(subjectivity, 2)
                    })

                self.logger.info(f"Fetched {len(competitors)} competitors from Foursquare API")
                return competitors

            except Exception as e:
                self.logger.error(f"Error fetching competitors: {str(e)}")
                return []
            
        def _suggest_optimal_locations(self, competitors: List[Dict], current_location: Dict) -> List[Dict]:
            """Suggests optimal locations based on competition"""
            self.logger.info("Analyzing optimal locations based on competitor data")

            recommended_locations = []

            for competitor in competitors:
                rating = float(competitor.get('rating', 0)) if competitor.get('rating') not in [None, "N/A"] else 0
                distance = float(competitor.get('distance_km', 0)) if competitor.get('distance_km') != "unknown" else 0

                if rating >= 0 and distance >= 0:
                    recommended_locations.append({
                        "latitude": competitor["latitude"],
                        "longitude": competitor["longitude"],
                        "name": competitor["name"],
                        "rating": rating,
                        "distance_km": distance,
                    })

            return recommended_locations

        def _process_competitor_data(self, competitors: List[Dict]) -> str:
            """Process competitor data into a structured summary"""
            competitor_summary = []
            for competitor in competitors:
                summary = (
                    f"{competitor['name']} has a rating of {competitor.get('rating', 'N/A')} "
                    f"from {competitor.get('reviews_count', 0)} reviews, "
                    f"located {competitor.get('distance_km', 'unknown')} km away."
                )
                competitor_summary.append(summary)
            return "\n".join(competitor_summary)
        
        def _save_recommendations_to_json(self, recommendations: List[Dict], output_file: str):
            """Save recommendations to a JSON file"""
            with open(output_file, "w") as file:
                json.dump(recommendations, file, indent=4)
            self.logger.info(f"Recommendations saved to {output_file}")
            
        def _generate_gemini_insights(self, competitor_summary: str, location: str) -> str:
            """Generate market insights using Gemini AI"""
            prompt = f"""
                As an experienced business analyst, provide a detailed analysis for:
                Business Idea: {location}
                Market Data:
                Competitor Analysis: {competitor_summary}
            """
            response = self.model.generate_content(prompt)
            return response.text if response else "No insights available"
        
        def _analyze_reviews(self, reviews: List) -> (float, float):
            sentiments = []
            subjectivities = []

            for review in reviews:
                if isinstance(review, dict):  # If review is a dictionary, extract the text
                    review_text = review.get('text', '')
                elif isinstance(review, str):  # If review is already a string, use it directly
                    review_text = review
                else:
                    continue  # Ignore unexpected data types

                analysis = TextBlob(review_text).sentiment
                sentiments.append(analysis.polarity)
                subjectivities.append(analysis.subjectivity)

            return (
                sum(sentiments) / len(sentiments) if sentiments else 0.0,
                sum(subjectivities) / len(subjectivities) if subjectivities else 0.0
            )
    

    opencage_api_key = "7a5e2c08b15848f4ab7abca083e9b732"
    gemini_api_key = "AIzaSyBovJ8q8zWYX-tu1aCBXFsea-CtSa-Ra4M"
    analyzer = EnhancedMarketAnalyzer(opencage_api_key, gemini_api_key)
    location = "belgaum, India"
    result = analyzer.analyze_opportunity(location)

    with open('working/market_analysis_report.json', 'w') as json_file:
        json.dump(result, json_file, indent=4)
    print("Market analysis report saved to market_analysis_report.json")

    def load_json(uploaded_file_path):
        with open(uploaded_file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def display_recommendations(data):
        recommendations = data['market_insights'].get('recommendations', [])
        if recommendations:
            map_center = [recommendations[0]['latitude'], recommendations[0]['longitude']]
            m = folium.Map(location=map_center, zoom_start=14)

            st.subheader("Existing Establishment")
            for rec in recommendations:
                # You can use the default icon or customize it
                folium.Marker(
                    location=[rec['latitude'], rec['longitude']],
                    popup=f"{rec.get('name', 'Unknown Name')} - Rating: {rec.get('rating', 'N/A')} - Sentiment Score: {rec.get('sentiment_score', 'N/A')}",
                    icon=Icon(color='blue', icon='info-sign')  # Example of a custom marker
                ).add_to(m)

            st_folium(m, width=700, height=500)
        else:
            st.warning("No restaurant recommendations found.")

# Display opportunities as a section
    def display_opportunities(data):
        opportunities = data['market_insights'].get('opportunities', "")
        if opportunities:
            st.subheader("Business Opportunities")
            st.markdown(opportunities)
        else:
            st.warning("No business opportunities found.")


    st.title("Market Insights By Gemini 1.5 Flash")
    # Pass the uploaded file directly to the load_json function
    data = load_json('working\market_analysis_report.json')
    
    # Display market insights
    location = data['market_insights'].get('location', 'Unknown Location')
    st.subheader("Location: " + location)
    
    # Display restaurant recommendations with map
    display_recommendations(data)
    
    # Display business opportunities
    display_opportunities(data)










































