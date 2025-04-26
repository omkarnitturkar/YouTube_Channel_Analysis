import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
import isodate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sklearn.linear_model import LinearRegression
import base64

# Set page configuration
st.set_page_config(
    page_title="YouTube Analytics Pro",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with more professional styling
st.markdown("""
    <style>
    /* Global Styles */
    :root {
        --primary-color: #FF0000;
        --secondary-color: #282828;
        --text-color: #303030;
        --light-bg: #f9f9f9;
        --border-radius: 8px;
        --shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 30px;
        font-weight: 700;
        padding: 15px 0;
        border-bottom: 2px solid #f0f0f0;
    }
    
    /* Subheader styling */
    .sub-header {
        font-size: 1.5rem;
        color: var(--secondary-color);
        margin-top: 20px;
        margin-bottom: 15px;
        font-weight: 600;
        padding-bottom: 5px;
        border-bottom: 1px solid #f0f0f0;
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: var(--light-bg);
        border-radius: var(--border-radius);
        padding: 20px;
        box-shadow: var(--shadow);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Chart styling */
    .stPlotlyChart {
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        background-color: white;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--secondary-color);
    }
    
    .sidebar .sidebar-content {
        background-color: var(--secondary-color);
    }

    /* Search section styling */
    .search-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: var(--shadow);
    }

    .search-input {
        margin-bottom: 10px !important;
    }

    .search-button {
        background-color: #FF0000 !important;
        color: white !important;
        width: 100% !important;
        margin-top: 5px !important;
    }

    .channel-select {
        margin-top: 15px !important;
    }

    .channel-info-card {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        margin-top: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-positive {
        background-color: #00cc66;
    }
    
    .status-negative {
        background-color: #ff3366;
    }
    
    .status-neutral {
        background-color: #ffcc00;
    }
    
    /* Table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        box-shadow: var(--shadow);
        border-radius: var(--border-radius);
        overflow: hidden;
    }
    
    .styled-table thead tr {
        background-color: var(--primary-color);
        color: white;
        text-align: left;
    }
    
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
    }
    
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid var(--primary-color);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #f0f0f0;
        color: var(--text-color);
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Insert your API key here
API_KEY = "AIzaSyBetONqBCgl1VUTsw3BJoA1F7_JH7UZHJg"  # Replace with your actual API key

# Create a function to generate download links
def get_download_link(df, filename, text):
    """Generate a link to download the dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-btn">{text}</a>'
    return href

# Cache the API functions
@st.cache_data(ttl=3600)
def search_and_select_channel(api_key, channel_name):
    """Search for YouTube channels by name and return as list"""
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        search_response = youtube.search().list(
            part="snippet",
            q=channel_name,
            type="channel",
            maxResults=5
        ).execute()
        
        search_results = []
        
        for item in search_response.get('items', []):
            channel_info = {
                'value': item['id']['channelId'],
                'label': item['snippet']['title'],
                'Channel ID': item['id']['channelId'],
                'Channel Name': item['snippet']['title'],
                'Description': item['snippet']['description'],
                'Thumbnail': item['snippet']['thumbnails']['default']['url']
            }
            search_results.append(channel_info)
            
        return search_results
    
    except Exception as e:
        st.error(f"An error occurred during search: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def fetch_youtube_data(api_key, channel_id):
    """Fetch data from YouTube API with caching"""
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Get channel information
        channel_response = youtube.channels().list(
            part="snippet,contentDetails,statistics,brandingSettings",
            id=channel_id
        ).execute()
        
        if not channel_response['items']:
            return None, None
        
        channel_data = channel_response['items'][0]
        
        # Extract more detailed channel data
        channel_stats = {
            'Channel Name': channel_data['snippet']['title'],
            'Channel ID': channel_id,
            'Subscribers': int(channel_data['statistics'].get('subscriberCount', 0)),
            'Views': int(channel_data['statistics'].get('viewCount', 0)),
            'Total Videos': int(channel_data['statistics'].get('videoCount', 0)),
            'Channel Created': channel_data['snippet']['publishedAt'],
            'Description': channel_data['snippet']['description'],
            'Country': channel_data['snippet'].get('country', 'Not specified'),
            'Playlist ID': channel_data['contentDetails']['relatedPlaylists']['uploads'],
            'Banner URL': channel_data.get('brandingSettings', {}).get('image', {}).get('bannerExternalUrl', ''),
            'Custom URL': channel_data['snippet'].get('customUrl', '')
        }
        
        # Get video IDs from uploads playlist - limit to 100 videos
        video_ids = []
        next_page_token = None
        max_videos = 100
        
        while True:
            playlist_response = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=channel_stats['Playlist ID'],
                maxResults=50,
                pageToken=next_page_token
            ).execute()
            
            # Get video IDs
            for item in playlist_response['items']:
                video_ids.append(item['contentDetails']['videoId'])
                if len(video_ids) >= max_videos:
                    break
                    
            if len(video_ids) >= max_videos or not playlist_response.get('nextPageToken'):
                break
                
            next_page_token = playlist_response.get('nextPageToken')
                
        # Get video details in batches
        all_video_data = []
        
        # Process in batches of 50 (API limit)
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            
            video_response = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=','.join(batch_ids)
            ).execute()
            
            for video in video_response['items']:
                # Extract relevant data
                video_data = {
                    'Video ID': video['id'],
                    'Title': video['snippet']['title'],
                    'Published Date': video['snippet']['publishedAt'],
                    'Description': video['snippet']['description'],
                    'Tags': video['snippet'].get('tags', []),
                    'Category ID': video['snippet'].get('categoryId', ''),
                    'Duration': video['contentDetails']['duration'],
                    'Definition': video['contentDetails'].get('definition', ''),  # HD or SD
                    'Caption': video['contentDetails'].get('caption', 'false'),  # Has captions
                    'Views': int(video['statistics'].get('viewCount', 0)),
                    'Likes': int(video['statistics'].get('likeCount', 0)),
                    'Comments': int(video['statistics'].get('commentCount', 0)),
                    'Thumbnail': video['snippet']['thumbnails'].get('high', {}).get('url', '')
                }
                
                all_video_data.append(video_data)
                
        return channel_stats, all_video_data
        
    except HttpError as e:
        st.error(f"YouTube API error: {str(e)}")
        return None, None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None

def enhance_channel_data(channel_stats, video_df):
    """Add additional metrics to channel stats"""
    enhanced_stats = channel_stats.copy()
    
    # Calculate channel age
    pub_date = pd.to_datetime(enhanced_stats['Channel Created'])
    enhanced_stats['Age Days'] = (datetime.now() - pub_date.tz_localize(None)).days
    enhanced_stats['Age Years'] = enhanced_stats['Age Days'] / 365.25
    
    # Videos per time period
    enhanced_stats['Videos per Month'] = round(video_df.shape[0] / (enhanced_stats['Age Days'] / 30), 2)
    enhanced_stats['Videos per Year'] = round(video_df.shape[0] / enhanced_stats['Age Years'], 2)
    
    # Growth metrics
    enhanced_stats['Views per Day'] = round(enhanced_stats['Views'] / enhanced_stats['Age Days'], 2)
    enhanced_stats['Subs per Day'] = round(enhanced_stats['Subscribers'] / enhanced_stats['Age Days'], 2)
    
    # Engagement metrics
    enhanced_stats['Views per Video'] = round(enhanced_stats['Views'] / enhanced_stats['Total Videos'], 2)
    enhanced_stats['Comments per Video'] = round(video_df['Comments'].mean(), 2)
    enhanced_stats['Likes per Video'] = round(video_df['Likes'].mean(), 2)
    enhanced_stats['Engagement Rate'] = round((enhanced_stats['Likes per Video'] + enhanced_stats['Comments per Video']) / 
                                             enhanced_stats['Views per Video'] * 100, 2)
    
    # Performance trends
    if not video_df.empty and 'Published Date' in video_df.columns:
        # Make cutoff_date timezone-aware to match 'Published Date' in video_df
        cutoff_date = (datetime.now() - timedelta(days=90))
        # Ensure Published Date column is properly converted to datetime
        if pd.api.types.is_datetime64_any_dtype(video_df['Published Date']):
            # Make both objects timezone-naive for proper comparison
            if hasattr(video_df['Published Date'].iloc[0], 'tz') and video_df['Published Date'].iloc[0].tz is not None:
                video_df['Published Date'] = video_df['Published Date'].dt.tz_localize(None)
                
            recent_df = video_df[video_df['Published Date'] >= cutoff_date]
            
            if not recent_df.empty:
                enhanced_stats['Recent Videos'] = len(recent_df)
                enhanced_stats['Recent Avg Views'] = round(recent_df['Views'].mean(), 2)
                enhanced_stats['Overall Avg Views'] = round(video_df['Views'].mean(), 2)
                enhanced_stats['View Trend'] = round(enhanced_stats['Recent Avg Views'] / enhanced_stats['Overall Avg Views'] * 100 - 100, 2)
                
                # Add subscriber/view ratio
                enhanced_stats['Subs to Views Ratio'] = round(enhanced_stats['Subscribers'] / enhanced_stats['Views'] * 100, 2)
            else:
                enhanced_stats['Recent Videos'] = 0
                enhanced_stats['Recent Avg Views'] = 0
                enhanced_stats['Overall Avg Views'] = round(video_df['Views'].mean(), 2)
                enhanced_stats['View Trend'] = 0
                enhanced_stats['Subs to Views Ratio'] = round(enhanced_stats['Subscribers'] / enhanced_stats['Views'] * 100, 2)
        
    return enhanced_stats

def enhance_video_data(video_data):
    """Process video data and add additional metrics for analysis"""
    # Convert to DataFrame if it's a list
    if isinstance(video_data, list):
        df = pd.DataFrame(video_data)
    else:
        df = video_data.copy()
    
    # Convert duration from ISO format to seconds
    df['Duration (sec)'] = df['Duration'].apply(lambda x: isodate.parse_duration(x).total_seconds())
    
    # Add minutes for better readability
    df['Duration (min)'] = df['Duration (sec)'] / 60
    
    # Categorize videos by duration
    df['Duration Category'] = pd.cut(
        df['Duration (sec)'],
        bins=[0, 60, 300, 600, 1200, float('inf')],
        labels=['< 1 min', '1-5 min', '5-10 min', '10-20 min', '> 20 min']
    )
    
    # Convert published date to datetime and make timezone-naive
    df['Published Date'] = pd.to_datetime(df['Published Date'])
    if not df.empty and hasattr(df['Published Date'].iloc[0], 'tz') and df['Published Date'].iloc[0].tz is not None:
        df['Published Date'] = df['Published Date'].dt.tz_localize(None)
    
    # Extract day of week and hour information
    df['Publish Day of Week'] = df['Published Date'].dt.day_name()
    df['Hour Published'] = df['Published Date'].dt.hour
    
    # Extract month and year for time series analysis
    df['Year-Month'] = df['Published Date'].dt.strftime('%Y-%m')
    df['Year'] = df['Published Date'].dt.year
    df['Month'] = df['Published Date'].dt.month
    df['Month Name'] = df['Published Date'].dt.month_name()
    
    # Calculate engagement metrics
    df['Engagement Rate'] = ((df['Likes'] + df['Comments']) / df['Views'] * 100).fillna(0)
    df['Like to View Ratio'] = (df['Likes'] / df['Views'] * 100).fillna(0)
    df['Comment to View Ratio'] = (df['Comments'] / df['Views'] * 100).fillna(0)
    
    # Calculate daily metrics (views per day since publication)
    current_date = datetime.now()
    # Make timezone-aware dates comparable by removing timezone info
    days_since = (current_date - df['Published Date']).dt.days
    df['Days Since Published'] = days_since
    
    # Avoid division by zero
    df['Daily Views'] = df['Views'] / df['Days Since Published'].clip(lower=1)
    df['Daily Likes'] = df['Likes'] / df['Days Since Published'].clip(lower=1)
    df['Daily Comments'] = df['Comments'] / df['Days Since Published'].clip(lower=1)
    
    # Calculate performance category (quartiles)
    df['Performance'] = pd.qcut(
        df['Views'], 
        q=4, 
        labels=['Low', 'Medium', 'High', 'Top']
    )
    
    # Add relative performance (% of max views)
    max_views = df['Views'].max()
    if max_views > 0:
        df['Relative Performance'] = (df['Views'] / max_views * 100).round(1)
    else:
        df['Relative Performance'] = 0
    
    # Title analysis
    df['Title Length'] = df['Title'].str.len()
    df['Title Word Count'] = df['Title'].str.split().str.len()
    df['Has Question'] = df['Title'].str.contains(r'\?').fillna(False)
    df['Has Number'] = df['Title'].str.contains(r'\d').fillna(False)
    
    # Add emoji detection in title
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE
    )
    df['Has Emoji'] = df['Title'].apply(lambda x: bool(emoji_pattern.search(x)))
    
    # Extract keywords from titles
    df['Title Words'] = df['Title'].apply(
        lambda x: [word.lower() for word in re.findall(r'\b[a-zA-Z]{3,}\b', x) 
                  if word.lower() not in ['the', 'and', 'for', 'with', 'this', 'that', 'what', 'how', 'why']]
    )
    
    # Add description length analysis
    df['Description Length'] = df['Description'].str.len()
    df['Has Long Description'] = df['Description Length'] > 500
    
    # Add tag analysis
    df['Tag Count'] = df['Tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Add whether video has captions
    df['Has Captions'] = df['Caption'] == 'true'
    
    # Add HD vs SD indicator
    df['Is HD'] = df['Definition'] == 'hd'
    
    return df

def create_wordcloud(df):
    """Create word cloud from video titles"""
    all_words = []
    for word_list in df['Title Words']:
        if isinstance(word_list, list):
            all_words.extend(word_list)
        elif isinstance(word_list, str):
            try:
                # Try to parse the string as a list
                word_list = eval(word_list)
                if isinstance(word_list, list):
                    all_words.extend(word_list)
            except:
                pass
    
    word_counts = Counter(all_words)
    
    # Create word cloud
    if word_counts:
        wc = WordCloud(width=800, height=400, background_color='white', 
                       max_words=100, colormap='Reds')
        wc.generate_from_frequencies(word_counts)
        
        return wc
    return None

def predict_future_performance(df, days=30):
    """Create simple prediction for views growth"""
    if df.empty or len(df) < 5:
        return None
    
    # Sort by date
    df_sorted = df.sort_values('Published Date')
    
    # Create features based on video number (indicating channel growth over time)
    df_sorted['Video Number'] = range(1, len(df_sorted) + 1)
    
    # Train linear regression model
    X = df_sorted[['Video Number']].values
    y = df_sorted['Views'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future videos
    future_videos = np.array([[len(df_sorted) + i] for i in range(1, days + 1)])
    predictions = model.predict(future_videos)
    
    # Ensure predictions are non-negative
    predictions = np.maximum(predictions, 0)
    
    return predictions

def save_data_to_files(channel_stats, video_data):
    """Save data to CSV files"""
    try:
        if channel_stats:
            channel_df = pd.DataFrame([channel_stats])
            channel_df.to_csv('channel_stats.csv', index=False)
            
        if video_data:
            video_df = pd.DataFrame(video_data)
            video_df.to_csv('video_data.csv', index=False)
            
    except Exception as e:
        st.error(f"Error saving data to files: {e}")

def load_data_from_files():
    """Load data from CSV files if they exist"""
    try:
        if os.path.exists('channel_stats.csv') and os.path.exists('video_data.csv'):
            channel_df = pd.read_csv('channel_stats.csv')
            video_df = pd.read_csv('video_data.csv')
            
            # Convert to expected format
            channel_stats = channel_df.iloc[0].to_dict() if not channel_df.empty else None
            video_data = video_df.to_dict('records') if not video_df.empty else None
            
            return channel_stats, video_data
        return None, None
    except Exception as e:
        st.error(f"Error loading data from files: {e}")
        return None, None

# Create a function to show video thumbnails and links
def display_video_card(video):
    """Display a video card with thumbnail and info"""
    video_url = f"https://www.youtube.com/watch?v={video['Video ID']}"
    
    return f"""
    <div style="display: flex; margin-bottom: 15px; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
        <div style="flex: 0 0 200px;">
            <a href="{video_url}" target="_blank">
                <img src="{video['Thumbnail']}" width="200" style="object-fit: cover;">
            </a>
        </div>
        <div style="flex: 1; padding: 10px;">
            <a href="{video_url}" target="_blank" style="color: #282828; text-decoration: none;">
                <h4 style="margin-top: 0;">{video['Title']}</h4>
            </a>
            <p style="color: #606060; font-size: 0.9em; margin: 5px 0;">
                {pd.to_datetime(video['Published Date']).strftime('%b %d, %Y')} • 
                {int(video['Views']):,} views
            </p>
            <div style="display: flex; margin-top: 8px;">
                <div style="margin-right: 15px;">
                    <span style="color: #606060;">👍</span> {int(video['Likes']):,}
                </div>
                <div>
                    <span style="color: #606060;">💬</span> {int(video['Comments']):,}
                </div>
            </div>
        </div>
    </div>
    """

# Create a date filter function
def filter_data_by_date(df, start_date=None, end_date=None):
    """Filter dataframe by date range"""
    filtered_df = df.copy()
    
    # Ensure Published Date is datetime
    filtered_df['Published Date'] = pd.to_datetime(filtered_df['Published Date'])
    
    # Convert timezone-aware datetime to timezone-naive
    if hasattr(filtered_df['Published Date'].iloc[0], 'tz') and filtered_df['Published Date'].iloc[0].tz is not None:
        filtered_df['Published Date'] = filtered_df['Published Date'].dt.tz_localize(None)
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        filtered_df = filtered_df[filtered_df['Published Date'] >= start_date]
        
    if end_date:
        end_date = pd.to_datetime(end_date)
        filtered_df = filtered_df[filtered_df['Published Date'] <= end_date]
        
    return filtered_df

# Main app
def main():
    st.markdown("<h1 class='main-header'>🎥 YouTube Analytics Pro</h1>", unsafe_allow_html=True)
    
    # Initialize session state for storing data
    if 'channel_stats' not in st.session_state:
        st.session_state.channel_stats = None
    if 'video_data' not in st.session_state:
        st.session_state.video_data = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e1/Logo_of_YouTube_%282015-2017%29.svg", width=200)
        st.title("Channel Analysis")
        
        # Input options
        option = st.radio("Choose Input Method", 
                         ["Enter Channel Details", "Use Sample Data"])
        
        if option == "Enter Channel Details":
            st.markdown("### Channel Information")
        
            # Method selector
            input_method = st.radio("Choose method", ["Search by Name", "Enter ID"])
            
            channel_id = None
            
            if input_method == "Search by Name":
                # Search input with improved UX
                st.markdown('<div class="search-container">', unsafe_allow_html=True)
                channel_name = st.text_input(
                    "Channel Name",
                    placeholder="Enter channel name...",
                    key="channel_name_input",
                    help="Enter the name of the YouTube channel you want to analyze"
                )
                
                # Search button with improved styling
                if st.button(
                    "🔍 Search Channel",
                    key="search_btn",
                    help="Click to search for the channel",
                    type="primary"
                ):
                    if channel_name:
                        with st.spinner("Searching for channels..."):
                            st.session_state.search_results = search_and_select_channel(API_KEY, channel_name)
                            if not st.session_state.search_results:
                                st.error("No channels found. Try a different search term.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display search results with improved styling
                if st.session_state.search_results:
                    st.markdown('<div class="channel-select">', unsafe_allow_html=True)
                    # Create a dictionary for the selectbox
                    channel_options = {f"{r['Channel Name']} ({r['Channel ID']})": r['Channel ID'] 
                                    for r in st.session_state.search_results}
                    channel_options = list(channel_options.items())
                    
                    # Add a "Select channel" placeholder as the first option
                    channel_options.insert(0, ("Select a channel", ""))
                    
                    # Create the selectbox with improved styling
                    selected_channel = st.selectbox(
                        "📺 Select Channel",
                        options=[opt[0] for opt in channel_options],
                        format_func=lambda x: x,
                        key="channel_select"
                    )
                    
                    # Get the selected channel ID
                    selected_index = [opt[0] for opt in channel_options].index(selected_channel) if selected_channel else 0
                    if selected_index > 0:  # If a real channel is selected (not the placeholder)
                        channel_id = channel_options[selected_index][1]
                        
                        # Display channel info with improved styling
                        selected_channel_info = next((r for r in st.session_state.search_results 
                                                    if r['Channel ID'] == channel_id), None)
                        if selected_channel_info:
                            st.markdown('<div class="channel-info-card">', unsafe_allow_html=True)
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.image(selected_channel_info['Thumbnail'], width=80)
                            with col2:
                                st.markdown(f"**{selected_channel_info['Channel Name']}**")
                                st.markdown(f"*{selected_channel_info['Description'][:100]}...*" 
                                          if len(selected_channel_info['Description']) > 100 
                                          else f"*{selected_channel_info['Description']}*")
                            st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Manual ID input
                channel_id = st.text_input("Enter Channel ID", value="UC_x5XG1OV2P6uZZ5FSM9Ttw", placeholder="e.g., UC_x5XG1OV2P6uZZ5")
        
            # Fetch data button with improved styling
            fetch_col1, fetch_col2 = st.columns([1, 1])
            with fetch_col1:
                fetch_button = st.button("📊 Fetch Data", key="fetch", use_container_width=True)
            with fetch_col2:
                reset_button = st.button("🔄 Reset", key="reset", use_container_width=True)
                
            if fetch_button:
                if channel_id:
                    with st.spinner("Fetching data from YouTube API..."):
                        channel_stats, video_data = fetch_youtube_data(API_KEY, channel_id)
                        if channel_stats and video_data:
                            st.session_state.channel_stats = channel_stats
                            st.session_state.video_data = video_data
                            save_data_to_files(channel_stats, video_data)
                            st.success("✅ Data fetched successfully!")
                        else:
                            st.error("❌ Failed to fetch data. Please check the channel ID and try again.")
                else:
                    st.error("Please select a channel or enter a Channel ID")
            
            if reset_button:
                st.session_state.channel_stats = None
                st.session_state.video_data = None
                st.session_state.search_results = []
                st.success("Data reset successfully")
                st.experimental_rerun()
        
        elif option == "Use Sample Data":
            # Load data from files
            st.info("Loading saved data...")
            channel_stats, video_data = load_data_from_files()
            if channel_stats and video_data:
                st.session_state.channel_stats = channel_stats
                st.session_state.video_data = video_data
                st.success("Sample data loaded successfully!")
            else:
                st.error("No sample data available. Please fetch data first.")
        
        # Add date filter in sidebar if data is available
        if st.session_state.video_data:
            st.markdown("### 📅 Date Filter")
            
            # Convert video data to dataframe for date filtering
            df = pd.DataFrame(st.session_state.video_data)
            df['Published Date'] = pd.to_datetime(df['Published Date'])
            
            min_date = df['Published Date'].min().date()
            max_date = df['Published Date'].max().date()
            
            # Create date filters
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
            
            if start_date > end_date:
                st.error("Start date must be before end date")

    # Process and display data if available
    if st.session_state.channel_stats and st.session_state.video_data:
        # Convert video data to DataFrame
        video_df = pd.DataFrame(st.session_state.video_data)
        
        # Apply date filtering if set
        if 'start_date' in locals() and 'end_date' in locals():
            video_df['Published Date'] = pd.to_datetime(video_df['Published Date'])
            video_df = filter_data_by_date(video_df, start_date, end_date)
        
        # Process data
        enhanced_df = enhance_video_data(video_df)
        enhanced_channel = enhance_channel_data(st.session_state.channel_stats, enhanced_df)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Channel Overview", 
            "🎬 Video Analysis", 
            "📝 Publishing Strategy", 
            "🔍 Content Analytics",
            "⭐ Recommendations"
        ])
        
        with tab1:
            st.markdown("<h2 class='sub-header'>Channel Overview</h2>", unsafe_allow_html=True)
            
            # Channel banner if available
            if enhanced_channel.get('Banner URL'):
                st.image(enhanced_channel['Banner URL'], use_container_width=True)
            
            # Channel header with thumbnail
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                # Check if we have thumbnail info from search results
                channel_thumb = next(
                    (r['Thumbnail'] for r in st.session_state.search_results 
                     if r['Channel ID'] == enhanced_channel['Channel ID']),
                    "https://via.placeholder.com/100"
                )
                st.image(channel_thumb, width=100)
            
            with col2:
                st.markdown(f"### {enhanced_channel['Channel Name']}")
                if enhanced_channel.get('Custom URL'):
                    st.write(f"Custom URL: {enhanced_channel['Custom URL']}")
                st.write(f"Created: {pd.to_datetime(enhanced_channel['Channel Created']).strftime('%b %d, %Y')}")
            
            with col3:
                st.write("")
                st.button("🔗 Channel Link", 
                          help=f"https://youtube.com/channel/{enhanced_channel['Channel ID']}")
            
            # Description if available (truncated)
            if enhanced_channel.get('Description'):
                with st.expander("View Channel Description"):
                    st.write(enhanced_channel['Description'])
            
            # Channel metrics
            st.markdown("<h3 class='sub-header'>Channel Metrics</h3>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    f"""<div class="metric-card">
                    <h4 style="margin:0;color:#606060;">Subscribers</h4>
                    <p style="font-size:24px;font-weight:bold;margin:5px 0;">{int(enhanced_channel['Subscribers']):,}</p>
                    <p style="margin:0;color:#606060;font-size:14px;">~{enhanced_channel['Subs per Day']} per day</p>
                    </div>""", unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""<div class="metric-card">
                    <h4 style="margin:0;color:#606060;">Total Views</h4>
                    <p style="font-size:24px;font-weight:bold;margin:5px 0;">{int(enhanced_channel['Views']):,}</p>
                    <p style="margin:0;color:#606060;font-size:14px;">~{enhanced_channel['Views per Day']} per day</p>
                    </div>""", unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f"""<div class="metric-card">
                    <h4 style="margin:0;color:#606060;">Total Videos</h4>
                    <p style="font-size:24px;font-weight:bold;margin:5px 0;">{int(enhanced_channel['Total Videos']):,}</p>
                    <p style="margin:0;color:#606060;font-size:14px;">~{enhanced_channel['Videos per Month']} per month</p>
                    </div>""", unsafe_allow_html=True
                )
            
            with col4:
                # Add view trend indicator
                trend_icon = "↗️" if enhanced_channel.get('View Trend', 0) >= 0 else "↘️"
                trend_color = "#00cc66" if enhanced_channel.get('View Trend', 0) >= 0 else "#ff3366"
                
                st.markdown(
                    f"""<div class="metric-card">
                    <h4 style="margin:0;color:#606060;">Avg Views/Video</h4>
                    <p style="font-size:24px;font-weight:bold;margin:5px 0;">{int(enhanced_channel['Views per Video']):,}</p>
                    <p style="margin:0;color:{trend_color};font-size:14px;">{trend_icon} {abs(enhanced_channel.get('View Trend', 0)):.1f}% in 90 days</p>
                    </div>""", unsafe_allow_html=True
                )
            
            # Channel performance metrics
            st.markdown("<h3 class='sub-header'>Performance Metrics</h3>", unsafe_allow_html=True)
            
            # More detailed metrics with improved visuals
            col1, col2 = st.columns(2)
            
            with col1:
                # Create gauge chart for engagement rate
                engagement_rate = enhanced_channel.get('Engagement Rate', 0)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = engagement_rate,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Engagement Rate (%)", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 10], 'tickwidth': 1},
                        'bar': {'color': "#FF0000"},
                        'steps': [
                            {'range': [0, 3], 'color': '#FFCCCC'},
                            {'range': [3, 5], 'color': '#FFAA99'},
                            {'range': [5, 10], 'color': '#FF8866'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': engagement_rate
                        }
                    }
                ))
                
                fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=30))
                st.plotly_chart(fig, use_container_width=True)
                
                # Add context
                if engagement_rate < 3:
                    st.info("Engagement rate is relatively low. Consider strategies to increase audience interaction.")
                elif engagement_rate < 5:
                    st.success("Engagement rate is average. There's room for improvement.")
                else:
                    st.success("Excellent engagement rate! Your audience is highly interactive.")
            
            with col2:
                # Subscribers to views ratio (how many subscribers relative to views)
                subs_to_views = enhanced_channel.get('Subs to Views Ratio', 0)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = subs_to_views,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    number = {'suffix': '%'},
                    title = {'text': "Subscribers/Views Ratio", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [0, 20], 'tickwidth': 1},
                        'bar': {'color': "#FF0000"},
                        'steps': [
                            {'range': [0, 5], 'color': '#FFCCCC'},
                            {'range': [5, 10], 'color': '#FFAA99'},
                            {'range': [10, 20], 'color': '#FF8866'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': subs_to_views
                        }
                    }
                ))
                
                fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=30))
                st.plotly_chart(fig, use_container_width=True)
                
                # Add context
                if subs_to_views < 2:
                    st.info("Your subscriber conversion rate is low. Focus on encouraging viewers to subscribe.")
                elif subs_to_views < 5:
                    st.success("Average subscriber conversion. Consider adding stronger calls-to-action.")
                else:
                    st.success("Excellent subscriber conversion! Your content effectively turns viewers into subscribers.")
            
            # Views Distribution
            st.markdown("<h3 class='sub-header'>Views Distribution</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(enhanced_df, x='Views', nbins=30, 
                                  title='Distribution of Video Views',
                                  color_discrete_sequence=['#FF0000'])
                fig.update_layout(xaxis_title='Views', yaxis_title='Number of Videos')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Performance quartiles
                performance_counts = enhanced_df['Performance'].value_counts().reset_index()
                performance_counts.columns = ['Performance Level', 'Number of Videos']
                
                # Order correctly
                perf_order = ['Low', 'Medium', 'High', 'Top']
                performance_counts['Performance Level'] = pd.Categorical(
                    performance_counts['Performance Level'], 
                    categories=perf_order, 
                    ordered=True
                )
                performance_counts = performance_counts.sort_values('Performance Level')
                
                fig = px.pie(performance_counts, values='Number of Videos', names='Performance Level', 
                           title='Video Performance Categories',
                           color='Performance Level',
                           color_discrete_map={
                               'Low': '#FFCCCC',
                               'Medium': '#FFAA99', 
                               'High': '#FF8866', 
                               'Top': '#FF0000'
                           })
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(legend_title="Performance Level")
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance over time
            st.markdown("<h3 class='sub-header'>Performance Over Time</h3>", unsafe_allow_html=True)
            
            # Add time-based filters
            time_view_option = st.radio(
                "View",
                ["Cumulative Growth", "Monthly Performance", "Video-by-Video"],
                horizontal=True
            )
            
            if time_view_option == "Cumulative Growth":
                time_df = enhanced_df.sort_values('Published Date')
                time_df['Cumulative Views'] = time_df['Views'].cumsum()
                time_df['Cumulative Likes'] = time_df['Likes'].cumsum()
                time_df['Video Number'] = range(1, len(time_df) + 1)
                
                # Create figure with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add cumulative views
                fig.add_trace(
                    go.Scatter(
                        x=time_df['Published Date'], 
                        y=time_df['Cumulative Views'],
                        name="Cumulative Views",
                        line=dict(color="#FF0000", width=3)
                    ),
                    secondary_y=False,
                )
                
                # Add cumulative likes
                fig.add_trace(
                    go.Scatter(
                        x=time_df['Published Date'], 
                        y=time_df['Cumulative Likes'],
                        name="Cumulative Likes",
                        line=dict(color="#0080FF", width=2, dash='dot')
                    ),
                    secondary_y=True,
                )
                
                # Set titles
                fig.update_layout(
                    title="Cumulative Growth Over Time",
                    xaxis_title="Date",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                
                # Set y-axes titles
                fig.update_yaxes(title_text="Cumulative Views", secondary_y=False)
                fig.update_yaxes(title_text="Cumulative Likes", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add future prediction if enough data
                if len(enhanced_df) >= 5:
                    with st.expander("View Growth Forecast"):
                        # Predict future views
                        future_predictions = predict_future_performance(enhanced_df, days=30)
                        
                        if future_predictions is not None:
                            # Create prediction dataframe
                            last_video_num = len(enhanced_df)
                            last_date = enhanced_df['Published Date'].max()
                            
                            # Assume consistent publishing schedule
                            avg_days_between = enhanced_df['Days Since Published'].min()
                            if avg_days_between < 1:
                                avg_days_between = 7  # Default to weekly if not enough data
                                
                            future_dates = [last_date + timedelta(days=int(avg_days_between * i)) 
                                          for i in range(1, len(future_predictions) + 1)]
                            
                            pred_df = pd.DataFrame({
                                'Date': future_dates,
                                'Video Number': [last_video_num + i for i in range(1, len(future_predictions) + 1)],
                                'Predicted Views': future_predictions
                            })
                            
                            # Plot prediction
                            fig = px.line(
                                pred_df, x='Date', y='Predicted Views',
                                title='30-Day View Forecast for New Videos',
                                markers=True
                            )
                            fig.update_layout(xaxis_title='Date', yaxis_title='Predicted Views')
                            
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("This is a simple linear forecast based on channel history. Actual results may vary.")
                        else:
                            st.info("Not enough data for forecasting.")
            
            elif time_view_option == "Monthly Performance":
                # Group by month and calculate metrics
                month_df = enhanced_df.copy()
                month_df['YearMonth'] = month_df['Published Date'].dt.to_period('M')
                
                # Fix the groupby operations with observed=True
                monthly_stats = month_df.groupby('YearMonth', observed=True).agg(
                    Videos=('Video ID', 'count'),
                    Views=('Views', 'sum'),
                    Likes=('Likes', 'sum'),
                    Comments=('Comments', 'sum'),
                    Avg_Views=('Views', 'mean'),
                    Avg_Engagement=('Engagement Rate', 'mean')
                ).reset_index()
                
                # Convert Period to datetime for Plotly
                monthly_stats['Date'] = monthly_stats['YearMonth'].dt.to_timestamp()
                
                # Create a time series chart
                fig = px.line(
                    monthly_stats, x='Date', y=['Avg_Views', 'Avg_Engagement'],
                    title='Monthly Performance Trends',
                    labels={'Date': 'Month', 'value': 'Value', 'variable': 'Metric'},
                    color_discrete_map={
                        'Avg_Views': '#FF0000',
                        'Avg_Engagement': '#0080FF'
                    }
                )
                
                fig.update_layout(legend_title="Metrics", xaxis_title="Month")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show monthly stats table
                with st.expander("View Monthly Stats Table"):
                    monthly_stats['Month'] = monthly_stats['Date'].dt.strftime('%b %Y')
                    display_df = monthly_stats[['Month', 'Videos', 'Views', 'Avg_Views', 'Avg_Engagement']]
                    display_df.columns = ['Month', 'Videos Published', 'Total Views', 'Avg Views/Video', 'Avg Engagement Rate (%)']
                    
                    # Format numbers
                    display_df['Total Views'] = display_df['Total Views'].map('{:,.0f}'.format)
                    display_df['Avg Views/Video'] = display_df['Avg Views/Video'].map('{:,.0f}'.format)
                    display_df['Avg Engagement Rate (%)'] = display_df['Avg Engagement Rate (%)'].map('{:.2f}'.format)
                    
                    st.table(display_df)
            
            else:  # Video-by-Video
                # Sort videos by date
                video_time_df = enhanced_df.sort_values('Published Date')
                video_time_df['Video Number'] = range(1, len(video_time_df) + 1)
                
                # Create chart
                fig = px.scatter(
                    video_time_df, x='Published Date', y='Views',
                    size='Engagement Rate', color='Performance',
                    hover_name='Title',
                    size_max=30,
                    title='Individual Video Performance Over Time',
                    color_discrete_map={
                        'Low': '#FFCCCC',
                        'Medium': '#FFAA99', 
                        'High': '#FF8866', 
                        'Top': '#FF0000'
                    }
                )
                
                # Add trendline
                fig.update_layout(xaxis_title='Publish Date', yaxis_title='Views')
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                
                # Add trendline
                video_time_df['date_ordinal'] = pd.to_datetime(video_time_df['Published Date']).map(datetime.toordinal)
                z = np.polyfit(video_time_df['date_ordinal'], video_time_df['Views'], 1)
                p = np.poly1d(z)
                
                fig.add_trace(
                    go.Scatter(
                        x=video_time_df['Published Date'],
                        y=p(video_time_df['date_ordinal']),
                        mode='lines',
                        name='Trend',
                        line=dict(color='black', dash='dash')
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show trend analysis
                slope = z[0]
                if slope > 0:
                    st.success("📈 Your video views are trending upward over time, indicating growing channel performance.")
                else:
                    st.info("📉 Your video views are trending downward or flat. Consider reviewing your content strategy.")
        
        with tab2:
            st.markdown("<h2 class='sub-header'>Video Analysis</h2>", unsafe_allow_html=True)
            
            # Top videos section
            st.markdown("<h3 class='sub-header'>Top Performing Videos</h3>", unsafe_allow_html=True)
            
            metric_choice = st.radio(
                "Rank by:",
                ["Views", "Engagement Rate", "Likes", "Comments", "Daily Views"],
                horizontal=True
            )
            
            # Top 10 videos
            top_videos = enhanced_df.sort_values(metric_choice, ascending=False).head(10)
            
            # Create horizontal bar chart
            fig = px.bar(top_videos, y='Title', x=metric_choice, orientation='h',
                        title=f'Top 10 Videos by {metric_choice}',
                        color=metric_choice,
                        color_continuous_scale='Reds')
            
            fig.update_layout(
                yaxis_title='', 
                xaxis_title=metric_choice,
                height=500,
                yaxis={'categoryorder':'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display top videos with thumbnails
            st.markdown("<h4>Top Videos Details</h4>", unsafe_allow_html=True)
            for i, video in enumerate(top_videos.to_dict('records')[:5]):
                st.markdown(display_video_card(video), unsafe_allow_html=True)
            
            # Video comparison scatter plot
            st.markdown("<h3 class='sub-header'>Video Comparison</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_metric = st.selectbox("X-Axis Metric", 
                                      ["Views", "Likes", "Comments", "Duration (min)", 
                                       "Engagement Rate", "Title Length", "Days Since Published"],
                                      index=0)
                
            with col2:
                y_metric = st.selectbox("Y-Axis Metric", 
                                      ["Views", "Likes", "Comments", "Duration (min)", 
                                       "Engagement Rate", "Title Length", "Days Since Published"],
                                      index=4)
                
            # Create scatter plot with improved styling
            fig = px.scatter(
                enhanced_df, x=x_metric, y=y_metric,
                color='Performance',
                size='Relative Performance',
                hover_name='Title',
                size_max=25,
                color_discrete_map={
                    'Low': '#FFCCCC',
                    'Medium': '#FFAA99', 
                    'High': '#FF8866', 
                    'Top': '#FF0000'
                }
            )
            
            fig.update_layout(
                title=f'{x_metric} vs. {y_metric}',
                xaxis_title=x_metric,
                yaxis_title=y_metric,
                legend_title="Performance"
            )
            
            # Add trendline
            fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Duration analysis
            st.markdown("<h3 class='sub-header'>Video Duration Analysis</h3>", unsafe_allow_html=True)
            
            # Fix the duration analysis groupby
            duration_df = enhanced_df.groupby('Duration Category', observed=True).agg(
                Count=('Video ID', 'count'),
                Avg_Views=('Views', 'mean'),
                Avg_Likes=('Likes', 'mean'),
                Avg_Comments=('Comments', 'mean'),
                Avg_Engagement=('Engagement Rate', 'mean')
            ).reset_index()
            
            # Reorder categories
            duration_order = ['< 1 min', '1-5 min', '5-10 min', '10-20 min', '> 20 min']
            duration_df['Duration Category'] = pd.Categorical(
                duration_df['Duration Category'],
                categories=duration_order,
                ordered=True
            )
            duration_df = duration_df.sort_values('Duration Category')
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    duration_df, 
                    x='Duration Category', 
                    y='Count',
                    title='Number of Videos by Duration',
                    text_auto=True,
                    color='Duration Category',
                    color_discrete_map={
                        '< 1 min': '#FFCCCC',
                        '1-5 min': '#FFAA99', 
                        '5-10 min': '#FF8866', 
                        '10-20 min': '#FF5533', 
                        '> 20 min': '#FF0000'
                    }
                )
                
                fig.update_layout(
                    xaxis_title='Duration Category',
                    yaxis_title='Number of Videos',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Let user choose which metric to display
                duration_metric = st.selectbox(
                    "Choose metric:",
                    ["Avg_Views", "Avg_Engagement", "Avg_Likes", "Avg_Comments"],
                    format_func=lambda x: x.replace("Avg_", "Average ").replace("_", " ")
                )
                
                metric_title = duration_metric.replace("Avg_", "Average ").replace("_", " ")
                
                fig = px.bar(
                    duration_df, 
                    x='Duration Category', 
                    y=duration_metric,
                    title=f'{metric_title} by Duration',
                    text_auto=True,
                    color='Duration Category',
                    color_discrete_map={
                        '< 1 min': '#FFCCCC',
                        '1-5 min': '#FFAA99', 
                        '5-10 min': '#FF8866', 
                        '10-20 min': '#FF5533', 
                        '> 20 min': '#FF0000'
                    }
                )
                
                fig.update_layout(
                    xaxis_title='Duration Category',
                    yaxis_title=metric_title,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Optimal duration analysis
            best_duration = duration_df.iloc[duration_df['Avg_Views'].argmax()]['Duration Category']
            highest_engagement = duration_df.iloc[duration_df['Avg_Engagement'].argmax()]['Duration Category']
            
            st.info(f"📊 Your highest performing videos are in the **{best_duration}** duration category (by views) and **{highest_engagement}** category (by engagement).")
            
            # Video growth analysis
            st.markdown("<h3 class='sub-header'>Video Growth Analysis</h3>", unsafe_allow_html=True)
            
            # Create dataframe with relative days since publishing
            recent_videos = enhanced_df[enhanced_df['Days Since Published'] <= 90].copy()
            if not recent_videos.empty:
                # Group by days since published (buckets)
                recent_videos['Days Bucket'] = pd.cut(
                    recent_videos['Days Since Published'],
                    bins=[0, 1, 7, 14, 30, 60, 90],
                    labels=['First day', '2-7 days', '8-14 days', '15-30 days', '31-60 days', '61-90 days']
                )
                
                growth_df = recent_videos.groupby('Days Bucket').agg(
                    Avg_Views=('Views', 'mean'),
                    Avg_Daily_Views=('Daily Views', 'mean'),
                    Avg_Engagement=('Engagement Rate', 'mean'),
                    Count=('Video ID', 'count')
                ).reset_index()
                
                # Plot growth over time
                fig = px.line(
                    growth_df, 
                    x='Days Bucket', 
                    y='Avg_Views',
                    title='Average View Growth Pattern',
                    markers=True,
                    text='Avg_Views'
                )
                
                fig.update_traces(
                    texttemplate='%{text:.0f}',
                    textposition='top center',
                    line=dict(color='#FF0000', width=3)
                )
                
                fig.update_layout(
                    xaxis_title='Days Since Publishing',
                    yaxis_title='Average Views',
                    title_font_size=20,
                    xaxis=dict(
                        tickmode='linear'
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    title_x=0.5
                )

                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown("<h2 class='sub-header'>Publishing Strategy</h2>", unsafe_allow_html=True)
            
            # Upload frequency analysis
            st.markdown("<h3 class='sub-header'>Upload Frequency Analysis</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Day of week analysis
                day_counts = enhanced_df['Publish Day of Week'].value_counts()
                
                fig = px.bar(
                    x=day_counts.index,
                    y=day_counts.values,
                    title='Upload Distribution by Day of Week',
                    labels={'x': 'Day of Week', 'y': 'Number of Videos'},
                    color=day_counts.values,
                    color_continuous_scale='Reds'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Find best performing day
                day_performance = enhanced_df.groupby('Publish Day of Week')['Views'].mean().sort_values(ascending=False)
                best_day = day_performance.index[0]
                st.info(f"📅 Your videos published on **{best_day}** tend to perform best.")
            
            with col2:
                # Hour of day analysis
                hour_counts = enhanced_df['Hour Published'].value_counts().sort_index()
                
                fig = px.line(
                    x=hour_counts.index,
                    y=hour_counts.values,
                    title='Upload Distribution by Hour of Day',
                    labels={'x': 'Hour of Day (24h)', 'y': 'Number of Videos'},
                    markers=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Find best performing hour
                hour_performance = enhanced_df.groupby('Hour Published')['Views'].mean().sort_values(ascending=False)
                best_hour = hour_performance.index[0]
                st.info(f"⏰ Your videos published at **{best_hour:02d}:00** tend to perform best.")
            
            # Upload consistency
            st.markdown("<h3 class='sub-header'>Upload Consistency</h3>", unsafe_allow_html=True)
            
            # Calculate days between uploads
            upload_dates = pd.to_datetime(enhanced_df['Published Date']).sort_values()
            days_between = upload_dates.diff().dt.days
            
            fig = px.histogram(
                x=days_between.dropna(),
                nbins=20,
                title='Distribution of Days Between Uploads',
                labels={'x': 'Days Between Uploads', 'y': 'Frequency'},
                color_discrete_sequence=['#FF0000']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Fix the upload consistency calculation with proper error handling
            if not days_between.empty:
                median_days = days_between.median()
                if days_between.mean() != 0:  # Avoid division by zero
                    consistency = days_between.std() / days_between.mean()
                else:
                    consistency = float('inf')  # Handle case where mean is 0

            # Fix the Has Question analysis with proper error handling
            question_views = enhanced_df.groupby('Has Question', observed=True)['Views'].mean()
            if not question_views.empty:
                has_question_views = question_views.get(True, 0)
                no_question_views = question_views.get(False, 0)
                
                if has_question_views > no_question_views:
                    recommendations.append("❓ Using questions in your titles appears to increase views. Consider incorporating more question-based titles.")

            # Upload consistency insights
            if consistency < 0.5:
                consistency_msg = "Your upload schedule is very consistent! 👏"
            elif consistency < 1.0:
                consistency_msg = "Your upload schedule is moderately consistent."
            else:
                consistency_msg = "Your upload schedule could be more consistent."
                
            st.info(f"📊 You typically upload every **{median_days:.1f}** days. {consistency_msg}")

        with tab4:
            st.markdown("<h2 class='sub-header'>Content Analytics</h2>", unsafe_allow_html=True)
            
            # Title Analysis
            st.markdown("<h3 class='sub-header'>Title Analysis</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Title length analysis
                fig = px.histogram(
                    enhanced_df,
                    x='Title Length',
                    nbins=30,
                    title='Distribution of Title Lengths',
                    color_discrete_sequence=['#FF0000']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Title length vs views correlation
                correlation = enhanced_df['Title Length'].corr(enhanced_df['Views'])
                optimal_length = enhanced_df.loc[enhanced_df['Views'].idxmax(), 'Title Length']
                st.info(f"📏 Optimal title length appears to be around **{optimal_length}** characters.")
            
            with col2:
                # Word cloud of titles
                wc = create_wordcloud(enhanced_df)
                if wc:
                    # Convert word cloud to image
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wc, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt, use_container_width=True)
            
            # Title features analysis
            st.markdown("<h3 class='sub-header'>Title Features Impact</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Question marks impact
                question_views = enhanced_df.groupby('Has Question', observed=True)['Views'].mean()
                if not question_views.empty:
                    # Create a proper DataFrame for plotting
                    plot_data = pd.DataFrame({
                        'Question Type': ['No Question', 'Has Question'],
                        'Average Views': [
                            question_views.get(False, 0),
                            question_views.get(True, 0)
                        ]
                    })
                    
                    fig = px.bar(
                        plot_data,
                        x='Question Type',
                        y='Average Views',
                        title='Average Views: Questions in Title',
                        color_discrete_sequence=['#FF0000']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if question_views.get(True, 0) > question_views.get(False, 0):
                        recommendations.append("❓ Using questions in your titles appears to increase views. Consider incorporating more question-based titles.")
            
            with col2:
                # Numbers impact
                number_views = enhanced_df.groupby('Has Number')['Views'].mean().reset_index()
                number_views.columns = ['Has Number', 'Average Views']
                number_views['Has Number'] = number_views['Has Number'].map({False: 'No Number', True: 'Has Number'})
                
                fig = px.bar(
                    number_views,
                    x='Has Number',
                    y='Average Views',
                    title='Average Views: Numbers in Title',
                    color_discrete_sequence=['#FF0000']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                # Emoji impact
                emoji_views = enhanced_df.groupby('Has Emoji')['Views'].mean().reset_index()
                emoji_views.columns = ['Has Emoji', 'Average Views']
                emoji_views['Has Emoji'] = emoji_views['Has Emoji'].map({False: 'No Emoji', True: 'Has Emoji'})
                
                fig = px.bar(
                    emoji_views,
                    x='Has Emoji',
                    y='Average Views',
                    title='Average Views: Emojis in Title',
                    color_discrete_sequence=['#FF0000']
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab5:
            st.markdown("<h2 class='sub-header'>Recommendations</h2>", unsafe_allow_html=True)
            
            # Calculate key metrics for recommendations
            avg_views = enhanced_df['Views'].mean()
            top_videos = enhanced_df.nlargest(5, 'Views')
            common_duration = enhanced_df['Duration Category'].mode().iloc[0]
            best_performing_duration = enhanced_df.groupby('Duration Category')['Views'].mean().idxmax()
            
            # Content Strategy Recommendations
            st.markdown("<h3 class='sub-header'>Content Strategy Recommendations</h3>", unsafe_allow_html=True)
            
            recommendations = []
            
            # Duration recommendations
            if common_duration != best_performing_duration:
                recommendations.append(f"🎬 Consider creating more videos in the **{best_performing_duration}** range, as these tend to perform better than your usual **{common_duration}** videos.")
            
            # Title recommendations - Fix KeyError
            question_views = enhanced_df.groupby('Has Question')['Views'].mean()
            if True in question_views.index and False in question_views.index and question_views.get(True) > question_views.get(False):
                recommendations.append("❓ Using questions in your titles appears to increase views. Consider incorporating more question-based titles.")
            
            number_views = enhanced_df.groupby('Has Number')['Views'].mean()
            if True in number_views.index and False in number_views.index and number_views.get(True) > number_views.get(False):
                recommendations.append("📊 Videos with numbers in the title perform well. Try including more numbered lists or statistics.")
            
            # Engagement recommendations
            if enhanced_channel['Engagement Rate'] < 5:
                recommendations.append("👥 Work on increasing engagement by asking viewers questions and encouraging comments.")
            
            # Upload consistency
            if days_between.std() / days_between.mean() > 0.5:
                recommendations.append("📅 Your upload schedule could be more consistent. Try to maintain a regular publishing schedule.")
            
            # Display recommendations
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
            
            # Success Patterns
            st.markdown("<h3 class='sub-header'>Success Patterns</h3>", unsafe_allow_html=True)
            
            st.markdown("#### 🏆 Your Most Successful Videos")
            for _, video in top_videos.iterrows():
                st.markdown(
                    f"""
                    - **{video['Title']}**
                    - Views: {video['Views']:,.0f}
                    - Duration: {video['Duration Category']}
                    - Engagement Rate: {video['Engagement Rate']:.1f}%
                    """
                )
            
            # Growth Opportunities
            st.markdown("<h3 class='sub-header'>Growth Opportunities</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Subscriber conversion
                if enhanced_channel['Subs to Views Ratio'] < 5:
                    st.warning("🎯 Subscriber Conversion")
                    st.markdown("""
                    Your subscriber to view ratio could be improved:
                    - Add clear calls-to-action
                    - Highlight channel value proposition
                    - Create more series-based content
                    """)
            
            with col2:
                # Engagement improvement
                if enhanced_channel['Engagement Rate'] < 5:
                    st.warning("💬 Engagement Improvement")
                    st.markdown("""
                    Boost your engagement rates by:
                    - Asking questions in videos
                    - Creating topical content
                    - Responding to comments
                    """)

if __name__ == "__main__":
    main()
