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
from wordcloud import WordCloud
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Set page configuration
st.set_page_config(
    page_title="YouTube Channel Analysis",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF0000;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #606060;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stPlotlyChart {
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: white;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def fetch_youtube_data(api_key, channel_id):
    """Fetch data from YouTube API"""
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Get channel information
        channel_response = youtube.channels().list(
            part="snippet,contentDetails,statistics",
            id=channel_id
        ).execute()
        
        if not channel_response['items']:
            st.error(f"Channel with ID {channel_id} not found.")
            return None, None
        
        channel_data = channel_response['items'][0]
        
        # Extract relevant channel data
        channel_stats = {
            'Channel Name': channel_data['snippet']['title'],
            'Channel ID': channel_id,
            'Subscribers': int(channel_data['statistics'].get('subscriberCount', 0)),
            'Views': int(channel_data['statistics'].get('viewCount', 0)),
            'Total Videos': int(channel_data['statistics'].get('videoCount', 0)),
            'Channel Created': channel_data['snippet']['publishedAt'],
            'Playlist ID': channel_data['contentDetails']['relatedPlaylists']['uploads']
        }
        
        # Get video IDs from uploads playlist
        video_ids = []
        next_page_token = None
        
        # Limit to 100 videos to avoid quota issues
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
                    
            # Check if we've reached our limit or there are more pages
            if len(video_ids) >= max_videos:
                break
                
            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break
                
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
                    'Views': int(video['statistics'].get('viewCount', 0)),
                    'Likes': int(video['statistics'].get('likeCount', 0)),
                    'Comments': int(video['statistics'].get('commentCount', 0))
                }
                
                all_video_data.append(video_data)
                
        return channel_stats, all_video_data
        
    except HttpError as e:
        st.error(f"An HTTP error occurred: {e}")
        return None, None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None

def enhance_video_data(video_data):
    """Process and enhance video data"""
    # Convert to DataFrame
    df = pd.DataFrame(video_data)
    
    # Convert date strings to datetime objects
    df['Published Date'] = pd.to_datetime(df['Published Date'])
    
    # Process duration (convert ISO 8601 duration to seconds)
    df['Duration (sec)'] = df['Duration'].apply(lambda x: isodate.parse_duration(x).total_seconds())
    
    # Add duration categories
    df['Duration Category'] = pd.cut(
        df['Duration (sec)'], 
        bins=[0, 300, 600, 1200, 1800, 3600, float('inf')],
        labels=['< 5 min', '5-10 min', '10-20 min', '20-30 min', '30-60 min', '> 60 min']
    )
    
    # Calculate days since publishing - With timezone handling
    df['Days Since Published'] = (datetime.now() - df['Published Date'].dt.tz_localize(None)).dt.days
    
    # Calculate daily views and growth metrics
    df['Daily Views'] = df['Views'] / df['Days Since Published'].replace(0, 1)  # Avoid division by zero
    df['Daily Views'] = df['Daily Views'].fillna(0).round(2)
    
    # Calculate engagement metrics
    df['Engagement Rate'] = ((df['Likes'] + df['Comments']) / df['Views'] * 100).round(2)
    df['Likes per View'] = (df['Likes'] / df['Views'] * 100).round(2)
    df['Comments per View'] = (df['Comments'] / df['Views'] * 100).round(2)
    
    # Time-based metrics
    df['Hour Published'] = df['Published Date'].dt.hour
    df['Time Category'] = pd.cut(
        df['Hour Published'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']
    )
    df['Publish Day of Week'] = df['Published Date'].dt.day_name()
    df['Publish Year'] = df['Published Date'].dt.year
    df['Publish Month'] = df['Published Date'].dt.month
    df['Publish Day'] = df['Published Date'].dt.day
    df['Year-Month'] = df['Published Date'].dt.strftime('%Y-%m')
    
    # Extract title features
    df['Title Length'] = df['Title'].apply(len)
    df['Title Word Count'] = df['Title'].apply(lambda x: len(str(x).split()))
    
    # Add question mark detection
    df['Has Question'] = df['Title'].apply(lambda x: '?' in str(x))
    
    # Extract common words from titles (excluding stopwords)
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'is', 'are', 'how', 'what', 'why', 'when', 'where', 'who'}
    df['Title Words'] = df['Title'].apply(lambda x: [word.lower() for word in re.findall(r'\w+', str(x)) if word.lower() not in stopwords])
    
    # Performance categorization
    df['Performance'] = pd.qcut(df['Views'], q=4, labels=['Low', 'Medium', 'High', 'Viral'])
    
    return df

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
                       max_words=100, colormap='viridis')
        wc.generate_from_frequencies(word_counts)
        
        return wc
    return None

# Main app
def main():
    st.markdown("<h1 class='main-header'>🎥 YouTube Channel Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e1/Logo_of_YouTube_%282015-2017%29.svg", width=200)
    st.sidebar.title("YouTube Channel Analysis")
    
    # Input options
    option = st.sidebar.radio("Choose Input Method", 
                             ["Enter Channel Details", "Use Sample Data"])
    
    channel_stats = None
    video_data = None
    
    if option == "Enter Channel Details":
        st.sidebar.markdown("### Channel Information")
        api_key = st.sidebar.text_input("Enter YouTube API Key", type="password")
        channel_id = st.sidebar.text_input("Enter Channel ID", value="UC_x5XG1OV2P6uZZ5FSM9Ttw")
        
        if st.sidebar.button("Fetch Data", key="fetch"):
            if api_key and channel_id:
                with st.spinner("Fetching data from YouTube API..."):
                    channel_stats, video_data = fetch_youtube_data(api_key, channel_id)
                    if channel_stats and video_data:
                        save_data_to_files(channel_stats, video_data)
                        st.success("Data fetched successfully!")
            else:
                st.sidebar.error("Please enter both API Key and Channel ID")
    else:
        # Load sample data
        channel_stats, video_data = load_data_from_files()
        if not (channel_stats and video_data):
            st.warning("No sample data found. Please fetch data using API first.")
    
    # Process data if available
    if channel_stats and video_data:
        enhanced_df = enhance_video_data(video_data)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["Channel Overview", "Video Analysis", 
                                         "Publishing Strategy", "Content Analytics"])
        
        with tab1:
            st.markdown("<h2 class='sub-header'>Channel Overview</h2>", unsafe_allow_html=True)
            
            # Channel metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Channel Name", channel_stats['Channel Name'])
            with col2:
                st.metric("Subscribers", f"{int(channel_stats['Subscribers']):,}")
            with col3:
                st.metric("Total Views", f"{int(channel_stats['Views']):,}")
            with col4:
                st.metric("Total Videos", int(channel_stats['Total Videos']))
            
            # Channel performance metrics
            st.markdown("<h3 class='sub-header'>Performance Metrics</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_views = enhanced_df['Views'].mean()
                st.metric("Average Views per Video", f"{int(avg_views):,}")
            with col2:
                avg_engagement = enhanced_df['Engagement Rate'].mean()
                st.metric("Average Engagement Rate", f"{avg_engagement:.2f}%")
            with col3:
                avg_daily = enhanced_df['Daily Views'].mean()
                st.metric("Average Daily Views", f"{int(avg_daily):,}")
            
            # Views Distribution
            st.markdown("<h3 class='sub-header'>Views Distribution</h3>", unsafe_allow_html=True)
            fig = px.histogram(enhanced_df, x='Views', nbins=30, 
                              title='Distribution of Video Views',
                              color_discrete_sequence=['#FF0000'])
            fig.update_layout(xaxis_title='Views', yaxis_title='Number of Videos')
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance over time
            st.markdown("<h3 class='sub-header'>Performance Over Time</h3>", unsafe_allow_html=True)
            time_df = enhanced_df.sort_values('Published Date')
            time_df['Cumulative Views'] = time_df['Views'].cumsum()
            
            fig = px.line(time_df, x='Published Date', y='Cumulative Views', 
                         title='Cumulative Views Growth Over Time',
                         color_discrete_sequence=['#FF0000'])
            fig.update_layout(xaxis_title='Date', yaxis_title='Cumulative Views')
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.markdown("<h2 class='sub-header'>Video Analysis</h2>", unsafe_allow_html=True)
            
            # Top 10 videos
            st.markdown("<h3 class='sub-header'>Top 10 Videos by Views</h3>", unsafe_allow_html=True)
            top_videos = enhanced_df.sort_values('Views', ascending=False).head(10)
            fig = px.bar(top_videos, y='Title', x='Views', orientation='h',
                        title='Top 10 Videos by Views',
                        color='Views',
                        color_continuous_scale='Reds')
            fig.update_layout(yaxis_title='', xaxis_title='Views',
                             height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Views vs Engagement
            st.markdown("<h3 class='sub-header'>Views vs. Engagement Rate</h3>", unsafe_allow_html=True)
            fig = px.scatter(enhanced_df, x='Views', y='Engagement Rate',
                           title='Views vs. Engagement Rate',
                           color='Duration Category',
                           size='Comments',
                           hover_data=['Title'])
            fig.update_layout(xaxis_title='Views', yaxis_title='Engagement Rate (%)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Duration analysis
            st.markdown("<h3 class='sub-header'>Video Duration Analysis</h3>", unsafe_allow_html=True)
            duration_df = enhanced_df.groupby('Duration Category').agg(
                Count=('Video ID', 'count'),
                Avg_Views=('Views', 'mean'),
                Avg_Engagement=('Engagement Rate', 'mean')
            ).reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(duration_df, x='Duration Category', y='Count',
                           title='Number of Videos by Duration',
                           color_discrete_sequence=['#FF0000'])
                fig.update_layout(xaxis_title='Duration Category', yaxis_title='Number of Videos')
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = px.bar(duration_df, x='Duration Category', y='Avg_Views',
                           title='Average Views by Duration',
                           color_discrete_sequence=['#FF8000'])
                fig.update_layout(xaxis_title='Duration Category', yaxis_title='Average Views')
                st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            st.markdown("<h2 class='sub-header'>Publishing Strategy</h2>", unsafe_allow_html=True)
            
            # Day of week analysis
            st.markdown("<h3 class='sub-header'>Performance by Day of Week</h3>", unsafe_allow_html=True)
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            day_df = enhanced_df.groupby('Publish Day of Week').agg(
                Avg_Views=('Views', 'mean'),
                Avg_Engagement=('Engagement Rate', 'mean'),
                Video_Count=('Video ID', 'count')
            ).reindex(day_order).reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(day_df, x='Publish Day of Week', y='Avg_Views',
                           title='Average Views by Day of Week',
                           color_discrete_sequence=['#0080FF'])
                fig.update_layout(xaxis_title='Day of Week', yaxis_title='Average Views')
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = px.bar(day_df, x='Publish Day of Week', y='Avg_Engagement',
                           title='Average Engagement by Day of Week',
                           color_discrete_sequence=['#00FFFF'])
                fig.update_layout(xaxis_title='Day of Week', yaxis_title='Average Engagement Rate (%)')
                st.plotly_chart(fig, use_container_width=True)
            
            # Publishing hour analysis
            st.markdown("<h3 class='sub-header'>Performance by Hour of Day</h3>", unsafe_allow_html=True)
            hour_df = enhanced_df.groupby('Hour Published').agg(
                Avg_Views=('Views', 'mean'),
                Video_Count=('Video ID', 'count')
            ).reset_index()
            
            fig = px.line(hour_df, x='Hour Published', y='Avg_Views',
                         title='Average Views by Hour of Publishing',
                         markers=True,
                         color_discrete_sequence=['#0080FF'])
            fig.update_layout(xaxis_title='Hour (24-hour format)', yaxis_title='Average Views')
            fig.update_xaxes(tickvals=list(range(0, 24, 2)))
            st.plotly_chart(fig, use_container_width=True)
            
            # Upload consistency
            st.markdown("<h3 class='sub-header'>Upload Consistency</h3>", unsafe_allow_html=True)
            enhanced_df['Month_Year'] = pd.to_datetime(enhanced_df['Year-Month'] + '-01')
            monthly_counts = enhanced_df.groupby('Month_Year').size().reset_index(name='Video Count')
            
            fig = px.bar(monthly_counts, x='Month_Year', y='Video Count',
                       title='Videos Published per Month',
                       color_discrete_sequence=['#0080FF'])
            fig.update_layout(xaxis_title='Month', yaxis_title='Number of Videos')
            st.plotly_chart(fig, use_container_width=True)
            
        with tab4:
            st.markdown("<h2 class='sub-header'>Content Analytics</h2>", unsafe_allow_html=True)
            
            # Title word cloud
            st.markdown("<h3 class='sub-header'>Title Word Cloud</h3>", unsafe_allow_html=True)
            wordcloud = create_wordcloud(enhanced_df)
            if wordcloud:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("Not enough data to generate word cloud")
            
            # Title analysis
            st.markdown("<h3 class='sub-header'>Title Analysis</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(enhanced_df, x='Title Length', y='Views',
                               title='Title Length vs. Views',
                               trendline='ols',
                               color_discrete_sequence=['#FF0000'])
                fig.update_layout(xaxis_title='Title Length (characters)', yaxis_title='Views')
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                question_df = enhanced_df.groupby('Has Question').agg(
                    Avg_Views=('Views', 'mean'),
                    Avg_Engagement=('Engagement Rate', 'mean'),
                    Count=('Video ID', 'count')
                ).reset_index()
                
                fig = px.bar(question_df, x='Has Question', y='Avg_Views',
                           title='Average Views: Questions vs. Statements',
                           color_discrete_sequence=['#FF8000'])
                fig.update_layout(xaxis_title='Contains Question Mark', yaxis_title='Average Views')
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance factors
            st.markdown("<h3 class='sub-header'>Performance Metrics Comparison</h3>", unsafe_allow_html=True)
            
            metric_cols = ['Views', 'Likes', 'Comments', 'Engagement Rate', 
                          'Duration (sec)', 'Title Length', 'Title Word Count']
            selected_metrics = st.multiselect("Select metrics to compare", metric_cols, 
                                             default=['Views', 'Engagement Rate', 'Duration (sec)'])
            
            if selected_metrics:
                if len(selected_metrics) >= 2:
                    fig = px.scatter_matrix(enhanced_df, dimensions=selected_metrics,
                                          color='Performance',
                                          title='Correlation Between Metrics')
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least 2 metrics to compare")
    
    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit, Plotly, and the YouTube Data API")
    
if __name__ == "__main__":
    main()