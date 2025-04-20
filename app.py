import os
import pandas as pd
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class YouTubeAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        
    def get_channel_stats(self, channel_id):
        """Get basic statistics for a YouTube channel"""
        request = self.youtube.channels().list(
            part="snippet,contentDetails,statistics",
            id=channel_id
        )
        response = request.execute()
        
        if not response['items']:
            return None
        
        channel_data = response['items'][0]
        
        # Extract relevant data
        stats = {
            'Channel Name': channel_data['snippet']['title'],
            'Channel ID': channel_id,
            'Subscribers': int(channel_data['statistics'].get('subscriberCount', 0)),
            'Views': int(channel_data['statistics'].get('viewCount', 0)),
            'Total Videos': int(channel_data['statistics'].get('videoCount', 0)),
            'Channel Created': channel_data['snippet']['publishedAt'],
            'Playlist ID': channel_data['contentDetails']['relatedPlaylists']['uploads']
        }
        
        return stats
    
    def get_video_ids(self, playlist_id, max_results=50):
        """Get video IDs from a playlist"""
        video_ids = []
        next_page_token = None
        
        while True:
            request = self.youtube.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=max_results,
                pageToken=next_page_token
            )
            response = request.execute()
            
            # Get video IDs
            for item in response['items']:
                video_ids.append(item['contentDetails']['videoId'])
                
            # Check if there are more pages
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
        return video_ids
    
    def get_video_details(self, video_ids):
        """Get details of multiple videos"""
        all_video_data = []
        
        # Process in batches of 50 (API limit)
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            
            request = self.youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=','.join(batch_ids)
            )
            response = request.execute()
            
            for video in response['items']:
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
                
        return all_video_data
    
    def process_video_data(self, video_data):
        """Process and clean video data"""
        df = pd.DataFrame(video_data)
        
        # Convert date strings to datetime objects
        df['Published Date'] = pd.to_datetime(df['Published Date'])
        
        # Extract date components
        df['Publish Year'] = df['Published Date'].dt.year
        df['Publish Month'] = df['Published Date'].dt.month
        df['Publish Day'] = df['Published Date'].dt.day
        df['Publish Day of Week'] = df['Published Date'].dt.day_name()
        
        # Calculate engagement metrics
        df['Engagement Rate'] = ((df['Likes'] + df['Comments']) / df['Views'] * 100).round(2)
        
        # Clean and format tags
        df['Tag Count'] = df['Tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        return df

def main():
    # Replace with your API key
    API_KEY = "AIzaSyBetONqBCgl1VUTsw3BJoA1F7_JH7UZHJg"
    # Replace with the channel ID you want to analyze
    CHANNEL_ID = "UC_x5XG1OV2P6uZZ5FSM9Ttw"  # Google Developers channel as example
    
    try:
        # Create analyzer instance
        analyzer = YouTubeAnalyzer(API_KEY)
        
        # Get channel statistics
        channel_stats = analyzer.get_channel_stats(CHANNEL_ID)
        if not channel_stats:
            print(f"Channel with ID {CHANNEL_ID} not found.")
            return
        
        # Save channel stats to CSV
        channel_df = pd.DataFrame([channel_stats])
        channel_df.to_csv('channel_stats.csv', index=False)
        print(f"Channel stats saved: {channel_stats['Channel Name']}")
        
        # Get video IDs from uploads playlist
        playlist_id = channel_stats['Playlist ID']
        video_ids = analyzer.get_video_ids(playlist_id)
        print(f"Found {len(video_ids)} videos")
        
        # Get video details
        video_data = analyzer.get_video_details(video_ids)
        
        # Process video data
        video_df = analyzer.process_video_data(video_data)
        
        # Save to CSV
        video_df.to_csv('video_data.csv', index=False)
        print(f"Video data saved to 'video_data.csv'")
        
        print("Data extraction completed successfully!")
        
    except HttpError as e:
        print(f"An HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()