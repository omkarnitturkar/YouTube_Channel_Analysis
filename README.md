# YouTube Analytics Pro

A powerful tool to analyze YouTube channels and their content, built with Streamlit and the YouTube Data API.

## Features

- Channel overview and statistics
- Video performance analysis
- Publishing strategy insights
- Content analytics and recommendations
- Interactive visualizations

## Setup Instructions

### Prerequisites

- Python 3.7+
- A YouTube Data API key (instructions below)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/youtube-analytics-pro.git
cd youtube-analytics-pro
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your YouTube API key:

#### For Local Development:

Create a `.streamlit/secrets.toml` file with:
```toml
youtube_api_key = "YOUR_ACTUAL_API_KEY"
```

OR

Create a `.env` file with:
```
YOUTUBE_API_KEY=YOUR_ACTUAL_API_KEY
```

#### For Streamlit Cloud Deployment:

1. Go to your Streamlit Cloud dashboard
2. Navigate to your app's "Settings" → "Secrets"
3. Add a secret with the key `youtube_api_key` and your API key as the value

### Getting a YouTube API Key

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to "APIs & Services" → "Library"
4. Search for "YouTube Data API v3" and enable it
5. Go to "APIs & Services" → "Credentials"
6. Click "Create Credentials" → "API Key"
7. Copy your new API key

### Running the App

```bash
streamlit run UI2.py
```

## Security Note

- **IMPORTANT**: Never commit your actual API key to GitHub
- The app is designed to use Streamlit's secrets management for secure API key handling
- The API key values in the code are placeholders only

## Deployment

This app is ready for deployment on Streamlit Cloud. Make sure to:

1. Add your YouTube API key to the Streamlit Cloud secrets as described above
2. Make sure `.streamlit/secrets.toml` is in your `.gitignore` file
3. Connect your GitHub repository to Streamlit Cloud

## Usage

1. Launch the app
2. Enter a YouTube channel name or ID
3. Click "Analyze Channel" to fetch and analyze data
4. Explore the different tabs for various insights

## License

MIT
