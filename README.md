# LivabilityAI Pro

A comprehensive livability analysis dashboard that helps users find the perfect neighborhood based on air quality, amenities, pricing, and lifestyle preferences using advanced AI-powered semantic search and clustering.

## Features

- **AI-Powered Search**: Natural language queries to find neighborhoods (e.g., "quiet family-friendly area with good schools")
- **Smart Clustering**: Automatically groups similar neighborhoods into 6 distinct categories
- **Comprehensive Metrics**: Analyzes air quality (AQI), pricing, congestion, and amenities
- **Interactive Dashboard**: Modern React-based web interface
- **Real-time API**: FastAPI backend for instant results

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: React.js with Tailwind CSS
- **AI/ML**: Sentence Transformers, Scikit-learn (K-Means clustering)
- **Data Processing**: Pandas, NumPy
- **Deployment**: Uvicorn ASGI server

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LivabilityIndex
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download AI model** (happens automatically on first run)
   The SentenceTransformer model will be downloaded when you start the application.

## Usage

1. **Start the backend server**
   ```bash
   uvicorn main:app --reload
   ```

2. **Open the dashboard**
   Open `index.html` in your web browser.

3. **Explore neighborhoods**
   - Use natural language search (e.g., "affordable area with clean air")
   - Filter by price range, AQI levels, and amenities
   - View clustered neighborhoods by lifestyle type

## Data

The application uses livability data from `livability_final.csv` containing:
- Neighborhood information
- Air Quality Index (AQI)
- Price per square foot
- Amenity counts (schools, hospitals, parks, libraries)
- Congestion levels
- Livability scores

## API Endpoints

- `GET /` - Health check
- `GET /data` - Get all neighborhood data
- `GET /search?query={text}` - Semantic search for neighborhoods
- `GET /clusters` - Get clustered neighborhood data
- `GET /stats` - Get data statistics and ranges

## Project Structure

```
LivabilityIndex/
├── main.py                 # FastAPI backend
├── index.html             # React frontend
├── requirements.txt       # Python dependencies
├── livability_final.csv   # Main dataset
├── livability_data.csv    # Raw data
├── README.md             # This file
└── __pycache__/          # Python cache
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with FastAPI and React
- AI models from Sentence Transformers
- Data visualization with Tailwind CSS

