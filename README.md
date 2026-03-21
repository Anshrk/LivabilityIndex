# LivabilityAI Pro

A neighborhood recommendation dashboard that helps users find the perfect area based on air quality, amenities, pricing, and lifestyle preferences — powered by NLP semantic search and K-Means clustering.

## Features

- **AI-Powered Recommendations**: Natural language queries like *"cheap area with good schools and clean air"*
- **Semantic Search**: Uses sentence embeddings + cosine similarity to match queries to neighborhoods
- **Smart Clustering**: Groups neighborhoods into 6 lifestyle-based categories (e.g. "💎 Premium & 🌿 Pure Air")
- **Rule-based Fallback**: Fuzzy keyword matching if the AI model is unavailable
- **Comprehensive Metrics**: AQI, price per sqft, congestion, schools, hospitals, parks, libraries
- **REST API**: FastAPI backend with 4 endpoints

## Tech Stack

- **Backend**: FastAPI (Python), Uvicorn
- **Frontend**: HTML + JavaScript
- **AI/ML**: Sentence Transformers (`all-MiniLM-L6-v2`), Scikit-learn (K-Means, cosine similarity)
- **Data Processing**: Pandas, NumPy
- **Fuzzy Matching**: difflib

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

4. **AI model** downloads automatically on first run via `sentence-transformers`.

## Usage

1. **Start the backend**
   ```bash
   uvicorn main:app --reload
   ```

2. **Open the frontend**
   Open `index.html` in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/cities` | Returns list of all available cities |
| `GET` | `/data` | Returns neighborhood data, filterable by city, sortable by any column |
| `GET` | `/stats` | Returns per-city aggregates: avg price, AQI, livability score, area count |
| `GET` | `/recommend` | Main endpoint — semantic search returning top 5 neighborhoods for a query |

### `/recommend` query params
| Param | Type | Description |
|-------|------|-------------|
| `query` | string (required) | Natural language input e.g. `"quiet area with parks"` |
| `city` | string (optional) | Filter results to a specific city |

### `/data` query params
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `city` | string | all cities | Filter by city name |
| `sort_by` | string | `livability_score` | Column to sort by |
| `ascending` | bool | `false` | Sort order |
| `limit` | int | `50` | Max rows to return |

### Example request
```
GET /recommend?query=affordable area with clean air&city=Bangalore
```

### Example response
```json
{
  "results": [
    {
      "area": "HSR Layout",
      "city": "Bangalore",
      "sqft_price": 4200,
      "aqi": 72,
      "school": 6,
      "livability_score": 74.2,
      "match_score": 87.3,
      "cluster_label": "💰 Budget Friendly & 🌿 Pure Air"
    }
  ],
  "suggested_fields": ["sqft_price", "aqi"]
}
```

## How Recommendation Works

1. At startup, every neighborhood's features are converted into a text description and embedded using `all-MiniLM-L6-v2`
2. On a query, the user's input is embedded the same way
3. Cosine similarity is computed between the query and all neighborhood embeddings
4. Hard price constraints (e.g. "under ₹5000") are extracted via regex and applied
5. Top 5 results returned along with relevant fields to display
6. If the model is unavailable, a rule-based scorer using fuzzy keyword matching is used as fallback

## Data

`livability_final.csv` contains:
- Area and city names
- Price per square foot (`sqft_price`)
- Air Quality Index (`aqi`)
- Water Quality Index (`wqi`)
- Amenity counts: schools, hospitals, parks, libraries, playgrounds, supermarkets
- Congestion levels (`local_congestion`)
- Livability scores
- HDI rank, voter turnout

## Project Structure

```
LivabilityIndex/
├── main.py                # FastAPI backend
├── index.html             # Frontend
├── requirements.txt       # Python dependencies
├── livability_final.csv   # Main dataset
├── livability_data.csv    # Raw data
└── README.md
```

## License

MIT License