import pandas as pd
import numpy as np
import re
import difflib
from fastapi import FastAPI
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
# --- AI Imports ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re

app = FastAPI(title="Livability Dashboard API V7 (Final)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Load Data ---
try:
    df = pd.read_csv("livability_final.csv")
    
    # CRITICAL FIX: Multiply Price by 1000 (Convert 6.4 -> 6400)
    if 'sqft_price' in df.columns:
        df['sqft_price'] = df['sqft_price'] * 1000

    df = df.replace({np.nan: None})
    
    # Calculate max values dynamically for normalization
    # Fallback to 20,000 if data is missing (reasonable max for Indian metros)
    MAX_PRICE = df['sqft_price'].max() if not df.empty else 20000
    MAX_AQI = df['aqi'].max() if not df.empty else 500
    MAX_WQI = df['wqi'].max() if 'wqi' in df.columns else 200
    
    print(f"✅ Data loaded! Price Range: {df['sqft_price'].min()} - {MAX_PRICE}")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    df = pd.DataFrame()

# --- 2. AI Model Initialization ---
model = None
embeddings = None

try:
    print("🤖 Loading AI Model (this may take a moment)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create text representation for every row for semantic search
    def create_text_desc(row):
        congestion = row.get('local_congestion', 1)
        vibe = 'Quiet and peaceful' if congestion < 1.15 else 'Busy and lively'
        return f"Area: {row.get('area', '')}. City: {row.get('city', '')}. Price: {row.get('sqft_price', 0)}. " \
               f"Features: AQI {row.get('aqi', 0)}, {row.get('park', 0)} parks, {row.get('school', 0)} schools, " \
               f"{row.get('hospital', 0)} hospitals, {row.get('library', 0)} libraries. " \
               f"Vibe: {vibe}."

    if not df.empty:
        # 1. Generate Embeddings (NLP)
        df['text_desc'] = df.apply(create_text_desc, axis=1)
        print("🧠 Generating embeddings...")
        embeddings = model.encode(df['text_desc'].tolist())
        print("✅ Embeddings ready!")
        
        # 2. Generate Clusters (K-Means)
        print("🏙️  Clustering neighborhoods...")
        # Select features for clustering
        features = ['sqft_price', 'aqi', 'livability_score', 'local_congestion']
        # Fill NA just in case
        X = df[features].fillna(0)
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train KMeans (k=6 distinct vibes)
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        df['cluster_id'] = kmeans.fit_predict(X_scaled)
        
        # Assign Descriptive Names to Clusters based on their centers
        # We analyze the mean of each cluster to name it
        cluster_labels = {}
        for i in range(6):
            cluster_mean = df[df['cluster_id'] == i][features].mean()
            price_score = cluster_mean['sqft_price']
            aqi_score = cluster_mean['aqi']
            
            label = "Standard Area"
            if price_score > df['sqft_price'].mean() * 1.5:
                label = "💎 Premium / Luxury"
            elif price_score < df['sqft_price'].mean() * 0.7:
                label = "💰 Budget Friendly"
                
            if aqi_score < 70:
                label += " & 🌿 Pure Air"
            elif aqi_score > 120:
                label += " & 🏭 Urban/Industrial"
            
            cluster_labels[i] = label
            
        df['cluster_label'] = df['cluster_id'].map(cluster_labels)
        print("✅ Clustering complete!")

    else:
        print("⚠️ DataFrame is empty, skipping embeddings.")

except Exception as e:
    print(f"❌ Error loading AI model: {e}")
    print("⚠️ Falling back to keyword search.")


# --- 2. Concept Mapping ---
CONCEPT_MAP = {
    'school': ['school', 'education', 'study', 'kid', 'child', 'student'],
    'hospital': ['hospital', 'clinic', 'doctor', 'medical', 'health', 'care'],
    'park': ['park', 'garden', 'green', 'nature', 'tree', 'walk'],
    'traffic': ['traffic', 'jam', 'congestion', 'signal', 'commute', 'drive'],
    'shopping': ['supermarket', 'store', 'mall', 'shop', 'market', 'groceries'],
    'quiet': ['quiet', 'calm', 'peaceful', 'silent'],
    'lively': ['lively', 'active', 'busy', 'fun', 'happening', 'social'],
    'cheap': ['cheap', 'budget', 'affordable', 'low', 'economical', 'rent'],
    'expensive': ['expensive', 'luxury', 'premium', 'rich', 'posh'],
    'clean': ['clean', 'fresh', 'breathe', 'pure', 'air'],
    'library': ['library', 'book', 'reading', 'literacy'],
    'water': ['water', 'drink', 'tap', 'supply'],
    'playground': ['playground', 'play', 'sport', 'game'],
    'community': ['community', 'civic', 'vote', 'safe', 'people']
}

# Definitions for Column Relevance (AI Filter Selection)
COLUMN_DEFINITIONS = {
    'school': 'schools education study students university college academic',
    'hospital': 'hospitals clinics medical health doctors emergency',
    'park': 'parks gardens nature green walking trees outdoors',
    'library': 'libraries books reading quiet study research',
    'playground': 'playgrounds sports kids games recreation',
    'supermarket': 'supermarkets groceries shopping food daily needs',
    'store': 'stores shops malls retail shopping',
    'wqi': 'water quality drinking purity clean water pollution',
    'aqi': 'air quality pollution smog breathing clean air',
    'local_congestion': 'traffic congestion commute driving roads transport',
    'sqft_price': 'price cost budget expensive cheap affordable property value',
    'voter_turnout': 'community civic engagement voting safety people',
    'hdi_rank': 'development index standard of living'
}

def get_fuzzy_intent(token):
    """Finds the best matching concept for a user's word."""
    all_keywords = [word for words in CONCEPT_MAP.values() for word in words]
    matches = difflib.get_close_matches(token, all_keywords, n=1, cutoff=0.8)
    if matches:
        match = matches[0]
        for concept, words in CONCEPT_MAP.items():
            if match in words:
                return concept
    return None

def calculate_match_score(row, query_str):
    score = row.get('livability_score', 0) or 0
    query_lower = query_str.lower()
    tokens = query_lower.split()

    # --- A. Explicit Number Constraints ---
    # Example: "under 5000", "below 10000"
    price_cap_match = re.search(r'(?:under|below|max|less than)\s*₹?(\d+)', query_lower)
    if price_cap_match:
        cap = float(price_cap_match.group(1))
        # If the area price is higher than user budget, penalize heavily
        if row['sqft_price'] > cap:
            score -= 100 

    # --- B. Concept Scoring ---
    
    # Global modifiers
    if 'cheap' in query_lower or get_fuzzy_intent('cheap'):
        # Lower price = Higher score
        score += (1 - (row['sqft_price'] / MAX_PRICE)) * 8
    elif 'expensive' in query_lower or get_fuzzy_intent('expensive'):
        # Higher price = Higher score
        score += (row['sqft_price'] / MAX_PRICE) * 5

    if 'clean' in query_lower or get_fuzzy_intent('clean'):
        score += (1 - (row['aqi'] / MAX_AQI)) * 6

    # Specific Amenities
    for token in tokens:
        intent = get_fuzzy_intent(token)
        
        if intent == 'school': score += row.get('school', 0) * 3
        elif intent == 'hospital': score += row.get('hospital', 0) * 3
        elif intent == 'park': score += row.get('park', 0) * 3
        elif intent == 'library': score += row.get('library', 0) * 4
        elif intent == 'playground': score += row.get('playground', 0) * 3
        elif intent == 'shopping': score += (row.get('supermarket', 0) + row.get('store', 0)) * 2

        # Complex Metrics
        elif intent == 'water':
            current_wqi = row.get('wqi', 100) or 100
            score += (1 - (current_wqi / MAX_WQI)) * 4
            
        elif intent == 'traffic':
            score += (1 - row.get('local_congestion', 0.5)) * 5

        elif intent == 'community':
            score += (row.get('voter_turnout', 50) / 100) * 3

        elif intent == 'quiet':
            score += (1 - row.get('local_congestion', 0.5)) * 4
            score += row.get('park', 0)

    return score

# --- Endpoints ---

@app.get("/cities")
def get_cities():
    return df['city'].unique().tolist() if not df.empty else []

@app.get("/data")
def get_all_data(city: Optional[str] = None, sort_by: str = "livability_score", ascending: bool = False, limit: int = 50):
    if df.empty: return []
    filtered = df[df['city'].str.lower() == city.lower()] if city else df
    
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(by=sort_by, ascending=ascending)
        
    return filtered.head(limit).to_dict(orient="records")

@app.get("/stats")
def get_city_stats():
    if df.empty: return []
    stats = []
    for city_name, group in df.groupby('city'):
        stats.append({
            "city": city_name,
            "average_price": round(group['sqft_price'].mean(), 2),
            "average_aqi": round(group['aqi'].mean(), 2),
            "avg_livability": round(group['livability_score'].mean(), 2) if 'livability_score' in group else 0,
            "total_areas": int(len(group))
        })
    return stats

@app.get("/recommend")
def get_recommendation(query: str, city: Optional[str] = None):
    if df.empty: return []

    # Filter target indices
    target_idx = df.index
    if city:
        target_idx = df[df['city'] == city].index
    
    if len(target_idx) == 0: return []
    
    target_df = df.loc[target_idx].copy()
    
    # --- Option A: Semantic Search (AI) ---
    if model is not None and embeddings is not None:
        try:
            # 1. Embed query
            query_embedding = model.encode([query])
            
            # 2. Get target embeddings
            target_embeddings_matrix = embeddings[target_idx]
            
            # 3. Calculate Similarity
            sim_scores = cosine_similarity(query_embedding, target_embeddings_matrix)[0]
            
            # 4. Scale to 0-100 match score
            target_df['match_score'] = sim_scores * 100
            
            # 5. Apply Hard Constraints
            price_cap_match = re.search(r'(?:under|below|max|less than)\s*₹?(\d+)', query.lower())
            if price_cap_match:
                cap = float(price_cap_match.group(1))
                mask_over_budget = target_df['sqft_price'] > cap
                target_df.loc[mask_over_budget, 'match_score'] -= 50
            
            # 6. Identify Relevant Columns (Soft Filtering)
            suggested_fields = []
            try:
                col_names = list(COLUMN_DEFINITIONS.keys())
                col_texts = list(COLUMN_DEFINITIONS.values())
                col_embeddings = model.encode(col_texts)
                col_sim = cosine_similarity(query_embedding, col_embeddings)[0]
                
                for i, score in enumerate(col_sim):
                    if score > 0.25:
                        suggested_fields.append(col_names[i])
            except:
                pass

            top_results = target_df.sort_values(by='match_score', ascending=False).head(5).to_dict(orient="records")
            return {"results": top_results, "suggested_fields": suggested_fields}
            
        except Exception as e:
            print(f"AI Search Error: {e}. Falling back to rule-based.")

    # --- Option B: Fallback Rule-Based Search ---
    target_df['match_score'] = target_df.apply(lambda row: calculate_match_score(row, query), axis=1)
    results = target_df.sort_values(by='match_score', ascending=False).head(5).to_dict(orient="records")
    return {"results": results, "suggested_fields": []}