# 🎬 MovieMind — AI Movie Genre Predictor

MovieMind uses machine learning to analyse movie plot summaries and predict their genre instantly, trained on 54,000+ real movie plots across multiple genres.

## Features

- **Instant genre prediction** from any plot summary
- **Confidence score** with animated gauge display
- **Top 3 genre breakdown** with percentage bars
- **Keyword analysis** — shows which words influenced the prediction
- **Example movies** per genre
- **Session history** — tracks your recent predictions
- **Session statistics** — avg. confidence, top genre, fastest prediction
- **Export** results as JSON, CSV, or HTML report
- **Surprise Me** — pre-loads sample plots by genre to test the model

## Tech Stack

- **Backend**: Python, Flask, scikit-learn
- **NLP**: TF-IDF vectorizer + Calibrated Linear SVC
- **Frontend**: Vanilla JS, CSS animations, no frameworks

## Project Structure

```
moviemind/
├── app.py                  # Flask app + prediction logic
├── train.py                # Model training script
├── templates/
│   ├── index.html          # Landing page
│   ├── login.html          # Login page
│   ├── register.html       # Registration page
│   ├── predict.html        # Prediction dashboard
│   └── forgot.html         # Password reset page
├── requirements.txt        # Python dependencies
├── artifacts/
│   ├── model.pkl           # Trained model
│   ├── label_encoder.pkl   # Genre label encoder
│   └── evaluation_report.json
└── dataset/
    └── movies_dataset.csv  # Training data
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Retrain the model
```bash
python train.py
```

### 3. Start the Flask server
```bash
python app.py
```

### 4. Open in browser
Visit: **http://localhost:5000**

## Usage

1. Open `http://localhost:5000` to see the landing page
2. Click **Start Predicting** to go to the prediction dashboard
3. Type or paste a movie plot summary into the text box
4. Click **Predict** (or press `Ctrl+Enter`)
5. View the predicted genre, confidence score, top-3 breakdown, and example movies
6. Use **Surprise Me** to load a sample plot and test different genres
7. Export your results using the Export buttons

## API

The prediction endpoint accepts a JSON POST:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"plot": "A young wizard discovers his magical heritage..."}'
```

**Response:**
```json
{
  "genre": "Fantasy",
  "emoji": "🧙",
  "confidence": 87.3,
  "top3": [
    {"genre": "Fantasy", "score": 87.3},
    {"genre": "Adventure", "score": 8.1},
    {"genre": "Drama", "score": 4.6}
  ],
  "keywords": ["wizard", "magic", "spell"],
  "movies": ["The Lord of the Rings (2001)", "Harry Potter (2001)", "Pan's Labyrinth (2006)"],
  "prediction_time": "0.012s"
}
```

## Supported Genres

Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Fantasy, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, Western, War, History, Sport, Family
## run the website in terminal like
with command python app.py
 
