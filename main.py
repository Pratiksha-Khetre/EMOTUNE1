# main.py
import os
import base64
import requests
import cv2
import numpy as np
from dotenv import load_dotenv
from urllib.parse import urlencode

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import uvicorn

load_dotenv()

# ========== APP SETUP ==========
app = FastAPI(title="EmoTune API", version="1.0.0")

# ========== CONFIGURATION (from env) ==========
# Make sure these env vars are set in Render:
# SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv(
    "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/spotify/callback"
)

# main.py - CORS Configuration Section
# Replace your existing CORS setup with this:

import os
from fastapi.middleware.cors import CORSMiddleware

# Get allowed origins from environment or use defaults
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://emotune-nine.vercel.app,http://localhost:5173,http://127.0.0.1:5173"
).split(",")

# Add wildcard for Vercel preview deployments (optional but recommended)
ALLOWED_ORIGINS.extend([
    "https://emotune-*.vercel.app",  # Preview deployments
])

print(f"🌐 CORS Allowed Origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicit methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all response headers
    max_age=3600,  # Cache preflight requests for 1 hour
)

# CRITICAL: Add OPTIONS handler for preflight requests
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle CORS preflight OPTIONS requests"""
    return {"message": "OK"}

# You can add more allowed origins if needed
ALLOWED_ORIGINS = [
    "https://emotune-nine.vercel.app",
    "https://song-recommendation-system-f5zo.onrender.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=ALLOWED_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ========== MODEL CONFIG ==========
MODEL_PATH = os.getenv("MODEL_PATH", "./models/raf_db_model.h5")
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
IMG_HEIGHT = 75
IMG_WIDTH = 75

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

spotify_tokens = {}

# ========== LOAD MODEL ==========
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✓ Deep Learning Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading model from {MODEL_PATH}: {e}")
    print(f"   Current working directory: {os.getcwd()}")
    try:
        print(f"   Files in directory: {os.listdir('.')}")
    except Exception:
        pass
    model = None

# ========== EMOTION & LANGUAGE KEYWORDS ==========
EMOTION_KEYWORDS = {
    "Happy": ["happy", "cheerful", "uplifting", "feel good", "positive"],
    "Sad": ["sad", "melancholic", "emotional", "soulful", "deep"],
    "Angry": ["angry", "aggressive", "intense", "powerful", "rock"],
    "Fear": ["scary", "horror", "dark", "ominous", "creepy"],
    "Disgust": ["edgy", "dark", "alternative", "rebellious", "punk"],
    "Surprise": ["energetic", "exciting", "upbeat", "dance", "fun"],
    "Neutral": ["relaxing", "calm", "ambient", "peaceful", "chill"],
}

LANGUAGE_KEYWORDS = {
    "Hindi": ["hindi", "bollywood", "indian", "desi", "hindi songs"],
    "English": ["english", "pop", "rock", "rap", "american", "british"],
    "Marathi": ["marathi", "maharashtra", "marathi songs", "marathi music"],
    "Telugu": ["telugu", "telangana", "telugu songs", "telugu music"],
    "Tamil": ["tamil", "tamilnadu", "tamil songs", "tamil music"],
    "Gujarati": ["gujarati", "gujarati songs", "gujarati music", "gujarati folk"],
    "Urdu": ["urdu", "ghazal", "urdu poetry", "sufi", "qawwali"],
    "Kannada": ["kannada", "karnataka", "kannada songs", "kannada music"],
    "Bengali": ["bengali", "bengal", "bengali songs", "bengali folk"],
    "Malayalam": ["malayalam", "kerala", "malayalam songs", "malayalam music"],
}

# ========== SPOTIFY FUNCTIONS ==========
@app.get("/spotify/login")
async def spotify_login():
    """
    Returns the Spotify OAuth authorization URL for user login.
    Frontend should open the returned auth_url in a new window.
    """
    if not SPOTIFY_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Spotify client not configured on server.")
    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": "streaming user-read-private user-read-email",
    }
    auth_url = f"https://accounts.spotify.com/authorize?{urlencode(params)}"
    return {"auth_url": auth_url}


@app.get("/spotify/callback")
async def spotify_callback(code: str = Query(None), error: str = Query(None)):
    """
    Spotify will redirect here with code (if user accepts).
    This endpoint exchanges code for tokens and stores them in memory (demo only).
    """
    if error:
        return HTMLResponse(
            "<html><body style='background: #0f0f1c; color: #f0f0f0; font-family: Arial; padding: 50px; text-align: center;'><h2>Error connecting to Spotify</h2><script>window.close();</script></body></html>"
        )

    if not code:
        return HTMLResponse(
            "<html><body style='background: #0f0f1c; color: #f0f0f0; font-family: Arial; padding: 50px; text-align: center;'><h2>No code provided by Spotify.</h2></body></html>"
        )

    try:
        if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
            return HTMLResponse(
                "<html><body style='background: #0f0f1c; color: #f0f0f0; font-family: Arial; padding: 50px; text-align: center;'><h2>Spotify credentials not configured on server.</h2></body></html>"
            )

        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": SPOTIFY_REDIRECT_URI,
            "client_id": SPOTIFY_CLIENT_ID,
            "client_secret": SPOTIFY_CLIENT_SECRET,
        }

        response = requests.post("https://accounts.spotify.com/api/token", data=token_data)
        tokens = response.json()
        if response.status_code != 200:
            print("Spotify token exchange failed:", tokens)
            return HTMLResponse(
                f"<html><body style='background: #0f0f1c; color: #f0f0f0; font-family: Arial; padding: 50px; text-align: center;'><h2>Spotify token exchange failed.</h2><pre>{tokens}</pre></body></html>"
            )

        # Store tokens in memory for demo purposes (replace with persistent store for production)
        spotify_tokens["current_user"] = tokens

        return HTMLResponse(
            "<html><body style='background: #0f0f1c; color: #f0f0f0; font-family: Arial; padding: 50px; text-align: center;'><h2>Connected to Spotify!</h2><p>You can close this window.</p><script>setTimeout(() => window.close(), 1500);</script></body></html>"
        )
    except Exception as e:
        print("Error in spotify_callback:", e)
        return HTMLResponse(f"<html><body><h2>Error: {str(e)}</h2></body></html>")


def get_spotify_token():
    """
    Get a valid Spotify API token (client_credentials if user token missing).
    Returns access token string or None.
    """
    try:
        # If a user authorized recently, use that token if present
        if "current_user" in spotify_tokens:
            token = spotify_tokens["current_user"].get("access_token")
            return token

        # Otherwise use Client Credentials flow
        if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
            print("Spotify client id/secret missing in env")
            return None

        auth_str = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
        auth_bytes = base64.b64encode(auth_str.encode()).decode()
        headers = {
            "Authorization": f"Basic {auth_bytes}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"grant_type": "client_credentials"}
        response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
        if response.status_code == 200:
            token = response.json().get("access_token")
            return token
        else:
            print("Spotify client_credentials failed:", response.status_code, response.text)
            return None
    except Exception as e:
        print(f"Error getting Spotify token: {e}")
        return None


def search_spotify_tracks_by_emotion_and_language(emotion, language):
    """
    Search Spotify for tracks based on emotion + language keywords.
    Returns a list of track dicts.
    """
    try:
        token = get_spotify_token()
        if not token:
            print("Could not get Spotify token")
            return []

        if emotion not in EMOTION_KEYWORDS:
            emotion = "Neutral"
        if language not in LANGUAGE_KEYWORDS:
            language = "English"

        emotion_keywords = EMOTION_KEYWORDS[emotion]
        language_keywords = LANGUAGE_KEYWORDS[language]
        all_tracks = []

        print(f"\n🎵 Searching Spotify for {emotion} + {language} songs")
        print(f"   Emotion keywords: {emotion_keywords}")
        print(f"   Language keywords: {language_keywords}")

        for emotion_kw in emotion_keywords[:3]:
            for lang_kw in language_keywords[:2]:
                combined_query = f"{emotion_kw} {lang_kw}"
                try:
                    search_response = requests.get(
                        "https://api.spotify.com/v1/search",
                        headers={"Authorization": f"Bearer {token}"},
                        params={"q": combined_query, "type": "track", "limit": 15, "market": "US"},
                        timeout=10,
                    )

                    if search_response.status_code == 200:
                        results = search_response.json()
                        tracks = results.get("tracks", {}).get("items", [])
                        for track in tracks:
                            if not any(t["id"] == track["id"] for t in all_tracks):
                                album_image = ""
                                if track.get("album", {}).get("images"):
                                    album_image = track["album"]["images"][0]["url"]
                                embed_url = f"https://open.spotify.com/embed/track/{track['id']}"
                                track_data = {
                                    "id": track["id"],
                                    "title": track["name"],
                                    "artist": ", ".join([a["name"] for a in track["artists"]]),
                                    "image_url": album_image,
                                    "external_url": track.get("external_urls", {}).get("spotify", ""),
                                    "embed_url": embed_url,
                                }
                                all_tracks.append(track_data)
                    else:
                        print(f"Spotify search returned {search_response.status_code} for query '{combined_query}'")
                except Exception as e:
                    print(f"  ⚠ Error searching '{combined_query}': {e}")
                    continue

        # If not enough, fall back to emotion-only search
        if len(all_tracks) < 10:
            print(f"  ℹ Only {len(all_tracks)} tracks found with combined search, adding more from emotion search...")
            for emotion_kw in emotion_keywords:
                try:
                    search_response = requests.get(
                        "https://api.spotify.com/v1/search",
                        headers={"Authorization": f"Bearer {token}"},
                        params={"q": emotion_kw, "type": "track", "limit": 20, "market": "US"},
                        timeout=10,
                    )

                    if search_response.status_code == 200:
                        results = search_response.json()
                        tracks = results.get("tracks", {}).get("items", [])
                        for track in tracks:
                            if not any(t["id"] == track["id"] for t in all_tracks):
                                album_image = ""
                                if track.get("album", {}).get("images"):
                                    album_image = track["album"]["images"][0]["url"]
                                embed_url = f"https://open.spotify.com/embed/track/{track['id']}"
                                track_data = {
                                    "id": track["id"],
                                    "title": track["name"],
                                    "artist": ", ".join([a["name"] for a in track["artists"]]),
                                    "image_url": album_image,
                                    "external_url": track.get("external_urls", {}).get("spotify", ""),
                                    "embed_url": embed_url,
                                }
                                all_tracks.append(track_data)
                            if len(all_tracks) >= 20:
                                break
                    else:
                        print(f"Spotify search returned {search_response.status_code} for emotion '{emotion_kw}'")
                    if len(all_tracks) >= 20:
                        break
                except Exception as e:
                    print(f"  ⚠ Error searching '{emotion_kw}': {e}")
                    continue

        print(f"  📊 Total unique tracks found: {len(all_tracks)}")
        return all_tracks[:20]
    except Exception as e:
        print(f"Error searching Spotify: {e}")
        return []


# ========== EMOTION DETECTION ==========

def process_and_predict(image_file_bytes):
    """
    Face detection, preprocessing, prediction and image annotation.
    Returns dict with base64-encoded annotated image and prediction info.
    """
    if model is None:
        raise Exception("Model not loaded on server.")

    # Decode image bytes
    nparr = np.frombuffer(image_file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise Exception("Invalid or unsupported image format.")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(
        img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        raise Exception("No face detected in the image frame.")

    # Use the largest face
    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])

    face_roi = img_gray[y : y + h, x : x + w]
    face_resized = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT))

    processed_input = face_resized.astype("float32") / 255.0
    processed_input = np.expand_dims(processed_input, axis=-1)
    processed_input = np.expand_dims(processed_input, axis=0)

    predictions = model.predict(processed_input, verbose=0)
    probabilities = predictions[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_emotion = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    # Annotate image with bounding box and text
    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = f"{predicted_emotion} ({confidence * 100:.1f}%)"
    cv2.putText(
        img_bgr,
        text,
        (x, max(y - 10, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    # Encode to base64 JPEG
    success, buffer = cv2.imencode(".jpeg", img_bgr)
    if not success:
        raise Exception("Failed to encode processed image.")
    processed_image_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("utf-8")

    return {
        "processed_image_b64": processed_image_b64,
        "predicted_emotion": predicted_emotion,
        "confidence": confidence,
        "all_confidences": {CLASS_NAMES[i]: round(float(probabilities[i]), 4) for i in range(len(CLASS_NAMES))},
    }


@app.post("/analyze_emotion/")
async def analyze_emotion(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = process_and_predict(image_bytes)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        # Provide clear error message for frontend debugging
        print("Error in /analyze_emotion/:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ========== RECOMMENDATIONS ==========

@app.get("/get_recommendations/")
async def get_recommendations(emotion: str = Query("Neutral"), languages: str = Query("English"), offset: int = Query(0)):
    # Parse languages (comma-separated)
    language_list = [lang.strip() for lang in languages.split(",") if lang.strip()]
    if not language_list:
        language_list = ["English"]

    print(f"\n📍 RECOMMENDATION REQUEST: Emotion={emotion}, Languages={language_list}, Offset={offset}")

    all_recommendations = []
    for language in language_list:
        print(f"   🔍 Searching {language}...")
        language_results = search_spotify_tracks_by_emotion_and_language(emotion, language)
        for track in language_results:
            track["language"] = language
            if not any(t["id"] == track["id"] for t in all_recommendations):
                all_recommendations.append(track)

    total_available = len(all_recommendations)
    paginated_recommendations = all_recommendations[offset : offset + 20]

    print(f"   📊 Total tracks found: {total_available}; Returning {len(paginated_recommendations)} from index {offset}")

    return {
        "recommendations": paginated_recommendations,
        "emotion": emotion,
        "languages": language_list,
        "total_available": total_available,
        "offset": offset,
        "returned_count": len(paginated_recommendations),
    }


# ========== HEALTH CHECK & ROOT ==========
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "spotify_configured": bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET),
    }


@app.get("/")
async def root():
    return {"message": "EmoTune API is running"}


# ========== RUN (for local dev) ==========
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
