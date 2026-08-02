🔮 Spectra — AI Multi-Modal Authenticity Analyzer

Detecting reality in the age of artificial intelligence.

Spectra is a Streamlit web app that analyzes text, images, and video to estimate whether the content is human-created or AI-generated. It uses lightweight statistical and signal-processing heuristics — no external API keys or paid models required.

✨ Features
📝 Text Detection — Paste any text and get a human vs. AI verdict based on writing-style analysis.
🖼 Image Detection — Upload a photo/image and get a verdict based on pixel-level and noise analysis.
🎥 Video Detection — Upload a video and get a verdict based on motion and temporal-consistency analysis.
🎛 Confidence score (%) and a breakdown of the underlying signals for every analysis.
🌌 Sleek glassmorphism UI with a custom space-themed background.
📸 Screenshots
📝 Text Detection

Add a screenshot of the Text Detection tab here.

Show Image

🖼 Image Detection

Add a screenshot of the Image Detection tab here.

Show Image

🎥 Video Detection

Add a screenshot of the Video Detection tab here.

Show Image

🧠 How It Works

Spectra doesn't rely on a single trained ML classifier. Instead, each module combines several heuristic signals into a weighted "human-ness" score, then converts that into a label + confidence percentage.

Text Detector (text_detector.py)

Analyzes writing style using:

Burstiness — variation in sentence length (humans are "bursty"; AI is more uniform)
Punctuation richness — use of dashes, semicolons, parentheses, etc.
Repetition — how often 3-word phrases repeat
AI filler phrases — detection of common LLM phrasing (e.g. "furthermore," "delve into," "in conclusion")
Vocabulary richness — ratio of unique to total content words
Image Detector (image_detector.py)

Analyzes the uploaded image using:

Noise residual — high-pass/edge filtering to detect natural sensor noise vs. AI smoothness
Local block entropy — texture variation across image regions
Chroma channel balance — smoothness of color channels
Histogram spikiness — real photos tend to have "spikier" pixel-value histograms
Edge coherence — how "clean" vs. natural detected edges are
Video Detector (video_detector.py)

Samples up to 60 frames and analyzes:

Motion irregularity — natural motion is unpredictable; AI motion is often too smooth
Sensor noise — temporal noise in static regions (a fingerprint of real cameras)
Edge consistency across frames
Histogram variation over time
Spatial motion coherence — whether motion is localized (real) or uniform across the frame (AI)


🗂 Project Structure
ai-detector new/
├── app.py               # Streamlit UI — tabs, styling, and result rendering
├── text_detector.py      # Text authenticity analysis
├── image_detector.py     # Image authenticity analysis
├── video_detector.py     # Video authenticity analysis
└── space.jpg             # Background image used by the UI


⚙️ Requirements
Python 3.11+
Streamlit
NumPy
Pillow (PIL)
OpenCV (opencv-python)

Install dependencies:

bash
pip install streamlit numpy pillow opencv-python
🚀 Getting Started
Clone or download this repository.
Make sure space.jpg is in the same folder as app.py (used as the background image).
Install the dependencies above.
Run the app:
bash
streamlit run app.py
Open the local URL Streamlit prints (usually http://localhost:8501) in your browser.
Choose a tab — Text, Image, or Video — upload/paste your content, and click Analyze.


🛠 Tech Stack
Layer	Technology
UI/Frontend	Streamlit + custom CSS
Text Analysis	Regex + NumPy heuristics
Image Analysis	Pillow + NumPy
Video Analysis	OpenCV + NumPy
