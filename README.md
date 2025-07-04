# SaglamScan - AI Nutrition Analyzer

SaglamScan is a web application that uses the power of Google's Gemini Vision AI to analyze nutrition labels from images. Users can upload a photo of a product's nutrition facts, and the app will extract the data, provide a simple A-F health grade, and offer a nutritional summary.

## Core Features

*   **AI Data Extraction:** Upload an image of a nutrition label to have its contents automatically read and parsed.
*   **Health Grading:** Get an at-a-glance understanding of a product's healthiness with a simple A-F score.
*   **Product Comparison:** Upload two labels to see a side-by-side comparison and an AI-generated recommendation.
*   **User Accounts & History:** Sign up to save your scan history for future reference.
*   **Multilingual Support:** The interface is available in both English and Azerbaijani.

## Tech Stack

*   **Backend:** Python with Flask
*   **AI Vision & Analysis:** Google Gemini 1.5 Flash
*   **Database & Authentication:** Google Firebase (Firestore & Auth)
*   **Hosting:** Render.com
*   **Frontend:** Jinja2 Templating, HTML, CSS, Vanilla JavaScript

## Running Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Repla09/saglamscan-app.git
    cd saglamscan-app
    ```
2.  **Set up a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    *   Create a `serviceAccountKey.json` file for Firebase admin access.
    *   Set the `GEMINI_API_KEY` and `FIREBASE_WEB_API_KEY` environment variables.
5.  **Run the application:**
    ```bash
    flask run
    ```