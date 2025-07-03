import os
import secrets
from flask import Flask, request, render_template, url_for, redirect, flash, session, send_from_directory, g
from PIL import Image
import re
import uuid
import logging
import json
from werkzeug.utils import secure_filename
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, auth, firestore
import google.generativeai as genai
from urllib.parse import urlparse, parse_qs

# Import translations
from translations import translations

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- App Configuration ---
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(16)
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    MAX_IMAGE_SIZE = (2048, 2048)


app = Flask(__name__)
app.config.from_object(Config)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.jinja_env.add_extension('jinja2.ext.do')

# --- Securely load API Keys from Environment Variables ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FIREBASE_WEB_API_KEY = os.environ.get('FIREBASE_WEB_API_KEY')  # For frontend auth

# --- Firebase Initialization (Backend) ---
try:
    firebase_service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if firebase_service_account_json:
        cred_dict = json.loads(firebase_service_account_json)
        cred = credentials.Certificate(cred_dict)
    else:
        logger.warning("Loading Firebase credentials from serviceAccountKey.json for local development.")
        cred = credentials.Certificate('serviceAccountKey.json')

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    FIREBASE_ENABLED = True
    logger.info("Firebase initialized successfully.")
except Exception as e:
    FIREBASE_ENABLED = False
    logger.error(f"Firebase initialization failed: {e}")

# --- Gemini API Initialization ---
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_vision_model = genai.GenerativeModel('gemini-1.5-flash')
        GEMINI_AVAILABLE = True
        logger.info("Gemini API configured successfully.")
    else:
        GEMINI_AVAILABLE = False
        logger.warning("GEMINI_API_KEY environment variable not set. AI features will be disabled.")
except Exception as e:
    GEMINI_AVAILABLE = False
    logger.error(f"Gemini initialization failed: {e}")

# --- Nutrient Data & Helpers ---
NUTRIENT_CLASSES = {'health_grade': 'neutral', 'calories': 'negative', 'total_fat': 'neutral',
                    'saturated_fat': 'negative', 'trans_fat': 'negative', 'cholesterol': 'negative',
                    'sodium': 'negative', 'total_carbohydrates': 'neutral', 'dietary_fiber': 'positive',
                    'total_sugars': 'negative', 'added_sugars': 'negative', 'protein': 'positive', 'vit_d': 'positive',
                    'calcium': 'positive', 'iron': 'positive', 'potassium': 'positive', 'ai_summary': 'neutral'}

NUTRIENT_EXPLANATIONS_EN = {
    'calories': 'Energy from food. Consuming too many can lead to weight gain.',
    'total_fat': 'An essential macronutrient, but some types are healthier than others.',
    'saturated_fat': 'A type of fat that can raise LDL (bad) cholesterol. Recommended to limit.',
    'trans_fat': 'An unhealthy fat that raises bad cholesterol and lowers good cholesterol. Best to avoid.',
    'cholesterol': 'High intake of dietary cholesterol can contribute to high blood cholesterol in some people.',
    'sodium': 'An essential mineral, but high intake is linked to high blood pressure.',
    'total_carbohydrates': "The body's main source of energy.",
    'dietary_fiber': 'A type of carbohydrate that aids digestion and helps maintain stable blood sugar. Higher is better!',
    'total_sugars': 'Includes both natural and added sugars. High intake can contribute to health issues.',
    'added_sugars': 'Sugars added during processing. The AHA recommends limiting these for better health.',
    'protein': 'Essential for building and repairing tissues, and for immune function. A key building block for the body.',
    'vit_d': 'Important for bone health and immune function.', 'calcium': 'Crucial for strong bones and teeth.',
    'iron': 'Essential for carrying oxygen in the blood.',
    'potassium': 'Helps maintain fluid balance and supports normal blood pressure.'
}
NUTRIENT_EXPLANATIONS_AZ = {
    'calories': 'Qidadan gələn enerji. Çox qəbul etmək çəki artımına səbəb ola bilər.',
    'total_fat': 'Vacib bir makronutrientdir, lakin bəzi növləri digərlərindən daha sağlamdır.',
    'saturated_fat': 'LDL (pis) xolesterolu yüksəldə bilən bir yağ növüdür. Məhdudlaşdırmaq tövsiyə olunur.',
    'trans_fat': 'Pis xolesterolu yüksəldən və yaxşı xolesterolu azaldan zərərli bir yağdır. Uzaq durmaq ən yaxşısıdır.',
    'cholesterol': 'Bəzi insanlarda qida ilə yüksək xolesterol qəbulu qan xolesterolunun yüksəlməsinə səbəb ola bilər.',
    'sodium': 'Vacib bir mineraldır, lakin yüksək qəbulu yüksək qan təzyiqi ilə əlaqələndirilir.',
    'total_carbohydrates': "Bədənin əsas enerji mənbəyidir.",
    'dietary_fiber': 'Həzmi asanlaşdıran və qan şəkərinin sabit qalmasına kömək edən bir karbohidrat növüdür. Nə qədər çox olsa o qədər yaxşıdır!',
    'total_sugars': 'Həm təbii, həm də əlavə edilmiş şəkərləri əhatə edir. Yüksək qəbulu sağlamlıq problemlərinə səbəb ola bilər.',
    'added_sugars': 'İstehsal prosesində əlavə edilmiş şəkərlərdir. AHA daha yaxşı sağlamlıq üçün bunları məhdudlaşdırmağı tövsiyə edir.',
    'protein': 'Toxumaların qurulması və bərpası, habelə immun funksiyası üçün vacibdir. Bədənin əsas tikinti materialıdır.',
    'vit_d': 'Sümük sağlamlığı və immun funksiyası üçün vacibdir.',
    'calcium': 'Güclü sümüklər və dişlər üçün həyati əhəmiyyət daşıyır.',
    'iron': 'Qanda oksigen daşınması üçün zəruridir.',
    'potassium': 'Maye balansını saxlamağa və normal qan təzyiqini dəstəkləməyə kömək edir.'
}


def get_nutrient_explanations():
    return NUTRIENT_EXPLANATIONS_AZ if g.get('lang') == 'az' else NUTRIENT_EXPLANATIONS_EN


def create_empty_nutrition_data():
    return {key: 'N/A' for key in
            list(NUTRIENT_CLASSES.keys()) + ['serving_size', 'servings_per_container', 'ingredient_list', 'allergens',
                                             'warnings', 'filename', 'original_filename']}


def process_image_and_extract_data(file_storage):
    result = {'data': None, 'error': None}
    original_filename = secure_filename(file_storage.filename)
    unique_filename = f"{uuid.uuid4()}{os.path.splitext(original_filename)[1]}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    try:
        file_storage.save(filepath)
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            if img.mode != 'RGB': img = img.convert('RGB')
            if max(img.size) > max(app.config['MAX_IMAGE_SIZE']): img.thumbnail(app.config['MAX_IMAGE_SIZE'],
                                                                                Image.Resampling.LANCZOS)
            img.save(filepath, 'JPEG', quality=90)
    except Exception as e:
        if os.path.exists(filepath): os.remove(filepath)
        logger.error(f"Image processing failed: {e}");
        result['error'] = "Invalid or corrupted image file.";
        return result
    try:
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        parsed_data = parse_nutrition_facts_gemini(image_bytes)
        if not parsed_data: raise ValueError("AI returned no data.")

        parsed_data['allergens'] = [i for i in parsed_data.get('allergens', []) if
                                    i and i.strip().lower() not in ['n/a', 'none']]
        parsed_data['warnings'] = [i for i in parsed_data.get('warnings', []) if
                                   i and i.strip().lower() not in ['n/a', 'none']]
        parsed_data['health_grade'] = calculate_health_grade(parsed_data)
        parsed_data['filename'] = unique_filename
        parsed_data['original_filename'] = original_filename
        result['data'] = parsed_data
    except Exception as e:
        logger.error(f"Data extraction failed for {unique_filename}: {e}")
        data = create_empty_nutrition_data();
        data.update({'filename': unique_filename, 'original_filename': original_filename})
        result['data'] = data
        result['error'] = str(e)
    return result


def process_existing_file(filename):
    from werkzeug.datastructures import FileStorage
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return {'error': f"File {filename} not found."}
    file_storage = FileStorage(stream=open(filepath, 'rb'), filename=filename)
    return process_image_and_extract_data(file_storage)


def save_scan_to_history(data):
    if FIREBASE_ENABLED and 'user_id' in session and data and 'filename' in data:
        user_id = session['user_id']
        scan_data_to_save = {k: v for k, v in data.items() if v is not None}
        scan_data_to_save.setdefault('health_grade', 'N/A')
        scan_data_to_save['timestamp'] = firestore.SERVER_TIMESTAMP
        db.collection('users').document(user_id).collection('scans').document().set(scan_data_to_save)
        logger.info(f"Saved scan {data.get('filename')} for user {user_id}")


def parse_nutrition_facts_gemini(image_bytes):
    if not GEMINI_AVAILABLE: return None
    summary_instruction = g.t.get('ai_summary_instruction',
                                  'Provide a 3-4 sentence, easy-to-understand nutritional summary for a consumer.')
    prompt = f"""
    You are an expert food label data extractor. Analyze the image and return ONLY a single, valid JSON object.
    Do not include any text or markdown formatting outside the JSON object.
    Required JSON structure:
    {{"serving_size": "...", "servings_per_container": "...", "calories": "...", "total_fat": "...", "saturated_fat": "...", "trans_fat": "...", "cholesterol": "...", "sodium": "...", "total_carbohydrates": "...", "dietary_fiber": "...", "total_sugars": "...", "added_sugars": "...", "protein": "...", "vit_d": "...", "calcium": "...", "iron": "...", "potassium": "...", "ingredient_list": "...", "allergens": ["...", "..."], "warnings": ["...", "..."], "ai_summary": "..."}}
    Instructions:
    - Use "N/A" for any missing values.
    - In `allergens`, list common allergens based on the ingredient list.
    - In `warnings`, list ingredients of concern like artificial sweeteners or colors.
    - In the `ai_summary` field: {summary_instruction}
    """
    try:
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = gemini_vision_model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
                                                        generation_config=generation_config)
        full_data = create_empty_nutrition_data();
        full_data.update(json.loads(response.text))
        return full_data
    except Exception as e:
        logger.error(f"Gemini parsing error: {e}");
        return None


def extract_numeric_value(value_str):
    if isinstance(value_str, (int, float)): return float(value_str)
    if not isinstance(value_str, str) or value_str == 'N/A': return 0.0
    match = re.search(r'(\d+\.?\d*)', value_str);
    return float(match.group(1)) if match else 0.0


def calculate_health_grade(data):
    score = 0;
    penalties = {'added_sugars': [(20, -4), (10, -3), (5, -2)], 'saturated_fat': [(10, -4), (5, -3), (2, -2)],
                 'sodium': [(600, -4), (400, -3), (200, -2)], 'trans_fat': [(0, -3)],
                 'calories': [(400, -3), (250, -2)]}
    bonuses = {'dietary_fiber': [(5, 3), (3, 2)], 'protein': [(15, 3), (8, 2)]}
    for nutrient, thresholds in penalties.items():
        value = extract_numeric_value(data.get(nutrient))
        if nutrient == 'added_sugars' and data.get(nutrient, 'N/A') == 'N/A': value = extract_numeric_value(
            data.get('total_sugars'))
        for threshold, penalty in thresholds:
            if value > threshold: score += penalty; break
    for nutrient, thresholds in bonuses.items():
        value = extract_numeric_value(data.get(nutrient));
        for threshold, bonus in thresholds:
            if value >= threshold: score += bonus; break
    if score >= 3:
        return "A"
    elif score >= 1:
        return "B"
    elif score >= -2:
        return "C"
    elif score >= -5:
        return "D"
    else:
        return "F"


def get_ai_comparison(product1_data, product2_data):
    if not GEMINI_AVAILABLE: return g.t['comparison_error']
    p1_simple = {g.t.get(k, k): product1_data.get(k, 'N/A') for k in ['calories', 'protein', 'added_sugars', 'sodium']}
    p2_simple = {g.t.get(k, k): product2_data.get(k, 'N/A') for k in ['calories', 'protein', 'added_sugars', 'sodium']}
    prompt = g.t['ai_comparison_instruction'].format(p1_name=product1_data.get('original_filename', 'P1'),
                                                     p1_data=json.dumps(p1_simple),
                                                     p2_name=product2_data.get('original_filename', 'P2'),
                                                     p2_data=json.dumps(p2_simple))
    try:
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt);
        return response.text.strip()
    except Exception as e:
        logger.error(f"AI comparison failed: {e}");
        return g.t['comparison_error']


@app.context_processor
def inject_globals():
    """Injects variables into all templates."""
    return {
        'lang': g.lang,
        't': g.t,
        'user_name': session.get('user_name'),
        'email_verified': session.get('email_verified'),
        'firebase_web_api_key': FIREBASE_WEB_API_KEY
    }


@app.before_request
def before_request():
    g.lang = request.cookies.get('lang', 'en');
    g.t = translations.get(g.lang, translations['en'])


@app.route('/')
def home(): return render_template('upload.html', is_logged_in='user_id' in session)


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if 'user_id' not in session: flash(g.t['login_required_analyze'], "error"); return redirect(url_for('home'))
    if not session.get('email_verified'): flash(g.t['verify_email_analyze'], "error"); return redirect(url_for('home'))

    if request.method == 'POST':
        file_storage = request.files.get('file');
        if not file_storage or not file_storage.filename: flash(g.t['no_file_selected'], "error"); return redirect(
            url_for('home'))
        result = process_image_and_extract_data(file_storage);
        save_scan_to_history(result.get('data'))
    else:  # GET
        filename = request.args.get('filename')
        if not filename: return redirect(url_for('home'))
        result = process_existing_file(filename)

    if result.get('error'): flash(result['error'], "error")
    return render_template('result.html', parsed_data=result.get('data'), error_message=result.get('error'),
                           nutrient_classes=NUTRIENT_CLASSES, nutrient_explanations=get_nutrient_explanations())


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if 'user_id' not in session: flash(g.t['login_required_compare'], "error"); return redirect(url_for('home'))
    if not session.get('email_verified'): flash(g.t['verify_email_compare'], "error"); return redirect(url_for('home'))

    if request.method == 'POST':
        file1, file2 = request.files.get('file1'), request.files.get('file2')
        if not file1 or not file2: return render_template('compare.html', has_results=False,
                                                          error_message=g.t['upload_both_images'])
        result1, result2 = process_image_and_extract_data(file1), process_image_and_extract_data(file2)
        save_scan_to_history(result1.get('data'));
        save_scan_to_history(result2.get('data'))
    elif request.args.get('file1') and request.args.get('file2'):
        filename1, filename2 = request.args.get('file1'), request.args.get('file2')
        result1, result2 = process_existing_file(filename1), process_existing_file(filename2)
    else:
        return render_template('compare.html', has_results=False)

    error_message = None
    if result1.get('error') or result2.get(
            'error'): error_message = f"P1: {result1.get('error', 'OK')}. P2: {result2.get('error', 'OK')}."
    product1_data, product2_data = result1.get('data') or create_empty_nutrition_data(), result2.get(
        'data') or create_empty_nutrition_data()
    ai_comparison_summary = g.t['comparison_error'] if error_message else get_ai_comparison(product1_data,
                                                                                            product2_data)

    return render_template('compare.html', product1=product1_data, product2=product2_data, error_message=error_message,
                           nutrient_classes=NUTRIENT_CLASSES, has_results=True, ai_summary=ai_comparison_summary)


@app.route('/set_language/<lang>')
def set_language(lang):
    if lang not in translations or lang == g.lang:
        return redirect(request.referrer or url_for('home'))

    response = redirect(url_for('home'))  # Default redirect
    if request.referrer:
        parsed_url = urlparse(request.referrer)
        referrer_path = parsed_url.path

        if referrer_path == url_for('analyze'):
            filename = parse_qs(parsed_url.query).get('filename', [None])[0]
            if filename: response = redirect(url_for('analyze', filename=filename))
        elif referrer_path == url_for('compare'):
            params = parse_qs(parsed_url.query)
            file1, file2 = params.get('file1', [None])[0], params.get('file2', [None])[0]
            if file1 and file2: response = redirect(url_for('compare', file1=file1, file2=file2))
        else:
            response = redirect(request.referrer)

    response.set_cookie('lang', lang, max_age=365 * 24 * 60 * 60)
    return response


@app.route('/processed_images/<filename>')
def get_processed_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/signup', methods=['GET'])
def signup(): return render_template('signup.html')


@app.route('/login', methods=['GET'])
def login(): return render_template('login.html')

# --- NEW ROUTE FOR EMAIL VERIFICATION LANDING PAGE ---
@app.route('/verified')
def verified():
    # This page now handles its own messaging.
    # We show a success message on the page itself.
    return render_template('verified.html')
# --- END OF NEW ROUTE ---


@app.route('/session_login', methods=['POST'])
def session_login():
    try:
        id_token = request.form.get('id_token');
        decoded_token = auth.verify_id_token(id_token, check_revoked=True);
        user = auth.get_user(decoded_token['uid'])
        session.update(
            {'user_id': user.uid, 'user_name': user.display_name or 'User', 'email_verified': user.email_verified})
        flash(f"{g.t['welcome']}, {session['user_name']}!" if user.email_verified else g.t['verification_sent'],
              "success")
        return redirect(url_for('home'))
    except Exception as e:
        logger.error(f"Session login failed: {e}");
        flash(g.t['login_failed'], "error");
        return redirect(url_for('home'))


@app.route('/logout')
def logout():
    session.clear();
    flash(g.t['logged_out'], "success");
    return redirect(url_for('home'))


@app.route('/account', methods=['GET', 'POST'])
def account():
    if 'user_id' not in session: flash(g.t['login_required_account'], "error"); return redirect(url_for('home'))
    user_id = session['user_id']
    if request.method == 'POST':
        new_name = request.form.get('name')
        if new_name and new_name != session.get('user_name'):
            try:
                auth.update_user(user_id, display_name=new_name);
                session['user_name'] = new_name;
                flash(
                    g.t['name_updated'], "success")
            except Exception as e:
                logger.error(f"Failed to update user name: {e}");
                flash(g.t['name_update_failed'], "error")
        return redirect(url_for('account'))
    user = auth.get_user(user_id);
    return render_template('account.html', user_name=user.display_name, user_email=user.email)


@app.route('/history')
def history():
    if 'user_id' not in session: flash(g.t['login_required_history'], "error"); return redirect(url_for('home'))
    if not session.get('email_verified'): flash(g.t['verify_email_history'], "error"); return redirect(url_for('home'))
    scans_ref = db.collection('users').document(session['user_id']).collection('scans').order_by('timestamp',
                                                                                                 direction=firestore.Query.DESCENDING).stream()
    scan_history = []
    for scan in scans_ref:
        scan_data = scan.to_dict();
        scan_data['id'] = scan.id
        scan_data['formatted_timestamp'] = scan_data['timestamp'].strftime(
            '%Y-%m-%d %H:%M') if 'timestamp' in scan_data and hasattr(scan_data['timestamp'], 'strftime') else 'N/A'
        scan_history.append(scan_data)
    return render_template('history.html', history=scan_history)


@app.route('/how-it-works')
def how_it_works(): return render_template('how_it_works.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001));
    app.run(debug=True, host='0.0.0.0', port=port)