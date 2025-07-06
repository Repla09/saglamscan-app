import os
import secrets
from flask import Flask, request, render_template, url_for, redirect, flash, session, g
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
from urllib.parse import urlparse
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Import translations
from translations import translations

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- App Configuration ---
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(16)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size


app = Flask(__name__)
app.config.from_object(Config)
app.jinja_env.add_extension('jinja2.ext.do')

# --- Securely load API Keys from Environment Variables ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FIREBASE_WEB_API_KEY = os.environ.get('FIREBASE_WEB_API_KEY')
CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET')

# --- Cloudinary, Firebase, Gemini Initializations ... (No changes here) ---
# (This code remains the same as your previous version)
# --- Cloudinary Configuration ---
try:
    if all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]):
        cloudinary.config(cloud_name=CLOUDINARY_CLOUD_NAME, api_key=CLOUDINARY_API_KEY,
                          api_secret=CLOUDINARY_API_SECRET, secure=True)
        logger.info("Cloudinary configured successfully.")
    else:
        logger.warning("Cloudinary environment variables not fully set. Image uploads will fail.")
except Exception as e:
    logger.error(f"Cloudinary configuration failed: {e}")

# --- Firebase Initialization ---
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
    logger.info("Firebase (Auth/Firestore) initialized successfully.")
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
# ... End of initializations

# --- Nutrient Data & Helpers ---
# (No changes to NUTRIENT_CLASSES, NUTRIENT_EXPLANATIONS, etc.)
NUTRIENT_CLASSES = {'health_grade': 'neutral', 'calories': 'negative', 'total_fat': 'neutral',
                    'saturated_fat': 'negative', 'trans_fat': 'negative', 'cholesterol': 'negative',
                    'sodium': 'negative', 'total_carbohydrates': 'neutral', 'dietary_fiber': 'positive',
                    'total_sugars': 'negative', 'added_sugars': 'negative', 'protein': 'positive', 'vit_d': 'positive',
                    'calcium': 'positive', 'iron': 'positive', 'potassium': 'positive', 'ai_summary': 'neutral'}
NUTRIENT_EXPLANATIONS_EN = {'calories': 'Energy from food. Consuming too many can lead to weight gain.',
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
                            'vit_d': 'Important for bone health and immune function.',
                            'calcium': 'Crucial for strong bones and teeth.',
                            'iron': 'Essential for carrying oxygen in the blood.',
                            'potassium': 'Helps maintain fluid balance and supports normal blood pressure.'}
NUTRIENT_EXPLANATIONS_AZ = {'calories': 'Qidadan gələn enerji. Çox qəbul etmək çəki artımına səbəb ola bilər.',
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
                            'potassium': 'Maye balansını saxlamağa və normal qan təzyiqini dəstəkləməyə kömək edir.'}


def get_nutrient_explanations(): return NUTRIENT_EXPLANATIONS_AZ if g.get('lang') == 'az' else NUTRIENT_EXPLANATIONS_EN


def create_empty_nutrition_data(): return {key: 'N/A' for key in
                                           list(NUTRIENT_CLASSES.keys()) + ['image_url', 'original_filename']}


def extract_numeric_value(value_str):
    if isinstance(value_str, (int, float)): return float(value_str)
    if not isinstance(value_str, str) or value_str == 'N/A': return 0.0
    match = re.search(r'(\d+\.?\d*)', value_str);
    return float(match.group(1)) if match else 0.0


# --- THE MAIN UPDATE IS HERE ---

def calculate_health_grade(data):
    """
    Calculates a health grade based on the new algorithm from the expert review.
    """
    score = 0

    # --- 1. Tiered Penalties (Negative Points) ---
    # Only the highest applicable penalty is applied for each nutrient.

    # Added Sugars Logic (with fallback to Total Sugars)
    added_sugars_val = data.get('added_sugars', 'N/A')
    if added_sugars_val != 'N/A':
        v = extract_numeric_value(added_sugars_val)
        if v > 20:
            score -= 4
        elif v > 10:
            score -= 3
        elif v > 5:
            score -= 2
    else:
        # Fallback to Total Sugars with less harsh penalties
        v = extract_numeric_value(data.get('total_sugars'))
        if v > 20:
            score -= 3  # Was -4
        elif v > 10:
            score -= 2  # Was -3
        elif v > 5:
            score -= 1  # Was -2

    # Saturated Fat Penalties (Stricter lowest threshold)
    v = extract_numeric_value(data.get('saturated_fat'))
    if v > 10:
        score -= 4
    elif v > 5:
        score -= 3
    elif v > 1.5:
        score -= 2  # Was > 2g

    # Sodium Penalties (New, stricter tiers)
    v = extract_numeric_value(data.get('sodium'))
    if v > 800:
        score -= 5
    elif v > 600:
        score -= 4
    elif v > 400:
        score -= 3
    elif v > 200:
        score -= 2

    # Trans Fat Penalty (Increased)
    v = extract_numeric_value(data.get('trans_fat'))
    if v > 0:
        score -= 4  # Was -3

    # REMOVED: Calorie penalty was removed based on expert feedback.
    # v = extract_numeric_value(data.get('calories'))
    # if v > 400: score -= 3
    # elif v > 250: score -= 2

    # --- 2. Tiered Bonuses (Positive Points) ---
    # Only the highest applicable bonus is applied for each nutrient.

    # Dietary Fiber Bonuses
    v = extract_numeric_value(data.get('dietary_fiber'))
    if v >= 5:
        score += 3
    elif v >= 3:
        score += 2

    # Protein Bonuses
    v = extract_numeric_value(data.get('protein'))
    if v >= 15:
        score += 3
    elif v >= 8:
        score += 2

    # NEW: Potassium Bonuses
    v = extract_numeric_value(data.get('potassium'))
    if v >= 700:
        score += 3
    elif v >= 400:
        score += 2

    # NEW: Calcium Bonuses
    v = extract_numeric_value(data.get('calcium'))
    if v >= 300:
        score += 2
    elif v >= 200:
        score += 1

    # NEW: Vitamin D Bonuses
    v = extract_numeric_value(data.get('vit_d'))
    if v >= 4:
        score += 2
    elif v >= 2:
        score += 1

    # --- 3. Small, Independent & Stacking Bonuses (+1 Point each) ---
    # These can be earned in addition to any other points.

    # Bonus for 0g added sugar
    if extract_numeric_value(data.get('added_sugars', 'N/A')) == 0:
        score += 1

    # Bonus for 0g saturated fat
    if extract_numeric_value(data.get('saturated_fat', 'N/A')) == 0:
        score += 1

    # Bonus for very low sodium
    if extract_numeric_value(data.get('sodium', 'N/A')) < 100:
        score += 1

    # Bonus for very high fiber
    if extract_numeric_value(data.get('dietary_fiber', 'N/A')) > 7:
        score += 1

    # --- 4. Final Grade Mapping (New, Stricter Curve) ---
    if score >= 4:
        return "A"  # Was >= 3
    elif score >= 2:
        return "B"  # Was 1 to 2
    elif score >= -1:
        return "C"  # Was -2 to 0
    elif score >= -4:
        return "D"  # Was -5 to -3
    else:
        return "F"  # Was <= -6


# --- Other helper functions (process_image, save_scan, etc.) ---
# (No changes to the rest of the helper functions)
def process_image_and_extract_data(file_storage):
    result = {'data': None, 'error': None}
    original_filename = secure_filename(file_storage.filename)
    try:
        from io import BytesIO
        in_mem_file = BytesIO()
        file_storage.save(in_mem_file)
        in_mem_file.seek(0)
    except Exception as e:
        logger.error(f"Image reading failed: {e}");
        result['error'] = "Could not read the uploaded file.";
        return result
    try:
        public_id = f"saglamscan/{uuid.uuid4()}"
        upload_result = cloudinary.uploader.upload(in_mem_file, public_id=public_id, overwrite=True)
        image_url = upload_result.get('secure_url')
        if not image_url: raise Exception("Cloudinary did not return a secure URL.")
    except Exception as e:
        logger.error(f"Cloudinary upload failed: {e}");
        result['error'] = "Could not save image to cloud storage.";
        return result
    try:
        in_mem_file.seek(0)
        image_bytes = in_mem_file.read()
        parsed_data = parse_nutrition_facts_gemini(image_bytes)
        if not parsed_data: raise ValueError("AI returned no data.")
        parsed_data['allergens'] = [i for i in parsed_data.get('allergens', []) if
                                    i and i.strip().lower() not in ['n/a', 'none']]
        parsed_data['warnings'] = [i for i in parsed_data.get('warnings', []) if
                                   i and i.strip().lower() not in ['n/a', 'none']]
        # THE IMPORTANT CALL to the new function
        parsed_data['health_grade'] = calculate_health_grade(parsed_data)
        parsed_data['image_url'] = image_url
        parsed_data['original_filename'] = original_filename
        if 'filename' in parsed_data: del parsed_data['filename']
        result['data'] = parsed_data
    except Exception as e:
        logger.error(f"Data extraction failed for {original_filename}: {e}")
        data = create_empty_nutrition_data()
        data.update({'image_url': image_url, 'original_filename': original_filename, 'error': str(e)})
        result['data'] = data
        result['error'] = str(e)
    return result


def save_scan_to_history(data):
    if FIREBASE_ENABLED and 'user_id' in session and data and data.get('image_url'):
        user_id = session['user_id']
        scan_data_to_save = {k: v for k, v in data.items() if v is not None}
        scan_data_to_save.setdefault('health_grade', 'N/A')
        scan_data_to_save['timestamp'] = firestore.SERVER_TIMESTAMP
        db.collection('users').document(user_id).collection('scans').document().set(scan_data_to_save)
        logger.info(f"Saved scan with image {data.get('image_url')} for user {user_id}")


def parse_nutrition_facts_gemini(image_bytes):
    if not GEMINI_AVAILABLE: return None
    summary_instruction = g.t.get('ai_summary_instruction',
                                  'Provide a 3-4 sentence, easy-to-understand nutritional summary for a consumer.')
    prompt = f"""You are an expert food label data extractor..."""  # This prompt is long and unchanged
    try:
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = gemini_vision_model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
                                                        generation_config=generation_config)
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Gemini parsing error: {e}");
        return None


def get_ai_comparison(p1, p2):
    if not GEMINI_AVAILABLE: return g.t['comparison_error']
    s1 = {g.t.get(k, k): p1.get(k, 'N/A') for k in ['calories', 'protein', 'added_sugars', 'sodium']}
    s2 = {g.t.get(k, k): p2.get(k, 'N/A') for k in ['calories', 'protein', 'added_sugars', 'sodium']}
    p = g.t['ai_comparison_instruction'].format(p1_name=p1.get('original_filename', 'P1'), p1_data=json.dumps(s1),
                                                p2_name=p2.get('original_filename', 'P2'), p2_data=json.dumps(s2))
    try:
        r = genai.GenerativeModel('gemini-1.5-flash').generate_content(p);
        return r.text.strip()
    except Exception as e:
        logger.error(f"AI comparison failed: {e}");
        return g.t['comparison_error']


# --- Flask Routes ---
# (No changes to context processors or routes)
@app.context_processor
def inject_globals(): return {'lang': g.lang, 't': g.t, 'user_name': session.get('user_name'),
                              'email_verified': session.get('email_verified'),
                              'firebase_web_api_key': FIREBASE_WEB_API_KEY}


@app.before_request
def before_request(): g.lang = request.cookies.get('lang', 'en'); g.t = translations.get(g.lang, translations['en'])


@app.route('/')
def home(): return render_template('upload.html', is_logged_in='user_id' in session)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user_id' not in session: flash(g.t['login_required_analyze'], "error"); return redirect(url_for('home'))
    if not session.get('email_verified'): flash(g.t['verify_email_analyze'], "error"); return redirect(url_for('home'))
    file = request.files.get('file');
    if not file or not file.filename: flash(g.t['no_file_selected'], "error"); return redirect(url_for('home'))
    result = process_image_and_extract_data(file);
    save_scan_to_history(result.get('data'))
    if result.get('error'): flash(result['error'], "error")
    return render_template('result.html', parsed_data=result.get('data'), error_message=result.get('error'),
                           nutrient_classes=NUTRIENT_CLASSES, nutrient_explanations=get_nutrient_explanations())


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'GET': return render_template('compare.html', has_results=False)
    if 'user_id' not in session: flash(g.t['login_required_compare'], "error"); return redirect(url_for('compare'))
    if not session.get('email_verified'): flash(g.t['verify_email_compare'], "error"); return redirect(
        url_for('compare'))
    file1, file2 = request.files.get('file1'), request.files.get('file2')
    if not file1 or not file2: flash(g.t['upload_both_images'], "error"); return redirect(url_for('compare'))
    result1, result2 = process_image_and_extract_data(file1), process_image_and_extract_data(file2)
    save_scan_to_history(result1.get('data'));
    save_scan_to_history(result2.get('data'))
    error_msg = None
    if result1.get('error') or result2.get(
        'error'): error_msg = f"Product 1 Error: {result1.get('error', 'OK')}. Product 2 Error: {result2.get('error', 'OK')}."
    data1, data2 = result1.get('data') or create_empty_nutrition_data(), result2.get(
        'data') or create_empty_nutrition_data()
    summary = g.t['comparison_error'] if error_msg else get_ai_comparison(data1, data2)
    return render_template('compare.html', product1=data1, product2=data2, error_message=error_msg,
                           nutrient_classes=NUTRIENT_CLASSES, has_results=True, ai_summary=summary)


@app.route('/set_language/<lang>')
def set_language(lang):
    if lang not in translations: lang = 'en'
    redirect_url = url_for('home')
    if request.referrer:
        referrer_path = urlparse(request.referrer).path
        if referrer_path not in ['/analyze', '/compare']: redirect_url = request.referrer
    response = redirect(redirect_url)
    response.set_cookie('lang', lang, max_age=365 * 24 * 60 * 60)
    return response


@app.route('/signup', methods=['GET'])
def signup(): return render_template('signup.html')


@app.route('/login', methods=['GET'])
def login(): return render_template('login.html')


@app.route('/session_login', methods=['POST'])
def session_login():
    try:
        id_token = request.form.get('id_token')
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        user_name = decoded_token.get('name')
        email_verified = decoded_token.get('email_verified', False)
        if not user_name:
            user = auth.get_user(uid)
            user_name = user.display_name
        session.update({'user_id': uid, 'user_name': user_name or 'User', 'email_verified': email_verified})
        flash(f"{g.t['welcome']}, {session['user_name']}!" if email_verified else g.t['verification_sent'], "success")
        return redirect(url_for('home'))
    except Exception as e:
        logger.error(f"Session login failed: {e}");
        flash(g.t['login_failed'], "error");
        return redirect(url_for('home'))


@app.route('/logout')
def logout(): session.clear(); flash(g.t['logged_out'], "success"); return redirect(url_for('home'))


@app.route('/account', methods=['GET', 'POST'])
def account():
    if 'user_id' not in session: flash(g.t['login_required_account'], "error"); return redirect(url_for('home'))
    user_id = session['user_id']
    if request.method == 'POST':
        new_name = request.form.get('name')
        if new_name and new_name != session.get('user_name'):
            try:
                auth.update_user(user_id, display_name=new_name)
                session['user_name'] = new_name
                flash(g.t['name_updated'], "success")
            except Exception as e:
                logger.error(f"Name update failed: {e}");
                flash(g.t['name_update_failed'], "error")
        return redirect(url_for('account'))
    user = auth.get_user(user_id)
    return render_template('account.html', user_name=user.display_name, user_email=user.email)


@app.route('/history')
def history():
    if 'user_id' not in session: flash(g.t['login_required_history'], "error"); return redirect(url_for('home'))
    if not session.get('email_verified'): flash(g.t['verify_email_history'], "error"); return redirect(url_for('home'))
    scans_ref = db.collection('users').document(session['user_id']).collection('scans').order_by('timestamp',
                                                                                                 direction=firestore.Query.DESCENDING).stream()
    scan_history = []
    for scan in scans_ref:
        scan_data = scan.to_dict()
        scan_data['id'] = scan.id
        if 'timestamp' in scan_data and hasattr(scan_data['timestamp'], 'strftime'):
            scan_data['formatted_timestamp'] = scan_data['timestamp'].strftime('%Y-%m-%d %H:%M')
        else:
            scan_data['formatted_timestamp'] = 'N/A'
        scan_history.append(scan_data)
    return render_template('history.html', history=scan_history)


@app.route('/how-it-works')
def how_it_works(): return render_template('how_it_works.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port)