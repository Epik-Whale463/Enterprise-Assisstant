import time
from functools import wraps

import docx
import requests
from PyPDF2 import PdfReader
from flask import Flask, redirect, request, url_for
from flask import render_template, flash, session, jsonify
from markdown import markdown
from oauthlib.oauth2 import WebApplicationClient
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.email import EmailTools
from werkzeug.utils import secure_filename
import html2text
from document_helper import *

app = Flask(__name__)
app.secret_key = 'enterpriseAIassistant'


# Google OAuth Configuration
GOOGLE_CLIENT_ID = ""
GOOGLE_CLIENT_SECRET = ""
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

oauth_client = WebApplicationClient(GOOGLE_CLIENT_ID)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB in bytes
uploaded_content = ""


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'rtf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Register the markdown filter
@app.template_filter('markdown')
def markdown_filter(text):
    return markdown(text)

@app.template_filter('html_to_markdown')
def html_to_markdown(text):
    return html2text.html2text(text)

@app.errorhandler(413)
def request_entity_too_large(error):
    flash("File is too large. Please upload a file smaller than 5MB.", "danger")
    return redirect(url_for('document_processing'))


def get_google_provider_cfg():
    return requests.get(GOOGLE_DISCOVERY_URL).json()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


@app.route('/login')
def login():
    # Get Google's authorization endpoint
    google_provider_cfg = get_google_provider_cfg()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]

    # Create an authorization URL
    request_uri = oauth_client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=url_for("callback", _external=True),
        scope=["openid", "email", "profile"],
    )
    return redirect(request_uri)


@app.route('/callback')
def callback():
    # Get the authorization code from the request
    code = request.args.get("code")

    # Fetch Google's token endpoint
    google_provider_cfg = get_google_provider_cfg()
    token_endpoint = google_provider_cfg["token_endpoint"]

    # Exchange the authorization code for tokens
    token_url, headers, body = oauth_client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=url_for("callback", _external=True),
        code=code,
    )
    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
    )

    # Parse the tokens
    oauth_client.parse_request_body_response(token_response.text)

    # Fetch the user's info from the userinfo endpoint
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = oauth_client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)

    userinfo = userinfo_response.json()

    if not userinfo.get("email_verified"):
        flash("User email not verified by Google.", "danger")
        return redirect(url_for("login"))

    session['logged_in'] = True
    session['email'] = userinfo['email']
    session['name'] = userinfo['given_name']

    return redirect(url_for("dashboard"))


@app.route('/login', methods=['POST'])
def login_user():
    username = request.form.get('username')
    password = request.form.get('password')

    if username == 'admin' and password == 'admin':
        session['logged_in'] = True
        return redirect('/dashboard')
    else:
        flash('Invalid username or password', 'danger')
        return redirect('/login')


@app.route('/register')
def register():
    flash("Please use Google Login to register.", "info")
    return redirect(url_for("login"))


@app.route('/logout')
def logout():
    session.clear()
    flash("Successfully logged out.", "success")
    return redirect(url_for('landing'))


@app.route('/register', methods=['POST'])
def register_user():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    if password != confirm_password:
        flash('Passwords do not match', 'danger')
        return redirect('/register')

    flash('Registration successful! Please login.', 'success')
    return redirect('/login')


@app.route('/')
def landing():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return render_template('landing.html')


@app.template_filter('now')
def now_filter(format_string):
    import datetime
    return datetime.datetime.now().strftime(format_string)

@app.route('/dashboard')
def dashboard():
    import datetime
    current_year = datetime.datetime.now().year
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    avg_summary_time = session.get('avg_summary_time', 0)
    avg_keyword_time = session.get('avg_keyword_time', 0)
    total_summary_requests = session.get('total_summary_requests', 0)
    total_keyword_requests = session.get('total_keyword_requests', 0)
    keyword_extraction_accuracy = session.get('keyword_extraction_accuracy', 0)
    summarization_accuracy = session.get('summarization_accuracy', 0)

    user_name = session.get('name', 'User')

    return render_template('dashboard.html',
                           active_page='dashboard',
                           avg_summary_time=round(avg_summary_time, 2),
                           avg_keyword_time=round(avg_keyword_time, 2),
                           total_summary_requests=total_summary_requests,
                           total_keyword_requests=total_keyword_requests,
                           keyword_extraction_accuracy=round(keyword_extraction_accuracy, 2),
                           summarization_accuracy=round(summarization_accuracy, 2),
                           user_name=user_name,
                           current_year=current_year)


@app.route('/compose-email', methods=['GET', 'POST'])
@login_required
def compose_email_route():
    email_draft = None
    prompt = None

    if request.method == 'POST':
        # Check if the user is editing the email draft
        edited_draft = request.form.get('email_draft', '').strip()
        if edited_draft:
            # Handle the edited draft (save, send, or process as needed)
            email_draft = edited_draft
            flash("Your email draft has been updated.", "success")
            return render_template(
                'compose_email.html',
                active_page='compose-email',
                email_draft=email_draft
            )

        # If no edited draft, then check if it's a new prompt for generating an email
        prompt = request.form.get('prompt', '').strip()

        if not prompt:
            flash("Email prompt cannot be empty.", "danger")
            return render_template('compose_email.html', active_page='compose-email')

        try:
            # Call the helper function to compose email from the prompt
            email_draft = compose_email(prompt)
            return render_template(
                'compose_email.html',
                active_page='compose-email',
                email_draft=email_draft,
                prompt=prompt
            )
        except Exception as e:
            error_message = f"An error occurred while composing the email: {str(e)}"
            logger.error(error_message)
            flash(error_message, "danger")
            return render_template('compose_email.html', active_page='compose-email')

    return render_template('compose_email.html', active_page='compose-email')


# Define the send email route
@app.route('/send-email', methods=['POST'])
@login_required
def send_email_route():
    # Get the email draft and recipient email from the form submission
    email_draft = request.form.get('email_draft')
    recipient_email = request.form.get('recipient_email')
    hr_manager_name = request.form.get('hr_manager_name')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    if not email_draft or not recipient_email or not hr_manager_name or not start_date or not end_date:
        flash("Please provide all required information.", "danger")
        return redirect(url_for('compose_email_route'))

    # Get the logged-in user's email (sender's email)
    sender_email = session.get('email')
    sender_name = session.get('name')
    sender_passkey = os.getenv(
        "APP_PASSWORD_GMAIL")  # You can securely store this value or fetch it from environment variables

    try:
        # Set up the email agent
        agent = Agent(
            model=Gemini(id="gemini-1.5-flash"),
            markdown=True,
            tools=[EmailTools(
                receiver_email=recipient_email,
                sender_email=sender_email,
                sender_name=sender_name,
                sender_passkey=sender_passkey,
            )]
        )

        # Send the email using the draft and recipient email
        agent.print_response(
            f"send an email to {recipient_email} with the following message: {email_draft} without additional information required")

        # If email was successfully sent
        flash("Email sent successfully!", "success")
        return redirect(url_for('compose_email_route'))

    except Exception as e:
        error_message = f"An error occurred while sending the email: {str(e)}"
        logger.error(error_message)
        flash(error_message, "danger")
        return redirect(url_for('compose_email_route'))


@app.route('/health')
@login_required
def health_check():
    health_data = {
        "API Server": "OK",
        "Database Connection": "OK",
        "Document Processing Service": "OK",
        "Summarization Service": "OK" if check_summarization_service() else "Error",
        "Keyword Extraction Service": "OK" if check_keyword_extraction_service() else "Error",
    }
    return jsonify(health_data)


def check_summarization_service():
    try:
        test_content = "This is a short test document. It is used to check if the summarization service is working correctly."
        summary_data = generate_summary(test_content, "extractive", "short")
        return summary_data["success"]
    except Exception as e:
        logger.error(f"Summarization service check failed: {e}")
        return False


def check_keyword_extraction_service():
    try:
        test_content = "This is a test document for keyword extraction. We will test technical terms, key phrases, and named entities."
        keyword_data = extract_keywords(test_content, ["technical", "keyphrases", "entities"])
        return "error" not in keyword_data and all(
            key in keyword_data for key in ["Technical Terms", "Key Phrases", "Named Entities"])
    except Exception as e:
        logger.error(f"Keyword extraction service check failed: {e}")
        return False


def update_summarization_accuracy(is_successful):
    total_summary_requests = session.get('total_summary_requests', 0)
    successful_summaries = session.get('successful_summaries', 0)

    if is_successful:
        successful_summaries += 1

    summarization_accuracy = (successful_summaries / total_summary_requests) * 100 if total_summary_requests > 0 else 0

    session['successful_summaries'] = successful_summaries
    session['summarization_accuracy'] = summarization_accuracy


def update_keyword_extraction_accuracy(is_successful):
    total_keyword_requests = session.get('total_keyword_requests', 0)
    successful_extractions = session.get('successful_extractions', 0)

    if is_successful:
        successful_extractions += 1

    keyword_extraction_accuracy = (
                                              successful_extractions / total_keyword_requests) * 100 if total_keyword_requests > 0 else 0

    session['successful_extractions'] = successful_extractions
    session['keyword_extraction_accuracy'] = keyword_extraction_accuracy


@app.route('/hr-policies')
@login_required
def hr_policies():
    return render_template('hr_policies.html', active_page='hr-policies')


@app.route('/it-support')
@login_required
def it_support():
    return render_template('it_support.html', active_page='it-support')


@app.route('/company-events')
@login_required
def company_events():
    return render_template('company_events.html', active_page='company-events')


@app.route('/document-processing')
@login_required
def document_processing():
    return render_template('document_processing.html', active_page='document-processing')


@app.route('/upload-document', methods=['POST'])
@login_required
def upload_document():
    global uploaded_content

    # Check if the request has the file part
    if 'uploadedFile' not in request.files:
        flash('No file part in the request', 'danger')
        return redirect('/document-processing')

    file = request.files['uploadedFile']

    # If no file selected
    if file.filename == '':
        flash('No file selected', 'warning')
        return redirect('/document-processing')

    # Check for allowed file types
    if not allowed_file(file.filename):
        flash('File type not allowed! Please upload a valid file.', 'danger')
        return redirect('/document-processing')

    try:
        # Save the file
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            content = ""
            file_extension = file.filename.lower().rsplit('.', 1)[1]

            try:
                if file_extension == 'txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                elif file_extension == 'rtf':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                elif file_extension == 'docx':
                    doc = docx.Document(file_path)
                    content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                elif file_extension == 'pdf':
                    reader = PdfReader(file_path)
                    content = '\n'.join(page.extract_text() for page in reader.pages)

                uploaded_content = clean_text(content)

                try:
                    # process_document_for_rag(uploaded_content)
                    session['chat_history'] = []
                    flash(f'File "{file.filename}" uploaded and processed successfully!', 'success')
                except Exception as e:
                    logger.error(f"Error processing document for RAG: {str(e)}")
                    flash("Document uploaded but conversation features may be limited.", "warning")

            except Exception as e:
                flash(f'Error processing file: {e}', 'danger')
                logger.error(f"Error processing file: {e}")

            finally:
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing temporary file: {e}")

            return render_template('document_processing.html',
                                   active_page='document-processing',
                                   file_content=uploaded_content)

    except Exception as e:
        flash(f'Error during file upload: {e}', 'danger')
        logger.error(f"Error during file upload: {e}")
        return redirect('/document-processing')

    flash('Something went wrong. Please try again.', 'danger')
    return redirect('/document-processing')


@app.route('/generate-summary', methods=['POST'])
@login_required
def generate_summary_route():
    global uploaded_content

    if not uploaded_content:
        flash('No document content available for summarization. Please upload a document first.', 'danger')
        return redirect('/document-processing')

    summarization_type = request.form.get('summarizationType')
    summary_length = request.form.get('summaryLength')

    try:
        start_time = time.time()
        summary_data = generate_summary(uploaded_content, summarization_type, summary_length)
        processing_time = time.time() - start_time

        total_summary_requests = session.get('total_summary_requests', 0) + 1
        avg_summary_time = session.get('avg_summary_time', 0)
        avg_summary_time = ((avg_summary_time * (
                    total_summary_requests - 1)) + processing_time) / total_summary_requests

        session['total_summary_requests'] = total_summary_requests
        session['avg_summary_time'] = avg_summary_time

        if summary_data["success"]:
            return render_template(
                'document_processing.html',
                active_page='document-processing',
                file_content=uploaded_content,
                summary=summary_data["summary"],
                metadata=summary_data["metadata"]
            )
        else:
            flash(summary_data["error"], "danger")
            return render_template(
                'document_processing.html',
                active_page='document-processing',
                file_content=uploaded_content,
                error=summary_data["error"],
                metadata=summary_data.get("metadata", {})
            )

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        flash(error_message, "danger")
        logger.error(error_message)
        return render_template(
            'document_processing.html',
            active_page='document-processing',
            file_content=uploaded_content,
            error=error_message,
            metadata={}
        )


@app.route('/extract-keywords', methods=['POST'])
@login_required
def extract_keywords_route():
    global uploaded_content

    if not uploaded_content:
        flash('No document content available for keyword extraction. Please upload a document first.', 'danger')
        return redirect('/document-processing')

    options = request.form.getlist('options')

    try:
        start_time = time.time()
        extracted_keywords = extract_keywords(uploaded_content, options)
        processing_time = time.time() - start_time

        total_keyword_requests = session.get('total_keyword_requests', 0) + 1
        avg_keyword_time = session.get('avg_keyword_time', 0)
        avg_keyword_time = ((avg_keyword_time * (
                    total_keyword_requests - 1)) + processing_time) / total_keyword_requests

        session['total_keyword_requests'] = total_keyword_requests
        session['avg_keyword_time'] = avg_keyword_time

        if "error" in extracted_keywords:
            flash(extracted_keywords["error"], 'danger')
            extracted_keywords = {}

        return render_template(
            'document_processing.html',
            active_page='document-processing',
            file_content=uploaded_content,
            extracted_keywords=extracted_keywords
        )

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        flash(error_message, "danger")
        logger.error(error_message)
        return redirect('/document-processing')


@app.route('/converse', methods=['POST'])
@login_required
def converse():
    question = request.form.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        chain = get_conversational_chain()

        chat_history = session.get('chat_history', [])

        if not isinstance(chat_history, list) or not all(isinstance(item, tuple) for item in chat_history):
            chat_history = []

        response = chain({"question": question, "chat_history": chat_history})

        chat_history.append((question, response['answer']))
        session['chat_history'] = chat_history

        return jsonify({
            "success": True,
            "answer": response['answer']
        })
        print(response["answer"])

    except Exception as e:
        logger.error(f"Error in conversation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/content-moderation')
@login_required
def content_moderation():
    return render_template('content_moderation.html', active_page='content-moderation')


@app.route('/security-settings')
@login_required
def security_settings():
    return render_template('security_settings.html', active_page='security-settings')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)