from flask import Blueprint, render_template, request, flash, jsonify, session
from flask_login import login_required, current_user
from flask import redirect, url_for
from . import db
import json
from flask_mail import Message
from . import mail
import os
import traceback
from website.chatbot import chatbot, process_message  # Import the chatbot function and process_message
from groq import Groq
from langdetect import detect # type: ignore
from .translations.translations import translations


views = Blueprint('views', __name__)
chatbot_instance = None

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_system_prompt(language):
    """Get system prompt based on language"""
    if language == 'zh':  # Chinese
        return """你是Sally，一个友好的助手。请遵循以下规则：
        1. 保持回答简短（最多2-3句话）
        2. 直接回答问题
        3. 每个回答最多使用一个表情符号
        4. 把每条消息都当作持续对话的一部分
        5. 不要自我介绍
        6. 不要解释你的能力
        7. 除非被要求，否则不要讲笑话"""
    
    elif language == 'ms':  # Malay
        return """Anda adalah Sally, pembantu yang mesra. Ikuti peraturan ini:
        1. Pastikan jawapan ringkas (2-3 ayat sahaja)
        2. Jawab soalan secara terus
        3. Gunakan maksimum satu emoji setiap jawapan
        4. Anggap setiap mesej sebagai sebahagian perbualan yang berterusan
        5. Jangan memperkenalkan diri
        6. Jangan terangkan kebolehan anda
        7. Jangan bercerita jenaka kecuali diminta"""
    
    else:  # Default English
        return """You are Sally, a friendly assistant. Follow these rules strictly:
        1. Never introduce yourself during conversation
        2. Keep responses brief (2-3 sentences maximum)
        3. Answer questions directly
        4. Use at most one emoji per response
        5. Don't explain what you can or can't do
        6. Don't tell jokes unless asked
        7. Treat every message as part of an ongoing conversation"""

def _(text):
    """Translate text based on current language"""
    current_lang = session.get('language', 'en')
    return translations[current_lang].get(text, text)

@views.route('/')
@login_required
def home():
    return render_template("home.html", user=current_user, _=_)

@views.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            # Get form data
            name = request.form.get('name')
            email = request.form.get('email')
            subject = request.form.get('subject')
            message = request.form.get('message')
            
            # Debug prints
            print("Form data received:")
            print(f"Name: {name}")
            print(f"Email: {email}")
            print(f"Subject: {subject}")
            print(f"Message: {message}")
            print(f"Using MAIL_USERNAME: {os.getenv('MAIL_USERNAME')}")
            
            if not all([name, email, subject, message]):
                flash('Please fill in all fields', 'error')
                return redirect(url_for('views.contact'))
            
            msg = Message(
                subject=f"New Contact Form Message: {subject}",
                recipients=[os.getenv('MAIL_USERNAME')],
                sender=os.getenv('MAIL_USERNAME'),
                reply_to=email,
                body=f"""
                New message from your website contact form:
                
                From: {name}
                Email: {email}
                Subject: {subject}
                
                Message:
                {message}
                """
            )
            
            print("Attempting to send email...")
            mail.send(msg)
            print("Email sent successfully!")
            
            flash('Thank you! Your message has been sent successfully.', 'success')
            
        except Exception as e:
            print("Error details:")
            print(traceback.format_exc())
            flash(f'Sorry, there was an error sending your message: {str(e)}', 'error')
        
        return redirect(url_for('views.contact'))
        
    return render_template("contact.html", user=current_user, _=_)

@views.route('/guide')
@login_required
def guide():
    return render_template("guide.html", user=current_user, _=_)

@views.route('/chat')
def chat():
    return render_template("chatbot.html", user=current_user, _=_)

@views.route('/chat/message', methods=['POST'])
def chat_message():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
        
    try:
        # Use session language
        language = session.get('language', 'en')
        
        # Get response from chatbot
        response = process_message(user_message)
        print(f"Chatbot response: {response}")
        
        # Translate response
        translated_response = _(response)
        
        return jsonify({'response': translated_response})
    except Exception as e:
        print(f"Error in chat_message: {str(e)}")
        return jsonify({'error': 'Sorry, there was an error processing your message.'}), 500
    
@views.route('/guide/overview')
@login_required
def guide_overview():
    return render_template("guide_overview.html", user=current_user, _=_)

@views.route('/guide/operation')
@login_required
def guide_operation():
    return render_template("guide_operation.html", user=current_user, _=_)

@views.route('/guide/maintenance-check')
@login_required
def maintenance_check():
    return render_template('guide_maintenance_check.html', user=current_user, _=_)

@views.route('/guide/stock-testing')
@login_required
def stock_testing():
    return render_template('guide_stock_testing.html', user=current_user, _=_)

@views.route('/guide/normal-production')
@login_required
def normal_production():
    return render_template('guide_normal_production.html', user=current_user, _=_)

@views.route('/change-language', methods=['POST'])
def change_language():
    language = request.json.get('language', 'en')
    session['language'] = language
    print(f"Language changed to: {language}")  # Debugging statement
    return jsonify({'success': True})

@views.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@views.route('/login')
def login():
    return render_template("login.html", _=_)

@views.route('/sign-up')
def sign_up():
    return render_template("sign_up.html", _=_)




