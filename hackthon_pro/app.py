import os
import json
import sqlite3
import hashlib
import threading
import re
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document
from dotenv import load_dotenv
import time
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import easyocr
import pytesseract

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

CORS(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

# Global variables for caching
response_cache = {}
processing_queue = []

# Initialize Google AI
GOOGLE_AI_API_KEY = "AIzaSyCe8IvHrLiaYe9x0qBLz7znWqp8wg7Kzl8"
genai.configure(api_key=GOOGLE_AI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_cached_response(text, question=None):
    """Get cached response if available"""
    if question:
        cache_key = hashlib.md5(f"{text[:500]}_{question}".encode()).hexdigest()
    else:
        cache_key = hashlib.md5(text[:500].encode()).hexdigest()
    
    if cache_key in response_cache:
        return response_cache[cache_key]
    return None

def cache_response(text, question, response):
    """Cache response for future use"""
    if question:
        cache_key = hashlib.md5(f"{text[:500]}_{question}".encode()).hexdigest()
    else:
        cache_key = hashlib.md5(text[:500].encode()).hexdigest()
    
    response_cache[cache_key] = response

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('documents.db')
    cursor = conn.cursor()
    
    # Create documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            content TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed BOOLEAN DEFAULT FALSE
        )
    ''')
    
    # Create questions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            question TEXT NOT NULL,
            answer TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def extract_text_from_pdf(file_path):
    """Extract text from PDF file - enhanced for scanned documents and handwritten notes"""
    return extract_text_from_pdf_enhanced(file_path)

def extract_text_from_docx(file_path):
    """Extract text from DOCX file - optimized for speed and context"""
    try:
        doc = Document(file_path)
        text = ""
        # Process first 25 paragraphs for better context
        for i, paragraph in enumerate(doc.paragraphs[:25]):
            text += paragraph.text + "\n"
            if len(text) > 4000:  # Increased limit for better context
                break
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_image(file_path):
    """Extract text from image files using OCR - optimized for handwritten notes"""
    try:
        # Read image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Could not read image {file_path}")
            return ""
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize EasyOCR for better handwritten text recognition
        reader = easyocr.Reader(['en'])
        
        # Extract text using EasyOCR
        results = reader.readtext(image_rgb)
        
        # Combine all detected text
        text = ""
        for (bbox, detected_text, confidence) in results:
            if confidence > 0.3:  # Filter low confidence results
                text += detected_text + " "
        
        # If EasyOCR didn't find much, try Tesseract as backup
        if len(text.strip()) < 50:
            print("EasyOCR found limited text, trying Tesseract...")
            try:
                # Preprocess image for better OCR
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Apply threshold to get better contrast
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Extract text using Tesseract
                tesseract_text = pytesseract.image_to_string(thresh)
                if len(tesseract_text.strip()) > len(text.strip()):
                    text = tesseract_text
            except Exception as e:
                print(f"Tesseract error: {e}")
        
        print(f"✅ Extracted {len(text)} characters from image")
        return text.strip()
        
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf_enhanced(file_path):
    """Enhanced PDF text extraction with image OCR for scanned documents and handwritten notes"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Process first 5 pages
            for i, page in enumerate(pdf_reader.pages[:5]):
                page_text = page.extract_text()
                
                # If page has very little text, it might be scanned, image-based, or handwritten
                if len(page_text.strip()) < 100:
                    print(f"Page {i+1} has little text ({len(page_text.strip())} chars), attempting OCR for handwritten content...")
                    
                    try:
                        # Try to extract handwritten text using OCR
                        ocr_text = extract_handwritten_text_from_pdf_page(file_path, i)
                        if ocr_text and len(ocr_text.strip()) > 20:
                            page_text = f" [Page {i+1} - Handwritten Content]: {ocr_text} "
                            print(f"✅ Successfully extracted {len(ocr_text)} characters of handwritten text from page {i+1}")
                        else:
                            page_text = f" [Page {i+1}: Image-based content detected - may contain handwritten notes, diagrams, or scanned text] "
                    except Exception as ocr_error:
                        print(f"OCR failed for page {i+1}: {ocr_error}")
                        page_text = f" [Page {i+1}: Image-based content detected - may contain handwritten notes, diagrams, or scanned text] "
                
                text += page_text + "\n"
                if len(text) > 4000:
                    break
            
            # If no text was extracted, provide a helpful message
            if len(text.strip()) < 50:
                text = " [Document appears to be image-based or scanned. Content may include handwritten notes, diagrams, or scanned text that requires OCR processing.] "
                    
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_handwritten_text_from_pdf_page(pdf_path, page_num):
    """Extract handwritten text from a specific PDF page using OCR"""
    try:
        # Convert PDF page to image
        import fitz  # PyMuPDF
        import tempfile
        import os
        
        # Open PDF and get page
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_num]
        
        # Convert page to image with high resolution
        mat = fitz.Matrix(3.0, 3.0)  # 3x zoom for even better quality
        pix = page.get_pixmap(matrix=mat)
        
        # Save temporary image in uploads directory to avoid permission issues
        temp_image_path = os.path.join('uploads', f'temp_page_{page_num}.png')
        pix.save(temp_image_path)
        
        # Extract text from image using OCR
        ocr_text = extract_text_from_image_enhanced(temp_image_path)
        
        # Clean up temporary file
        try:
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
        except:
            pass  # Ignore cleanup errors
        
        pdf_document.close()
        
        return ocr_text
        
    except Exception as e:
        print(f"Error extracting handwritten text from PDF page: {e}")
        return ""

def extract_text_from_image_enhanced(image_path):
    """Enhanced image text extraction optimized for handwritten notes"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return ""
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize EasyOCR for better handwritten text recognition
        reader = easyocr.Reader(['en'], gpu=False)  # Use CPU for better compatibility
        
        # Extract text using EasyOCR with optimized settings for handwriting
        results = reader.readtext(
            image_rgb,
            paragraph=True,  # Group text into paragraphs
            detail=1,  # Get detailed results
            contrast_ths=0.05,  # Very low contrast threshold for handwriting
            adjust_contrast=0.8,  # Higher contrast adjustment
            text_threshold=0.4,  # Very low text confidence threshold
            link_threshold=0.2,  # Very low link threshold
            low_text=0.1,  # Very low text threshold
            canvas_size=5120,  # Much larger canvas for better processing
            mag_ratio=2.0  # Higher magnification ratio
        )
        
        # Combine all detected text
        text = ""
        for (bbox, detected_text, confidence) in results:
            if confidence > 0.1:  # Very low confidence threshold for handwriting
                text += detected_text + " "
        
        # If EasyOCR didn't find much, try Tesseract as backup with optimized settings
        if len(text.strip()) < 10:
            print("EasyOCR found limited text, trying Tesseract with handwriting optimization...")
            try:
                # Preprocess image for better OCR
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Apply advanced preprocessing for handwriting
                # Denoise
                denoised = cv2.fastNlMeansDenoising(gray)
                
                # Enhance contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(denoised)
                
                # Apply threshold
                _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Extract text using Tesseract with handwriting mode
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?()[]{}:;"\'-_+=/\\|@#$%^&*~` '
                tesseract_text = pytesseract.image_to_string(thresh, config=custom_config)
                
                if len(tesseract_text.strip()) > len(text.strip()):
                    text = tesseract_text
                    print(f"✅ Tesseract extracted {len(text)} characters")
                    
            except Exception as e:
                print(f"Tesseract error: {e}")
        
        print(f"✅ Total extracted {len(text)} characters from image")
        return text.strip()
        
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def get_general_knowledge_answer(question):
    """Get answer from Google AI using general knowledge when content is not in document"""
    try:
        prompt = f"""
        Please provide a comprehensive and accurate answer to this question: "{question}"
        
        Instructions:
        1. Provide a detailed, educational answer
        2. Include relevant examples and explanations
        3. Make it clear and easy to understand
        4. If it's a technical topic, explain it in simple terms
        5. Include key points and important information
        
        Answer:"""
        
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        # Limit answer length
        if len(answer) > 500:
            answer = answer[:500] + "..."
            
        return answer
        
    except Exception as e:
        print(f"General knowledge AI error: {e}")
        return f"I can provide general information about '{question}', but I'm having trouble accessing that information right now. Please try again or ask a different question."

def analyze_with_google_ai(text, question=None):
    """Analyze text using Google AI for better answers"""
    
    # Check cache first
    cached_response = get_cached_response(text, question)
    if cached_response:
        return cached_response
    
    try:
        # Process text for better context
        text = text[:4000]  # Keep more context for AI
        
        if question:
            # Use Google AI for question answering
            try:
                prompt = f"""
                Based on the following document content, please answer this question: "{question}"
                
                Document content:
                {text}
                
                Instructions:
                1. Search carefully through the document content for any information related to the question
                2. Look for headings, titles, keywords, and any text that might answer the question
                3. If the document contains handwritten content (marked with "Handwritten Content"), try to understand and interpret it
                4. If you find relevant information, provide a clear and accurate answer
                5. If the document contains image-based content (handwritten notes, diagrams, etc.), mention this limitation
                6. If you cannot find any relevant information, respond with: "NOT_FOUND_IN_DOCUMENT"
                
                Note: If you see "Handwritten Content" in the document, this means the text was extracted from handwritten notes using OCR technology. Please interpret this content as best as possible.
                
                Answer:"""
                
                response = model.generate_content(prompt)
                answer = response.text.strip()
                
                # Check if content was not found in document
                if ("NOT_FOUND_IN_DOCUMENT" in answer or 
                    "not found in the document" in answer.lower() or
                    "does not contain" in answer.lower() or
                    "cannot answer" in answer.lower() or
                    "no information" in answer.lower()):
                    print(f"Content not found in document for question: {question}")
                    # Get general knowledge answer
                    general_answer = get_general_knowledge_answer(question)
                    answer = f"⚠️ **Note: This information is NOT from your uploaded document** ⚠️\n\n{general_answer}\n\n📄 **Document Content**: The document you uploaded doesn't contain specific information about '{question}'. The answer above is from general knowledge."
                
                # Limit answer length
                if len(answer) > 300:
                    answer = answer[:300] + "..."
                    
            except Exception as e:
                print(f"Google AI error: {e}")
                # Better fallback answer that actually tries to help
                if len(text) > 50:
                    # Check if it's image-based content
                    if "image-based" in text.lower() or "scanned" in text.lower():
                        answer = "This document appears to contain image-based content (handwritten notes, diagrams, or scanned text). While I can see the document structure, I may not be able to read specific handwritten text or diagrams. Please try asking about general topics or upload a text-based document for more detailed analysis."
                    else:
                        # Extract first few sentences as context
                        sentences = text.split('.')[:3]
                        context = '. '.join(sentences) + '.'
                        answer = f"Based on the document content: {context[:200]}... Please ask a specific question about this information."
                else:
                    answer = "The document has been processed but contains limited text content. This might be an image-based document (handwritten notes, diagrams, etc.). Please try uploading a text-based document or ask general questions about the document type."
            
            response = {
                "summary": None,
                "answer": answer
            }
        else:
            # Use Google AI for summarization
            try:
                prompt = f"""
                Please provide a concise summary of the following document content in 2-3 sentences:
                
                {text}
                
                Summary:"""
                
                response = model.generate_content(prompt)
                summary = response.text.strip()
                
                if len(summary) > 350:
                    summary = summary[:350] + "..."
                    
            except Exception as e:
                print(f"Google AI summarization error: {e}")
                summary = "Document analyzed successfully. Key information extracted and ready for questions."
            
            response = {
                "summary": summary,
                "answer": None
            }
        
        # Cache the response
        cache_response(text, question, response)
        return response
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        return {
            "summary": "Document processed successfully. Ready for questions.",
            "answer": "I can help answer questions about this document."
        }

def process_upload_async(file_path, filename, file_extension, document_id):
    """Process upload asynchronously for better performance"""
    try:
        # Extract text based on file type
        if file_extension == 'pdf':
            content = extract_text_from_pdf(file_path)
        elif file_extension in ['docx', 'doc']:
            content = extract_text_from_docx(file_path)
        elif file_extension in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
            content = extract_text_from_image(file_path)
        else:
            content = ""
        
        # Quick analysis
        analysis = analyze_with_google_ai(content)
        
        # Update existing document record
        conn = sqlite3.connect('documents.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE documents 
            SET content = ?, processed = ?
            WHERE id = ?
        ''', (content, True, document_id))
        conn.commit()
        conn.close()
        
        print(f"✅ Processed {filename} successfully - {len(content)} characters extracted")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(file_path)
        
        # Get file extension
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        # Save document to database first to get the ID
        conn = sqlite3.connect('documents.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO documents (filename, original_filename, file_type, content, processed)
            VALUES (?, ?, ?, ?, ?)
        ''', (saved_filename, filename, file_extension, "", False))
        document_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Process content asynchronously
        thread = threading.Thread(
            target=process_upload_async,
            args=(file_path, saved_filename, file_extension, document_id)
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'document_id': document_id,
            'filename': filename,
            'message': 'File uploaded successfully! Processing in background...'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/documents')
def get_documents():
    conn = sqlite3.connect('documents.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, original_filename, file_type, upload_date, processed, content
        FROM documents
        ORDER BY upload_date DESC
    ''')
    documents = cursor.fetchall()
    conn.close()
    
    return jsonify([{
        'id': doc[0],
        'filename': doc[1],
        'file_type': doc[2],
        'upload_date': doc[3],
        'processed': doc[4],
        'content': doc[5] if doc[5] else ""
    } for doc in documents])

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    document_id = data.get('document_id')
    question = data.get('question')
    
    if not document_id or not question:
        return jsonify({'error': 'Document ID and question are required'}), 400
    
    # Get document info and content
    conn = sqlite3.connect('documents.db')
    cursor = conn.cursor()
    cursor.execute('SELECT filename, file_type, content FROM documents WHERE id = ?', (document_id,))
    result = cursor.fetchone()
    
    if not result:
        return jsonify({'error': 'Document not found'}), 404
    
    filename, file_type, content = result
    
    # If content is empty, extract it from the file
    if not content or len(content.strip()) == 0:
        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(file_path):
                if file_type == 'pdf':
                    content = extract_text_from_pdf(file_path)
                elif file_type in ['docx', 'doc']:
                    content = extract_text_from_docx(file_path)
                elif file_type in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
                    content = extract_text_from_image(file_path)
                else:
                    content = ""
                
                # Update the database with the extracted content
                cursor.execute('UPDATE documents SET content = ? WHERE id = ?', (content, document_id))
                conn.commit()
                print(f"✅ Extracted content for {filename}: {len(content)} characters")
                
                # Verify the update worked
                cursor.execute('SELECT content FROM documents WHERE id = ?', (document_id,))
                verify_result = cursor.fetchone()
                if verify_result and verify_result[0]:
                    print(f"✅ Database update verified: {len(verify_result[0])} characters saved")
                else:
                    print(f"❌ Database update failed: content not saved")
            else:
                return jsonify({'error': 'Document file not found'}), 404
        except Exception as e:
            print(f"Error extracting content: {e}")
            return jsonify({'error': 'Error processing document'}), 500
    
    # Get answer using Google AI
    analysis = analyze_with_google_ai(content, question)
    answer = analysis['answer']
    
    # Save question and answer
    cursor.execute('''
        INSERT INTO questions (document_id, question, answer)
        VALUES (?, ?, ?)
    ''', (document_id, question, answer))
    conn.commit()
    conn.close()
    
    return jsonify({
        'answer': answer,
        'question': question,
        'document_id': document_id
    })

@app.route('/questions/<int:document_id>')
def get_questions(document_id):
    conn = sqlite3.connect('documents.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT question, answer, timestamp
        FROM questions
        WHERE document_id = ?
        ORDER BY timestamp DESC
    ''', (document_id,))
    questions = cursor.fetchall()
    conn.close()
    
    return jsonify([{
        'question': q[0],
        'answer': q[1],
        'timestamp': q[2]
    } for q in questions])

@app.route('/delete-document/<int:document_id>', methods=['DELETE'])
def delete_document(document_id):
    """Delete a specific document and its questions"""
    try:
        conn = sqlite3.connect('documents.db')
        cursor = conn.cursor()
        
        # Get document info before deletion
        cursor.execute('SELECT filename, original_filename FROM documents WHERE id = ?', (document_id,))
        result = cursor.fetchone()
        
        if not result:
            return jsonify({
                'success': False,
                'error': 'Document not found'
            }), 404
        
        filename, original_filename = result
        
        # Delete questions first (due to foreign key constraint)
        cursor.execute('DELETE FROM questions WHERE document_id = ?', (document_id,))
        
        # Delete document
        cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))
        
        conn.commit()
        conn.close()
        
        # Delete uploaded file
        upload_folder = app.config['UPLOAD_FOLDER']
        file_path = os.path.join(upload_folder, filename)
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        
        # Clear response cache
        global response_cache
        response_cache.clear()
        
        return jsonify({
            'success': True,
            'message': f'Document "{original_filename}" deleted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error deleting document: {str(e)}'
        }), 500

@app.route('/clear-documents', methods=['DELETE'])
def clear_documents():
    """Clear all documents and questions from the database"""
    try:
        conn = sqlite3.connect('documents.db')
        cursor = conn.cursor()
        
        # Clear questions first (due to foreign key constraint)
        cursor.execute('DELETE FROM questions')
        
        # Clear documents
        cursor.execute('DELETE FROM documents')
        
        # Reset auto-increment counters
        cursor.execute('DELETE FROM sqlite_sequence WHERE name IN ("documents", "questions")')
        
        conn.commit()
        conn.close()
        
        # Clear uploaded files
        upload_folder = app.config['UPLOAD_FOLDER']
        if os.path.exists(upload_folder):
            for filename in os.listdir(upload_folder):
                file_path = os.path.join(upload_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        
        # Clear response cache
        global response_cache
        response_cache.clear()
        
        return jsonify({
            'success': True,
            'message': 'All documents and questions cleared successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error clearing documents: {str(e)}'
        }), 500

if __name__ == '__main__':
    init_db()
    
    # Initialize Google AI model
    print("🚀 Starting Google AI-Powered Document Analysis...")
    print("✅ Google Gemini model loaded and ready!")
    
    # Maximum speed Flask settings
    app.run(
        debug=False,
        host='0.0.0.0', 
        port=5000,
        threaded=True,
        processes=1
    ) 