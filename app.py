# app.py
from flask import Flask, render_template, request, jsonify
import os
import requests
import wikipedia
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import io
import base64
from diffusers import StableDiffusionPipeline
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models
# PDF Summarization models
extractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
abstractive_summarizer = pipeline("summarization", model="t5-base")

# Translation models
translator = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")
translator_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")

# Code generation model
code_generator = pipeline("text-generation", model="Salesforce/codegen-350M-mono")

# Text-to-image model (Stable Diffusion)
device = "cuda" if torch.cuda.is_available() else "cpu"
stable_diffusion = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
if device == "cuda":
    stable_diffusion.to(device)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pdf-summary', methods=['POST'])
def pdf_summary():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'})
    
    file = request.files['pdf']
    model_choice = request.form.get('model', 'extractive')
    word_count = int(request.form.get('word_count', 150))
    
    # Process PDF (using a placeholder for now)
    # In real implementation, you'd use PyPDF2 or pdfminer to extract text
    text = "This is a placeholder for PDF text extraction"
    
    # Summarize based on model choice
    if model_choice == 'extractive':
        summary = extractive_summarizer(text, max_length=word_count, min_length=30, do_sample=False)[0]['summary_text']
    else:
        summary = abstractive_summarizer(text, max_length=word_count, min_length=30, do_sample=False)[0]['summary_text']
    
    key_points = ["Key point 1", "Key point 2", "Key point 3"]  # Placeholder
    
    return jsonify({
        'summary': summary,
        'key_points': key_points
    })

@app.route('/wiki-summary', methods=['POST'])
def wiki_summary():
    topic = request.form.get('topic')
    word_count = int(request.form.get('word_count', 150))
    
    try:
        wiki_text = wikipedia.summary(topic, sentences=10)
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(wiki_text, max_length=word_count, min_length=30, do_sample=False)[0]['summary_text']
        
        return jsonify({
            'summary': summary,
            'original_text': wiki_text
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/wiki-question', methods=['POST'])
def wiki_question():
    topic = request.form.get('topic')
    question = request.form.get('question')
    
    try:
        wiki_text = wikipedia.summary(topic, sentences=15)
        qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
        answer = qa_model(question=question, context=wiki_text)
        
        return jsonify({
            'answer': answer['answer'],
            'confidence': float(answer['score'])
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/translate', methods=['POST'])
def translate():
    text = request.form.get('text')
    target_lang = request.form.get('language')
    
    # Map target language to appropriate model
    lang_map = {
        'hindi': 'en-hi',
        'bengali': 'en-bn',
        'korean': 'en-ko',
        'french': 'en-fr',
        'spanish': 'en-es',
        'japanese': 'en-ja'
    }
    
    try:
        # In a real implementation, you'd use the appropriate model for each language
        # This is a simplified example
        inputs = translator_tokenizer(text, return_tensors="pt")
        outputs = translator.generate(**inputs)
        translated_text = translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            'translated_text': translated_text,
            'language': target_lang
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/image-generation', methods=['POST'])
def generate_image():
    prompt = request.form.get('prompt')
    
    try:
        image = stable_diffusion(prompt).images[0]
        
        # Convert PIL image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'image': img_str
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/code-generation', methods=['POST'])
def generate_code():
    question = request.form.get('question')
    
    try:
        generated_code = code_generator(
            question,
            max_length=500,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )[0]['generated_text']
        
        # Extract the code part from the generated text
        # This is a simplified approach
        code_part = generated_code.replace(question, "").strip()
        
        return jsonify({
            'code': code_part
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
