from flask import Flask, render_template, request
import requests
import time
from werkzeug.utils import secure_filename
from api_secrets import API_KEY_ASSEMBLYAI
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

app = Flask(__name__)

# AssemblyAI API configuration
upload_endpoint = 'https://api.assemblyai.com/v2/upload'
transcript_endpoint = 'https://api.assemblyai.com/v2/transcript'
headers_auth_only = {'authorization': API_KEY_ASSEMBLYAI}
headers = {
    "authorization": API_KEY_ASSEMBLYAI,
    "content-type": "application/json"
}

# Constants
CHUNK_SIZE = 5_242_880  # 5MB

# MBart model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

def read_file(filename):
    with open(filename, 'rb') as f:
        while True:
            data = f.read(CHUNK_SIZE)
            if not data:
                break
            yield data

def upload_audio(filename):
    response = requests.post(upload_endpoint, headers=headers_auth_only, data=read_file(filename))
    return response.json()['upload_url']

def transcribe_audio(audio_url):
    transcript_request = {'audio_url': audio_url}
    response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)
    return response.json()['id']

def poll_transcription(transcript_id):
    polling_endpoint = transcript_endpoint + '/' + transcript_id
    while True:
        response = requests.get(polling_endpoint, headers=headers)
        data = response.json()
        if data['status'] == 'completed':
            return data['text'], None
        elif data['status'] == 'error':
            return None, data['error']
        time.sleep(5)

def translate(text, target_lang):
    model_inputs = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
    )
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translation[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = secure_filename(file.filename)
        file.save(filename)
        audio_url = upload_audio(filename)
        text, error = poll_transcription(transcribe_audio(audio_url))
        if text:
            return render_template('transcript.html', transcript=text)
        elif error:
            return f"Error: {error}"

@app.route('/translate', methods=['POST'])
def translate_route():
    text = request.form['text']
    language = request.form['language']
    translated_text = translate(text, language)
    return render_template('translation.html', original_text=text, translation=translated_text)

if __name__ == '__main__':
    app.run(debug=True)
