from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import tensorflow as tf
import requests
import os
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()

app = Flask(__name__)

def download_model():
    api = KaggleApi()
    api.authenticate()
    api.model_download_file('mataajohn/cassava_disease_detection', 'Cassava_Disease_Model.h5', path='models/')
    # Unzip if necessary
    os.system('unzip models/Cassava_Disease_Model.h5.zip -d models/')

def load_cassava_model():
    model_path = 'models/Cassava_Disease_Model.h5'
    if not os.path.exists(model_path):
        download_model()
    model = tf.keras.models.load_model(model_path)
    return model

# Load your trained model
model = load_cassava_model()

# Define the cassava disease labels
disease_labels = [
    "Healthy",  # 0
    "Cassava Bacterial Blight (CBB)",  # 1
    "Cassava Brown Streak Disease (CBSD)",  # 2
    "Cassava Green Mottle (CGM)",  # 3
    "Cassava Mosaic Disease (CMD)"  # 4
]

# Configure Generative AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0.7,  # Adjust temperature for more varied responses
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 600,  # Limit the length of the response
    "response_mime_type": "text/plain",
}
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

gen_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction="Respond as a knowledgeable and friendly agricultural expert specifically for Zambian farmers. Provide clear, concise, and minimal explanations. Avoid technical jargon and keep responses brief. Tailor your response to the specific question and offer hopeful solutions. Do not answer anything out of agriculture. If the question is related to location, use the coordinates to provide the city or location name. If the question is not related to farming, respond politely and redirect to farming-related topics.",
)

chat_session = gen_model.start_chat(history=[])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((200, 200))  # Resize image to match model input size
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.expand_dims(img, 0)  # Add batch dimension
        prediction = model.predict(img)
        # Assuming your model returns a class index
        predicted_class = tf.argmax(prediction[0]).numpy()
        disease_name = disease_labels[predicted_class]
        return jsonify({'prediction': disease_name})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    location = request.json.get('location')
    
    if location and 'location' in user_message.lower():
        location_name = get_location_name(location['latitude'], location['longitude'])
        model_response = f"Your location is approximately {location_name}. How can I assist you with your farming needs in this area?"
    elif location and 'weather' in user_message.lower():
        weather_info = get_weather(location['latitude'], location['longitude'])
        model_response = f"The current weather at your location is {weather_info}. Understanding your local weather patterns is crucial for successful farming. Do you have any questions about how weather impacts specific crops in Zambia, or how to plan your planting schedule around rainfall?"
    else:
        try:
            response = chat_session.send_message(user_message)
            model_response = response.text
        except ResourceExhausted:
            model_response = "Sorry, the service is currently experiencing high demand. Please try again later."
    
    chat_session.history.append({"role": "user", "parts": [user_message]})
    chat_session.history.append({"role": "model", "parts": [model_response]})
    return jsonify({'response': model_response})

def get_weather(latitude, longitude):
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        return f"{weather_description} with a temperature of {temperature}°C"
    else:
        return "unable to fetch weather data at the moment"

def get_location_name(latitude, longitude):
    api_key = os.getenv("OPENCAGE_API_KEY")
    url = f'https://api.opencagedata.com/geocode/v1/json?q={latitude}+{longitude}&key={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print(data)  # Debugging information
        if data['results']:
            components = data['results'][0]['components']
            country = components.get('country', 'Zambia')
            state = components.get("state", "Copperbelt")
            return f"{country}, {state}"
        else:
            return "an unknown location"
    else:
        print(response.text)  # Debugging information
        return "unable to fetch location data at the moment"

if __name__ == '__main__':
    app.run(debug=True)