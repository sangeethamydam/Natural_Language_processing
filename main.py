from flask import Flask, render_template, request, jsonify, send_file
import requests
import matplotlib.pyplot as plt
from io import BytesIO
from transformers import pipeline

app = Flask(__name__)
summarizer = pipeline("summarization")

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/generate_headline', methods=['POST'])
def generate_headline():
    text = request.form['text']

    # Use Transformers for headline generation
    headline = summarizer(text, max_length=10, min_length=5, length_penalty=2.0, num_beams=4, no_repeat_ngram_size=2)[0]['summary_text']

    return render_template('index.html', text=text, headline=headline)

@app.route('/train_metrics')
def train_metrics():
    # Placeholder for training metrics
    # You can replace this with actual training metrics
    epochs = list(range(1, 11))
    accuracy = [0.8, 0.85, 0.88, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]

    # Plot training metrics
    plt.plot(epochs, accuracy, marker='o')
    plt.title('Training Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Save the plot to BytesIO
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close()

    # Send the plot as an image response
    img_bytes.seek(0)
    return send_file(img_bytes, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
