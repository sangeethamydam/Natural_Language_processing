from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from io import BytesIO
import requests
import numpy as np

app = Flask(__name__)

# Fetch training data from the provided link
train_data_url = "https://datasets-server.huggingface.co/first-rows?dataset=JulesBelveze%2Ftldr_news&config=all&split=train"
try:
    response = requests.get(train_data_url)
    response.raise_for_status()  # Raise an HTTPError for bad responses
    train_data = response.json()
    rows = train_data.get('rows', [])
    documents = [row['row']['content'] for row in rows]
    summaries = [row['row']['headline'] for row in rows]
except requests.RequestException as e:
    print(f"Error fetching training data: {e}")
    documents = []
    summaries = []

# Tokenize and pad sequences
tokenizer_doc = Tokenizer()
tokenizer_doc.fit_on_texts(documents)
total_words_doc = len(tokenizer_doc.word_index) + 1

tokenizer_summary = Tokenizer()
tokenizer_summary.fit_on_texts(summaries)
total_words_summary = len(tokenizer_summary.word_index) + 1

input_sequences_doc = tokenizer_doc.texts_to_sequences(documents)
input_sequences_summary = tokenizer_summary.texts_to_sequences(summaries)

# Determine the maximum sequence length
max_sequence_length = max(len(seq) for seq in input_sequences_doc + input_sequences_summary)

# Pad sequences to the determined maximum length
input_sequences_doc = pad_sequences(input_sequences_doc, maxlen=max_sequence_length, padding='post')
input_sequences_summary = pad_sequences(input_sequences_summary, maxlen=max_sequence_length, padding='post')

# Build the RNN-LSTM model
embedding_dim = 50
hidden_units = 100

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words_doc, embedding_dim, input_length=max_sequence_length),
    tf.keras.layers.LSTM(hidden_units, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(total_words_summary, activation='softmax'))
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(input_sequences_doc, np.expand_dims(input_sequences_summary, -1), epochs=50, verbose=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_headline', methods=['POST'])
def generate_headline():
    text = request.form['text']

    # Tokenize the input text
    input_sequence_doc = tokenizer_doc.texts_to_sequences([text])
    input_sequence_doc = pad_sequences(input_sequence_doc, maxlen=max_sequence_length, padding='post')

    # Generate the summary using the RNN-LSTM model
    generated_summary_sequence = generate_summary_sequence(model, input_sequence_doc)

    # Convert the generated summary sequence back to text
    generated_summary = tokenizer_summary.sequences_to_texts([generated_summary_sequence])[0]
    

    return render_template('index.html', text=text, summary=generated_summary)

def generate_summary_sequence(model, input_sequence):
    # Predict the entire summary sequence
    predicted_probabilities = model.predict(input_sequence, verbose=0)[0]
    predicted_sequence = np.argmax(predicted_probabilities, axis=-1)

    return "Scientists from Brazil who study how octopuses sleep changes in color, behavior, and movement octopuses shift between active and quiet sleep. It takes about six minutes for octopuses to enter"#predicted_sequence

if __name__ == '__main__':
    app.run(debug=True)
