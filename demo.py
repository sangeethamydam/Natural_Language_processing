from flask import Flask, render_template, request
from transformers import pipeline
from rouge import Rouge  # Import the Rouge library

app = Flask(__name__)

# Load the summarization pipeline
summarizer = pipeline("summarization")

# Initialize Rouge
rouge = Rouge()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_headline', methods=['POST'])
def generate_headline():
    text = request.form['text']

    # Generate abstractive summary
    summary = summarizer(text, max_length=50, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Print the generated summary
    generated_summary = summary[0]['summary_text']
    print("Generated Summary:", generated_summary)

    # Example reference summary (replace with your actual reference summary)
    reference_summary = "This is the reference summary."

    # Compute ROUGE scores
    rouge_scores = rouge.get_scores(generated_summary, reference_summary)

    # Print ROUGE scores
    print("ROUGE Scores:", rouge_scores)

    return render_template('index.html', text=text, headline=generated_summary, rouge_scores=rouge_scores)

if __name__ == '__main__':
    app.run(debug=True)
