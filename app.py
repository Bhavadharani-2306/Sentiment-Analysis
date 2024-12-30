from flask import Flask, render_template, request
from textblob import TextBlob

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    text = request.form.get('text')
    sentiment = ""
    sentiment_class = ""

    if text:
        # Create a TextBlob object to calculate sentiment polarity
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity

        # Check sentiment based on polarity score
        if sentiment_score > 0.1:  # Adjusted threshold for positive
            sentiment = "Positive"
            sentiment_class = "positive"  # Green color
        else:  # Adjusted threshold for negative
            sentiment = "Negative"
            sentiment_class = "negative"  # Red color

    return render_template('index.html', sentiment=f'Sentiment: {sentiment}', sentiment_class=sentiment_class)

if __name__ == '__main__':
    app.run(debug=True)
