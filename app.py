from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import re
import string
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing import sequence , text

model = load_model('model.h5')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/about')
def about():
	return render_template('about.html')


@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    
    if request.method == 'GET':
        return render_template('prediction.html')
    
    if request.method == 'POST':
        title = request.form['title']
        text1 = request.form['text'] 
        
        raw_text = text1 + ' ' + title

    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)

    
    #Removing the square brackets
    def remove_between_square_brackets2(text):
      return re.sub('\[[^]]*\]', '', text)

    # Removing URL's
    def remove_between_square_brackets1(text):
      return re.sub(r'http\S+', '', text)

    #Removing the stopwords from text
    def remove_stopwords(text):
      final_text = []
      for i in text.split():
        if i.strip().lower() not in stop:
          final_text.append(i.strip())
          return " ".join(final_text)

    #Removing the noisy text
    def denoise_text(text):
      text = remove_between_square_brackets1(text)
      text = remove_between_square_brackets2(text)
      text = remove_stopwords(text)
      return text

    clean_text = denoise_text(raw_text)

    max_features = 10000
    maxlen = 300

    tokenizer = text.Tokenizer(num_words=max_features )
    
    tokenizer.fit_on_texts(clean_text)

    tokenized_train = tokenizer.texts_to_sequences(clean_text)

    X = sequence.pad_sequences(tokenized_train, maxlen=maxlen)

    prediction = model.predict(X[0])
    
    if prediction[-1][0] > 0.5:
        prediction = 'Real News'
    else:
        prediction = 'Fake News'
    
    return render_template('result.html',my_pred = prediction , raw = text1 )


if __name__ == '__main__':
    app.run()