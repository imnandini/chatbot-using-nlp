from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pickle
import random
import nltk
import json
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS

lemmatizer = WordNetLemmatizer()
model = tf.keras.models.load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
# Load intents from JSON file
intents = json.loads(open('intents.json').read())
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']

    ints = predict_class(message)
    intent = ints[0]['intent']

    for i in intents['intents']:
        if i['tag'] == intent:
            response = random.choice(i['responses'])
            break

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
