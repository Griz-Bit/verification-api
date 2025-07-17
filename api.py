# app libraries
from flask import Flask, request, jsonify
# from flask_restful import Api

# algorithm libraries
from deep_translator import GoogleTranslator
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

app = Flask(__name__)
# api = Api(app)

nltk.data.path.append('./nltk_data')

API_KEY = "Irumlj-B0n6RQCkdF-XI0DGAohzLyMMmbjvQKc9Y"

def verify(ref, new):
    # Round-trip translation
    try:
        translated = GoogleTranslator(source='en', target='fr').translate(new)
        back_translated = GoogleTranslator(source='fr', target='en').translate(translated)
        round_trip = SequenceMatcher(None, new, back_translated).ratio()
    except Exception as e:
        print("Round-trip translation failed:", e)
        round_trip = 0.0

    # BERT-based Semantic Similarity
    try:
        model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        embeddings = model.encode([ref, new], convert_to_tensor=True)
        semantic = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    except Exception as e:
        print("Semantic similarity failed:", e)
        semantic = 0.0

    # BLEU Score
    try:
        smoothie = SmoothingFunction().method4
        ref_tokens = word_tokenize(ref.lower())
        new_tokens = word_tokenize(new.lower())
        if len(new_tokens) < 5 or len(ref_tokens) < 5:
            bleu = sentence_bleu([ref_tokens], new_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        else:
            bleu = sentence_bleu([ref_tokens], new_tokens, smoothing_function=smoothie)
    except Exception as e:
        print("BLEU score failed:", e)
        bleu = 0.0

    # compile into final score
    weights = [0.3, 0.4, 0.3]
    combined_score = (
        weights[0] * round_trip +
        weights[1] * semantic +
        weights[2] * bleu
    )
    
    return round(combined_score, 3)

@app.route('/process', methods=['POST'])
def process_data():
    # authenticate API key
    incoming_key = request.headers.get('x-api-key')
    if incoming_key != API_KEY:
        return jsonify({"result": "401 Unauthorized", "accuracy": "NA"})

    reference = request.json.get("reference")
    new_output = request.json.get("new_output")
    accuracy = verify(reference, new_output)
    if accuracy > 0.7:
        result = "Similar"
    else:
        result = "Different"
    return jsonify({"result": result, "accuracy": accuracy})

if __name__ == '__main__':
    app.run(debug=True)