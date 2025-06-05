from flask import Flask, request, render_template
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os

app = Flask(__name__)

model_path = './my_distilbert_model'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

classes = ["sadness", "joy", "love", "anger", "fear", "surprise"]
emoji_map = {
    "joy": "😄",
    "sadness": "😢",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲"
}

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    label = classes[pred]
    return label, emoji_map.get(label, "🤔")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    emoji = None
    if request.method == 'POST':
        text = request.form['text']
        prediction, emoji = predict(text)
    return render_template('index.html', prediction=prediction, emoji=emoji)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)




