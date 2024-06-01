from flask import Flask, request, render_template
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def plot_embeddings(embeddings, labels):
    # For simplicity, plot the embeddings directly
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], embeddings[:, 1])

    for i, label in enumerate(labels):
        plt.annotate(label, (embeddings[i, 0], embeddings[i, 1]))

    plt.title('Direct Plot of Sentence Embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    similarity = None
    sentence1 = ''
    sentence2 = ''
    embedding1 = None
    embedding2 = None
    plot_url = None

    if request.method == 'POST':
        sentence1 = request.form['sentence1']
        sentence2 = request.form['sentence2']

        embedding1 = get_embeddings(sentence1).numpy()
        embedding2 = get_embeddings(sentence2).numpy()

        similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

        embeddings = np.array([embedding1, embedding2])
        labels = ['Sentence 1', 'Sentence 2']
        plot_url = plot_embeddings(embeddings, labels)

    return render_template('index.html', similarity=similarity, sentence1=sentence1, sentence2=sentence2,
                           embedding1=embedding1, embedding2=embedding2, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
