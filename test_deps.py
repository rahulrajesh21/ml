
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np
from scipy.spatial.distance import cosine

def get_embedding_sentiment_model(text):
    # What the current code does
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Last hidden state, first token [CLS]
    return outputs.last_hidden_state[0, 0, :].numpy()

def test_embeddings():
    print("Testing embeddings...")
    
    roles = ["Developer", "Designer"]
    sentence = "We need to change the CSS for the button."
    
    print("\n--- Current Model (Sentiment) ---")
    try:
        dev_emb = get_embedding_sentiment_model(roles[0])
        des_emb = get_embedding_sentiment_model(roles[1])
        sent_emb = get_embedding_sentiment_model(sentence)
        
        sim_dev = 1 - cosine(dev_emb, sent_emb)
        sim_des = 1 - cosine(des_emb, sent_emb)
        
        print(f"Similarity 'Developer' vs Sentence: {sim_dev:.4f}")
        print(f"Similarity 'Designer' vs Sentence: {sim_des:.4f}")
        print(f"Diff: {abs(sim_dev - sim_des):.4f}")
    except Exception as e:
        print(f"Error with current model: {e}")

    print("\n--- Checking sentence-transformers ---")
    try:
        from sentence_transformers import SentenceTransformer
        print("sentence-transformers is installed!")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        dev_emb = model.encode(roles[0])
        des_emb = model.encode(roles[1])
        sent_emb = model.encode(sentence)
        
        sim_dev = 1 - cosine(dev_emb, sent_emb)
        sim_des = 1 - cosine(des_emb, sent_emb)
        
        print(f"Similarity 'Developer' vs Sentence: {sim_dev:.4f}")
        print(f"Similarity 'Designer' vs Sentence: {sim_des:.4f}")
        print(f"Diff: {abs(sim_dev - sim_des):.4f}")
        
    except ImportError:
        print("sentence-transformers is NOT installed.")

if __name__ == "__main__":
    test_embeddings()
