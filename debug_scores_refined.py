
import sys
import os
import numpy as np
import torch

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from text_analysis import TextAnalyzer, RoleBasedHighlightScorer

def debug_text():
    print("Initializing...")
    # Explicitly use CPU to avoid potential MPS issues in this test script
    analyzer = TextAnalyzer(device="cpu")
    scorer = RoleBasedHighlightScorer(analyzer)

    # Text from user (combined)
    text_lines = [
        "It’s a lot for them to manage right now, so I can look into resources, workflow templates, or automation options that could lighten their load and support their team better",
        "I spoke with their operations team today — they’re under a lot of stress because of internal restructuring, and they don’t have enough capacity right now",
        "It might help if we push a quick optimization patch and share a short announcement explaining what improvements are coming",
        "Thanks for joining our weekly product success meeting",
        "We could also add a banner with best-practice tips to help users avoid common performance problems",
        "There’s one more thing I wanted to discuss",
        "Let’s get started"
    ]

    roles = ["Developer", "Product Manager", "Designer", "QA Engineer", "Scrum Master"]

    print("\n--- Score Analysis (Refined) ---")
    
    for sentence in text_lines:
        print(f"\nSentence: '{sentence[:50]}...'")
        
        # Check generic score first
        generic_emb = scorer._get_generic_embedding()
        sent_emb = analyzer.get_embedding(sentence)
        if generic_emb is not None and sent_emb is not None:
             # Manual cosine similarity
            generic_sim = np.dot(sent_emb, generic_emb) / (np.linalg.norm(sent_emb) * np.linalg.norm(generic_emb))
            print(f"  [Generic Score]: {generic_sim:.4f}")
        
        scores = []
        for role in roles:
            score = scorer.score_sentence(sentence, role)
            scores.append((role, score))
            
        # Sort scores
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for role, score in scores:
            print(f"  {role}: {score:.4f}")

if __name__ == "__main__":
    debug_text()
