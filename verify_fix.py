
import sys
import os
import numpy as np
import torch

# Force CPU for stability during test
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from text_analysis import TextAnalyzer, RoleBasedHighlightScorer

def test_highlights():
    print("Initializing TextAnalyzer...")
    # Initialize with explicit CPU device to avoid any ambiguity
    analyzer = TextAnalyzer(device="cpu")
    scorer = RoleBasedHighlightScorer(analyzer)

    # Sample text with mixed content
    text = """
    The database schema needs to be updated to support the new user fields.
    We should change the button color to blue to match the brand guidelines.
    """

    roles = ["Developer", "Designer"]

    print("\n--- Testing Highlights ---")
    for role in roles:
        print(f"\nRole: {role}")
        highlights = scorer.extract_highlights(text, role, top_n=1)
        for h in highlights:
            print(f"- {h}")

    print("\n--- Verifying Embeddings ---")
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Calculate similarity matrix
    for sentence in sentences:
        print(f"\nSentence: '{sentence[:40]}...'")
        sent_emb = analyzer.get_embedding(sentence)
        
        scores = {}
        for role in roles:
            role_emb = scorer._get_role_embedding(role)
            # Manual cosine similarity
            similarity = np.dot(sent_emb, role_emb) / (np.linalg.norm(sent_emb) * np.linalg.norm(role_emb))
            scores[role] = similarity
            print(f"  Similarity with {role}: {similarity:.4f}")
            
        # Check if the correct role has higher score
        if "database" in sentence.lower():
            if scores["Developer"] > scores["Designer"]:
                print("  ✅ CORRECT: Developer > Designer")
            else:
                print("  ❌ INCORRECT: Developer <= Designer")
        elif "color" in sentence.lower():
            if scores["Designer"] > scores["Developer"]:
                print("  ✅ CORRECT: Designer > Developer")
            else:
                print("  ❌ INCORRECT: Designer <= Developer")

if __name__ == "__main__":
    test_highlights()
