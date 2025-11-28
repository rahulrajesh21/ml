
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from text_analysis import TextAnalyzer, RoleBasedHighlightScorer

def test_highlights():
    print("Initializing TextAnalyzer...")
    analyzer = TextAnalyzer()
    scorer = RoleBasedHighlightScorer(analyzer)

    # Sample text with mixed content
    text = """
    The database schema needs to be updated to support the new user fields.
    We should change the button color to blue to match the brand guidelines.
    The sprint planning meeting is scheduled for tomorrow at 10 AM.
    I found a bug in the login flow where the error message isn't displaying.
    We need to prioritize the mobile responsiveness for the next release.
    """

    roles = ["Developer", "Designer", "Product Manager"]

    print("\n--- Testing Highlights ---")
    for role in roles:
        print(f"\nRole: {role}")
        highlights = scorer.extract_highlights(text, role, top_n=1)
        for h in highlights:
            print(f"- {h}")

    print("\n--- Debugging Embeddings ---")
    # Check if embeddings are distinct
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    for role in roles:
        role_emb = scorer._get_role_embedding(role)
        print(f"\nRole Embedding ({role}): {role_emb[:5]}... (First 5 dims)")
        
    for sentence in sentences[:3]:
        sent_emb = analyzer.get_embedding(sentence)
        print(f"\nSentence Embedding ('{sentence[:20]}...'): {sent_emb[:5]}... (First 5 dims)")
        
        # Calculate similarity with each role manually to verify
        for role in roles:
            role_emb = scorer._get_role_embedding(role)
            similarity = scorer.score_sentence(sentence, role)
            print(f"  Similarity with {role}: {similarity:.4f}")

if __name__ == "__main__":
    test_highlights()
