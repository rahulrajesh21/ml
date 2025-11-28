import logging
from typing import Dict, List, Any, Optional
import torch
from transformers import pipeline

class TextAnalyzer:
    """
    Analyzer for extracting insights from text using BERT-based models.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the text analyzer.
        
        Args:
            device: Device to use ("cpu" or "cuda" or "mps")
        """
        self.device = -1  # CPU by default
        if device == "cuda" and torch.cuda.is_available():
            self.device = 0
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = 0  # MPS is often mapped to 0 in HF pipelines or handled automatically
            
        print(f"Initializing TextAnalyzer on device: {device}")
        
        try:
            # Load sentiment analysis pipeline
            # using distilbert-base-uncased-finetuned-sst-2-english (fast and good for sentiment)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device
            )
            self.is_ready = True
        except Exception as e:
            print(f"Error initializing TextAnalyzer: {e}")
            self.is_ready = False

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the given text.
        Handles long text by splitting into chunks and aggregating.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with 'label' (POSITIVE/NEGATIVE) and 'score' (confidence)
        """
        if not self.is_ready or not text or not text.strip():
            return {"label": "UNKNOWN", "score": 0.0}
            
        # BERT has a token limit (usually 512). We need to chunk the text.
        # Simple chunking by words for now (approx 300 words ~ 400-500 tokens)
        words = text.split()
        chunk_size = 300
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        
        if not chunks:
            return {"label": "NEUTRAL", "score": 0.5}
            
        results = []
        try:
            # Process chunks
            # We can pass the list of chunks directly to the pipeline
            batch_results = self.sentiment_pipeline(chunks, truncation=True, max_length=512)
            
            # Aggregate results
            # Simple aggregation: majority vote or average score
            positive_score = 0
            negative_score = 0
            count = 0
            
            for res in batch_results:
                score = res['score']
                label = res['label']
                
                if label == 'POSITIVE':
                    positive_score += score
                else:
                    negative_score += score
                count += 1
            
            # Normalize
            if count > 0:
                avg_pos = positive_score / count
                avg_neg = negative_score / count
                
                if avg_pos > avg_neg:
                    return {"label": "POSITIVE", "score": avg_pos}
                else:
                    return {"label": "NEGATIVE", "score": avg_neg}
            
            return {"label": "NEUTRAL", "score": 0.5}
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {"label": "ERROR", "score": 0.0}
