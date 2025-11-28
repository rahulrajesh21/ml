import logging
from typing import Dict, List, Any, Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cosine

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
            
            # Load sentence embedding model (all-MiniLM-L6-v2)
            # We load this manually to have full control over pooling and normalization
            print("Loading sentence embedding model...")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
            # Move model to device
            if self.device != -1:
                # If device is an integer (GPU index), we need to handle it
                # For simplicity in this manual loading, let's stick to CPU or handle CUDA/MPS if needed
                # But since we are using 'device' which might be an int from pipeline logic, let's be careful.
                # The pipeline uses device IDs, but .to() expects device objects or strings.
                
                if torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.to("cuda")
                elif torch.backends.mps.is_available():
                    self.embedding_model = self.embedding_model.to("mps")
            
            self.is_ready = True
        except Exception as e:
            print(f"Error initializing TextAnalyzer: {e}")
            self.is_ready = False

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get the dense numerical vector embedding for any text.
        Uses the internal DistilBERT model from the sentiment pipeline.
        
        Args:
            text: The text to encode
            
        Returns:
            Numpy array representing the embedding vector (768 dimensions), or None if error.
        """
        if not self.is_ready:
            return None
            
        try:
            # Use the dedicated embedding model
            tokenizer = self.embedding_tokenizer
            model = self.embedding_model
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move inputs to device
            if next(model.parameters()).device.type != 'cpu':
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get model output
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Mean Pooling - Take attention mask into account for correct averaging
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
            # Normalize embeddings
            normalized_embeddings = F.normalize(mean_pooled, p=2, dim=1)
            
            # Convert to numpy
            return normalized_embeddings[0].cpu().numpy()
            
        except Exception as e:
            print(f"Error extracting embedding for '{text[:20]}...': {e}")
            return None

    # Alias for backward compatibility if needed, but we should update calls
    def get_role_embedding(self, role_name: str) -> List[float]:
        emb = self.get_embedding(role_name)
        if emb is not None:
            return emb.tolist()
        return []

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

class RoleBasedHighlightScorer:
    """
    Scores sentences based on their semantic relevance to specific roles using BERT embeddings.
    """
    
    def __init__(self, text_analyzer: Optional[TextAnalyzer] = None):
        self.text_analyzer = text_analyzer
        
        # Define semantic descriptions for each role
        self.role_descriptions = {
            "Developer": "Technical details about code, APIs, databases, bugs, latency, servers, deployment, git, refactoring, and software architecture.",
            "Product Manager": "Product strategy, user requirements, roadmap, KPIs, launch dates, customer feedback, market analysis, and feature prioritization.",
            "Designer": "User interface visuals, color palettes, typography, layout, user experience flows, prototypes, wireframes, and design consistency.",
            "QA Engineer": "Testing procedures, bug reproduction, test cases, automation scripts, regression testing, and quality verification.",
            "Scrum Master": "Agile ceremonies, sprint velocity, blockers, standups, retrospectives, team capacity, and process improvements."
        }
        
        # Generic description for filtering
        self.generic_description = "General meeting pleasantries, scheduling, administrative tasks, greetings, farewells, small talk, and vague project management statements like 'let's get started' or 'thanks for joining'."
        
        # Cache for role embeddings
        self.role_embeddings = {}
        
    def _get_role_embedding(self, role: str) -> Optional[np.ndarray]:
        """Get or compute embedding for a role description."""
        if role in self.role_embeddings:
            return self.role_embeddings[role]
            
        if not self.text_analyzer:
            return None
            
        description = self.role_descriptions.get(role, "")
        if not description:
            return None
            
        embedding = self.text_analyzer.get_embedding(description)
        if embedding is not None:
            self.role_embeddings[role] = embedding
            
        return embedding

    def _get_generic_embedding(self) -> Optional[np.ndarray]:
        """Get or compute embedding for the generic description."""
        if "Generic" in self.role_embeddings:
            return self.role_embeddings["Generic"]
            
        if not self.text_analyzer:
            return None
            
        embedding = self.text_analyzer.get_embedding(self.generic_description)
        if embedding is not None:
            self.role_embeddings["Generic"] = embedding
            
        return embedding
        
    def score_sentence(self, sentence: str, role: str) -> float:
        """
        Score a sentence based on Cosine Similarity between the sentence embedding
        and the role description embedding.
        """
        if not sentence or not role or not self.text_analyzer:
            return 0.0
            
        # Get embeddings
        role_emb = self._get_role_embedding(role)
        sent_emb = self.text_analyzer.get_embedding(sentence)
        
        if role_emb is None or sent_emb is None:
            return 0.0
            
        # Compute Cosine Similarity
        # 1 - cosine_distance = cosine_similarity
        # Range: [-1, 1], where 1 is identical
        role_similarity = 1 - cosine(role_emb, sent_emb)
        
        # Check against Generic role
        generic_emb = self._get_generic_embedding()
        if generic_emb is not None:
            generic_similarity = 1 - cosine(generic_emb, sent_emb)
            
            # If the sentence is more generic than specific to the role, penalize heavily
            if generic_similarity > role_similarity:
                return 0.0
            
            # Or if it's just very generic (high generic score), penalize
            if generic_similarity > 0.4:
                role_similarity -= (generic_similarity * 0.5)
        
        return float(role_similarity)
        
    def extract_highlights(self, text: str, role: str, top_n: int = 3) -> List[str]:
        """
        Extract the top N most relevant sentences for a given role.
        """
        if not text or not self.text_analyzer:
            return []
            
        # Split text into sentences
        sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        
        scored_sentences = []
        for sentence in sentences:
            score = self.score_sentence(sentence, role)
            # Filter out low relevance (increased threshold to 0.25)
            if score > 0.25:
                scored_sentences.append((score, sentence))
                
        # Sort by score descending
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N sentences
        return [s[1] for s in scored_sentences[:top_n]]

