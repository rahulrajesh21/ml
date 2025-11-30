import logging
from typing import Dict, List, Any, Optional, Tuple
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
        
        # Description for important/substantive meeting content
        # This captures decisions, insights, proposals, concerns - regardless of role
        self.importance_description = """
        Important business decisions, strategic proposals, key insights, data-driven observations,
        problem identification, solution suggestions, action items, concerns raised, 
        metrics discussion, process improvements, technical explanations, resource allocation,
        timeline commitments, risk assessment, and substantive professional contributions.
        """
        
        # Generic description for filtering out non-substantive content
        self.generic_description = """
        General meeting pleasantries, scheduling, administrative tasks, greetings, farewells, 
        small talk, thank you messages, expressions of gratitude like 'Thanks Eric' or 'Thank you Steve',
        acknowledgments, transitions like 'let's get started', 'thanks for joining', 'moving on',
        short affirmations like 'yeah', 'okay', 'right', 'got it', 'sounds good',
        and vague statements without specific content.
        """
        
        # Cache for embeddings
        self.role_embeddings = {}
        self._importance_embedding = None
        
    def _get_importance_embedding(self) -> Optional[np.ndarray]:
        """Get or compute embedding for important/substantive content."""
        if self._importance_embedding is not None:
            return self._importance_embedding
            
        if not self.text_analyzer:
            return None
            
        embedding = self.text_analyzer.get_embedding(self.importance_description)
        if embedding is not None:
            self._importance_embedding = embedding
            
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
        
    def score_sentence(self, sentence: str, role: str = None) -> float:
        """
        Score a sentence based on its importance/substantiveness.
        Compares sentence embedding against 'important content' description
        and penalizes generic/filler content.
        
        Args:
            sentence: The sentence to score
            role: (unused, kept for API compatibility)
        """
        if not sentence or not self.text_analyzer:
            return 0.0
        
        # Skip very short sentences (likely filler)
        if len(sentence.split()) < 4:
            return 0.0
            
        # Get embeddings
        importance_emb = self._get_importance_embedding()
        sent_emb = self.text_analyzer.get_embedding(sentence)
        
        if importance_emb is None or sent_emb is None:
            return 0.0
            
        # Compute similarity to "important content"
        importance_similarity = 1 - cosine(importance_emb, sent_emb)
        
        # Check against Generic/filler content
        generic_emb = self._get_generic_embedding()
        if generic_emb is not None:
            generic_similarity = 1 - cosine(generic_emb, sent_emb)
            
            # If the sentence is more generic than important, return 0
            if generic_similarity > importance_similarity:
                return 0.0
            
            # Penalize sentences that are somewhat generic
            if generic_similarity > 0.35:
                importance_similarity -= (generic_similarity * 0.5)
        
        return max(0.0, float(importance_similarity))
        
    def _normalize_speaker(self, speaker: str) -> str:
        """Normalize speaker name for comparison."""
        # Remove extra whitespace and convert to lowercase
        normalized = speaker.lower().strip()
        # Remove common punctuation that might differ
        normalized = normalized.replace("'", "").replace('"', "").replace("-", " ")
        return normalized
    
    def _speakers_match(self, role: str, speaker: str) -> bool:
        """Check if a role matches a speaker tag."""
        role_norm = self._normalize_speaker(role)
        speaker_norm = self._normalize_speaker(speaker)
        
        # Exact match
        if role_norm == speaker_norm:
            return True
        
        # Role contains speaker or vice versa (handles partial matches)
        if role_norm in speaker_norm or speaker_norm in role_norm:
            return True
        
        # Check if the name part matches (before parentheses)
        # "Eric Johnson (CTO)" should match "Eric Johnson (CTO - Host)"
        import re
        role_name = re.split(r'\s*\(', role_norm)[0].strip()
        speaker_name = re.split(r'\s*\(', speaker_norm)[0].strip()
        
        if role_name and speaker_name and (role_name == speaker_name):
            return True
        
        return False

    def extract_highlights(self, text: str, role: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Extract the top N most relevant sentences for a given role.
        Strictly filters for sentences spoken by the role if speaker tags are present.
        Returns a list of (sentence, score) tuples.
        """
        if not text or not self.text_analyzer:
            return []
            
        scored_sentences = []
        
        # Split into lines to preserve speaker context
        lines = text.split('\n')
        
        import re
        # Regex to extract speaker: [Timestamp] [Speaker] Text
        # Matches: [10:00:00] [Developer] Some text.
        line_pattern = re.compile(r"\[.*?\] \[(.*?)\] (.*)")
        
        role_sentences = []
        all_sentences = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = line_pattern.match(line)
            current_speaker = None
            content = line
            
            if match:
                current_speaker = match.group(1)
                content = match.group(2)
            
            # Split content into sentences
            sentences = [s.strip() for s in content.replace("?", ".").replace("!", ".").split(".") if s.strip()]
            
            for sentence in sentences:
                all_sentences.append(sentence)
                # Check if this sentence was spoken by the target role
                if current_speaker and self._speakers_match(role, current_speaker):
                    role_sentences.append(sentence)
        
        # DECISION: If we found sentences spoken by the role, ONLY analyze those.
        # This creates a strict summary of what THEY said.
        # If not (maybe role name mismatch), fall back to analyzing everything.
        target_pool = role_sentences if role_sentences else all_sentences
        
        for sentence in target_pool:
            # Score based on importance/substantiveness
            score = self.score_sentence(sentence)
            
            # Filter out low importance (threshold 0.20)
            if score > 0.20:
                scored_sentences.append((score, sentence))
                
        # Sort by score descending
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N sentences
        # Remove duplicates while preserving order
        seen = set()
        unique_highlights = []
        for s in scored_sentences:
            if s[1] not in seen:
                unique_highlights.append((s[1], round(s[0], 2)))
                seen.add(s[1])
                if len(unique_highlights) >= top_n:
                    break
                    
        return unique_highlights

