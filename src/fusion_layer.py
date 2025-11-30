"""
Fusion Layer for Multi-Modal Meeting Analysis.

Combines three modalities to compute hyper-relevant importance scores:
1. Semantic (Text): BERT sentence embeddings capturing meaning
2. Tonal (Audio): MFCC/prosodic features capturing urgency/emphasis
3. Role: Static role embeddings for role-specific relevance

The fusion produces a single relevance score per segment that reflects
both WHAT was said (semantic), HOW it was said (tonal), and WHO it matters to (role).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SegmentFeatures:
    """Container for multi-modal features of a transcript segment."""
    start_time: float
    end_time: float
    text: str
    speaker: Optional[str] = None
    
    # Embeddings (populated by analyzers)
    text_embedding: Optional[np.ndarray] = None  # 384-dim from MiniLM
    mfcc_embedding: Optional[np.ndarray] = None  # 52-dim from AudioTonalAnalyzer
    prosodic_features: Optional[Dict[str, float]] = None  # urgency, emphasis, etc.
    
    # Computed scores
    semantic_score: float = 0.0
    tonal_score: float = 0.0
    role_relevance: float = 0.0
    fused_score: float = 0.0


class FusionLayer:
    """
    Multi-modal fusion layer combining text, audio, and role signals.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                      FUSION LAYER                           │
    │                                                             │
    │  ┌──────────┐   ┌──────────┐   ┌──────────┐                │
    │  │  Text    │   │  Audio   │   │  Role    │                │
    │  │ Encoder  │   │ Encoder  │   │ Encoder  │                │
    │  │ (384-d)  │   │ (52-d)   │   │ (384-d)  │                │
    │  └────┬─────┘   └────┬─────┘   └────┬─────┘                │
    │       │              │              │                       │
    │       ▼              ▼              ▼                       │
    │  ┌─────────┐   ┌──────────┐   ┌──────────┐                 │
    │  │Semantic │   │  Tonal   │   │  Role    │                 │
    │  │ Score   │   │  Score   │   │Relevance │                 │
    │  │ (0-1)   │   │  (0-1)   │   │  (0-1)   │                 │
    │  └────┬────┘   └────┬─────┘   └────┬─────┘                 │
    │       │             │              │                        │
    │       └─────────────┼──────────────┘                        │
    │                     ▼                                       │
    │              ┌────────────┐                                 │
    │              │  Weighted  │                                 │
    │              │   Fusion   │                                 │
    │              │  Function  │                                 │
    │              └─────┬──────┘                                 │
    │                    ▼                                        │
    │              ┌────────────┐                                 │
    │              │   Fused    │                                 │
    │              │   Score    │                                 │
    │              │   (0-1)    │                                 │
    │              └────────────┘                                 │
    └─────────────────────────────────────────────────────────────┘
    
    Fusion Strategies:
    1. Weighted Sum: α*semantic + β*tonal + γ*role
    2. Multiplicative: semantic * (1 + tonal_boost) * role_relevance
    3. Attention-based: Learn weights from cross-modal attention
    """
    
    def __init__(
        self,
        text_analyzer=None,
        audio_analyzer=None,
        fusion_strategy: str = "weighted",
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the fusion layer.
        
        Args:
            text_analyzer: TextAnalyzer instance for semantic embeddings
            audio_analyzer: AudioTonalAnalyzer instance for prosodic features
            fusion_strategy: "weighted", "multiplicative", or "gated"
            weights: Custom weights for fusion (default: balanced)
        """
        self.text_analyzer = text_analyzer
        self.audio_analyzer = audio_analyzer
        self.fusion_strategy = fusion_strategy
        
        # Default weights (can be tuned)
        self.weights = weights or {
            'semantic': 0.5,   # What was said
            'tonal': 0.2,     # How it was said
            'role': 0.3       # Who it matters to
        }
        
        # Importance description embedding (cached)
        self._importance_embedding = None
        
        # Role embeddings cache
        self.role_embeddings: Dict[str, np.ndarray] = {}
        
        logger.info(f"FusionLayer initialized with strategy: {fusion_strategy}")
    
    def set_role_embeddings(self, role_embeddings: Dict[str, np.ndarray]):
        """Set pre-computed role embeddings."""
        self.role_embeddings = role_embeddings
        logger.info(f"Loaded {len(role_embeddings)} role embeddings")
    
    def _get_importance_embedding(self) -> Optional[np.ndarray]:
        """Get cached embedding for 'important content' description."""
        if self._importance_embedding is not None:
            return self._importance_embedding
        
        if not self.text_analyzer:
            return None
        
        importance_desc = """
        Important business decisions, strategic proposals, key insights,
        data-driven observations, problem identification, solution suggestions,
        action items, concerns raised, metrics discussion, process improvements,
        technical explanations, resource allocation, timeline commitments,
        risk assessment, and substantive professional contributions.
        """
        
        self._importance_embedding = self.text_analyzer.get_embedding(importance_desc)
        return self._importance_embedding
    
    def compute_semantic_score(
        self,
        text: str,
        text_embedding: Optional[np.ndarray] = None
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Compute semantic importance score for text.
        
        Args:
            text: The text content
            text_embedding: Pre-computed embedding (optional)
            
        Returns:
            Tuple of (score, embedding)
        """
        if not self.text_analyzer:
            return 0.0, None
        
        # Get text embedding
        if text_embedding is None:
            text_embedding = self.text_analyzer.get_embedding(text)
        
        if text_embedding is None:
            return 0.0, None
        
        # Compare to importance description
        importance_emb = self._get_importance_embedding()
        if importance_emb is None:
            return 0.0, text_embedding
        
        # Cosine similarity
        similarity = self._cosine_similarity(text_embedding, importance_emb)
        
        # Normalize to 0-1 (similarity can be negative)
        score = max(0.0, (similarity + 1) / 2)
        
        return float(score), text_embedding
    
    def compute_tonal_score(
        self,
        prosodic_features: Optional[Dict[str, float]] = None,
        mfcc_embedding: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute tonal importance score from prosodic features.
        
        High urgency + high emphasis = high tonal score
        
        Args:
            prosodic_features: Dict with urgency_score, emphasis_score, etc.
            mfcc_embedding: MFCC embedding (for future use)
            
        Returns:
            Tonal score (0-1)
        """
        if prosodic_features is None:
            return 0.0
        
        urgency = prosodic_features.get('urgency_score', 0.0)
        emphasis = prosodic_features.get('emphasis_score', 0.0)
        energy_std = prosodic_features.get('energy_std', 0.0)
        
        # Combine signals
        # High variation in energy suggests dynamic speech (more engaging)
        energy_factor = min(energy_std * 20, 1.0)  # Normalize
        
        # Weighted combination
        tonal_score = (
            0.4 * urgency +
            0.4 * emphasis +
            0.2 * energy_factor
        )
        
        return float(min(max(tonal_score, 0.0), 1.0))
    
    def compute_role_relevance(
        self,
        text_embedding: np.ndarray,
        role: str
    ) -> float:
        """
        Compute relevance of content to a specific role.
        
        Args:
            text_embedding: Text embedding vector
            role: Role name (e.g., "Product Manager")
            
        Returns:
            Role relevance score (0-1)
        """
        if text_embedding is None:
            return 0.0
        
        # Get role embedding
        role_emb = self.role_embeddings.get(role)
        
        if role_emb is None and self.text_analyzer:
            # Generate on-the-fly
            role_emb = self.text_analyzer.get_embedding(role)
            if role_emb is not None:
                self.role_embeddings[role] = role_emb
        
        if role_emb is None:
            return 0.5  # Neutral if no role embedding
        
        # Cosine similarity
        similarity = self._cosine_similarity(text_embedding, role_emb)
        
        # Normalize to 0-1
        score = max(0.0, (similarity + 1) / 2)
        
        return float(score)
    
    def fuse_scores(
        self,
        semantic_score: float,
        tonal_score: float,
        role_relevance: float
    ) -> float:
        """
        Fuse multi-modal scores into a single relevance score.
        
        Args:
            semantic_score: Semantic importance (0-1)
            tonal_score: Tonal/prosodic importance (0-1)
            role_relevance: Role-specific relevance (0-1)
            
        Returns:
            Fused score (0-1)
        """
        if self.fusion_strategy == "weighted":
            # Simple weighted sum
            fused = (
                self.weights['semantic'] * semantic_score +
                self.weights['tonal'] * tonal_score +
                self.weights['role'] * role_relevance
            )
            
        elif self.fusion_strategy == "multiplicative":
            # Multiplicative with tonal boost
            # Base score from semantic, boosted by tonal, filtered by role
            tonal_boost = 1.0 + (tonal_score * 0.5)  # 1.0 to 1.5x
            fused = semantic_score * tonal_boost * (0.5 + role_relevance * 0.5)
            
        elif self.fusion_strategy == "gated":
            # Gated fusion: role acts as a gate
            # Only high role-relevance content gets through
            gate = self._sigmoid((role_relevance - 0.5) * 4)  # Sharp gate around 0.5
            content_score = 0.7 * semantic_score + 0.3 * tonal_score
            fused = content_score * gate
            
        else:
            # Default to weighted
            fused = (semantic_score + tonal_score + role_relevance) / 3
        
        return float(min(max(fused, 0.0), 1.0))
    
    def score_segment(
        self,
        segment: SegmentFeatures,
        role: str,
        audio_data: Optional[np.ndarray] = None,
        sample_rate: int = 16000
    ) -> SegmentFeatures:
        """
        Score a single segment using all modalities.
        
        Args:
            segment: SegmentFeatures with text and timing
            role: Target role for relevance scoring
            audio_data: Full audio array (for extracting segment audio)
            sample_rate: Audio sample rate
            
        Returns:
            SegmentFeatures with all scores populated
        """
        # 1. Semantic score
        semantic_score, text_emb = self.compute_semantic_score(
            segment.text,
            segment.text_embedding
        )
        segment.semantic_score = semantic_score
        segment.text_embedding = text_emb
        
        # 2. Tonal score (if audio available)
        if audio_data is not None and self.audio_analyzer:
            # Extract segment audio
            start_sample = int(segment.start_time * sample_rate)
            end_sample = int(segment.end_time * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            if len(segment_audio) > 0:
                segment.prosodic_features = self.audio_analyzer.extract_prosodic_features(
                    segment_audio
                )
                segment.mfcc_embedding = self.audio_analyzer.get_mfcc_embedding(
                    segment_audio
                )
        
        tonal_score = self.compute_tonal_score(
            segment.prosodic_features,
            segment.mfcc_embedding
        )
        segment.tonal_score = tonal_score
        
        # 3. Role relevance
        if text_emb is not None:
            role_relevance = self.compute_role_relevance(text_emb, role)
        else:
            role_relevance = 0.5
        segment.role_relevance = role_relevance
        
        # 4. Fuse scores
        segment.fused_score = self.fuse_scores(
            semantic_score,
            tonal_score,
            role_relevance
        )
        
        return segment
    
    def score_segments(
        self,
        segments: List[Dict],
        role: str,
        audio_data: Optional[np.ndarray] = None,
        sample_rate: int = 16000
    ) -> List[SegmentFeatures]:
        """
        Score multiple segments.
        
        Args:
            segments: List of dicts with 'text', 'start', 'end', 'speaker'
            role: Target role
            audio_data: Full audio array
            sample_rate: Audio sample rate
            
        Returns:
            List of scored SegmentFeatures
        """
        scored = []
        
        for seg in segments:
            features = SegmentFeatures(
                start_time=seg.get('start', 0),
                end_time=seg.get('end', 0),
                text=seg.get('text', ''),
                speaker=seg.get('speaker')
            )
            
            scored_segment = self.score_segment(
                features,
                role,
                audio_data,
                sample_rate
            )
            scored.append(scored_segment)
        
        return scored
    
    def get_top_segments(
        self,
        scored_segments: List[SegmentFeatures],
        top_n: int = 5,
        min_score: float = 0.3
    ) -> List[SegmentFeatures]:
        """
        Get top N segments by fused score.
        
        Args:
            scored_segments: List of scored segments
            top_n: Number of top segments to return
            min_score: Minimum fused score threshold
            
        Returns:
            Top segments sorted by fused score
        """
        # Filter by minimum score
        filtered = [s for s in scored_segments if s.fused_score >= min_score]
        
        # Sort by fused score descending
        sorted_segments = sorted(filtered, key=lambda x: x.fused_score, reverse=True)
        
        return sorted_segments[:top_n]
    
    def explain_score(self, segment: SegmentFeatures) -> str:
        """
        Generate human-readable explanation of a segment's score.
        
        Args:
            segment: Scored segment
            
        Returns:
            Explanation string
        """
        explanation = f"**Fused Score: {segment.fused_score:.2f}**\n"
        explanation += f"- Semantic (what): {segment.semantic_score:.2f}\n"
        explanation += f"- Tonal (how): {segment.tonal_score:.2f}\n"
        explanation += f"- Role (who): {segment.role_relevance:.2f}\n"
        
        if segment.prosodic_features:
            pf = segment.prosodic_features
            explanation += f"\nProsodic Details:\n"
            explanation += f"- Urgency: {pf.get('urgency_score', 0):.2f}\n"
            explanation += f"- Emphasis: {pf.get('emphasis_score', 0):.2f}\n"
        
        return explanation
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
