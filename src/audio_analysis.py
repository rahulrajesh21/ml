"""
Audio Tonal Analysis module for extracting prosodic features.
Computes MFCCs (Mel-frequency cepstral coefficients) to capture
pitch, energy, and emphasis patterns that indicate urgency or importance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import librosa for audio feature extraction
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available. Install with: pip install librosa")


class AudioTonalAnalyzer:
    """
    Analyzer for extracting prosodic/tonal features from audio.
    Uses MFCCs to capture pitch, energy, and emphasis patterns.
    """
    
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13):
        """
        Initialize the audio tonal analyzer.
        
        Args:
            sample_rate: Audio sample rate (default 16000 for Whisper compatibility)
            n_mfcc: Number of MFCC coefficients to extract (default 13)
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "librosa is required for audio tonal analysis.\n"
                "Install with: pip install librosa"
            )
        
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.is_ready = True
        logger.info(f"AudioTonalAnalyzer initialized (sr={sample_rate}, n_mfcc={n_mfcc})")
    
    def extract_mfcc(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract MFCC features from audio data.
        
        Args:
            audio_data: Audio samples as numpy array (float32, normalized to [-1, 1])
            
        Returns:
            MFCC matrix of shape (n_mfcc, time_frames) or None if error
        """
        if not self.is_ready:
            return None
            
        try:
            # Ensure float32 and proper range
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / 32768.0
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=512,
                n_fft=2048
            )
            
            return mfccs
            
        except Exception as e:
            logger.error(f"Error extracting MFCCs: {e}")
            return None
    
    def extract_prosodic_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive prosodic features from audio.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            Dictionary with prosodic feature values
        """
        features = {
            'energy_mean': 0.0,
            'energy_std': 0.0,
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'speech_rate': 0.0,
            'urgency_score': 0.0,
            'emphasis_score': 0.0
        }
        
        if not self.is_ready:
            return features
            
        try:
            # Ensure proper format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / 32768.0
            
            # 1. Energy (RMS)
            rms = librosa.feature.rms(y=audio_data, hop_length=512)[0]
            features['energy_mean'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            
            # 2. Pitch (F0) using piptrack
            pitches, magnitudes = librosa.piptrack(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=512
            )
            
            # Get pitch values where magnitude is significant
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # Valid pitch
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
            
            # 3. Speech rate (zero crossing rate as proxy)
            zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=512)[0]
            features['speech_rate'] = float(np.mean(zcr))
            
            # 4. Compute urgency score (high energy + high pitch variation)
            # Normalize components to 0-1 range approximately
            energy_norm = min(features['energy_mean'] * 10, 1.0)
            pitch_var_norm = min(features['pitch_std'] / 100, 1.0)
            
            features['urgency_score'] = float((energy_norm * 0.6) + (pitch_var_norm * 0.4))
            
            # 5. Compute emphasis score (energy peaks)
            if len(rms) > 0:
                energy_peaks = np.sum(rms > np.mean(rms) + np.std(rms)) / len(rms)
                features['emphasis_score'] = float(min(energy_peaks * 2, 1.0))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting prosodic features: {e}")
            return features
    
    def analyze_segment(
        self,
        audio_data: np.ndarray,
        start_time: float,
        end_time: float
    ) -> Dict:
        """
        Analyze a specific audio segment.
        
        Args:
            audio_data: Full audio as numpy array
            start_time: Segment start in seconds
            end_time: Segment end in seconds
            
        Returns:
            Dictionary with segment analysis results
        """
        # Extract segment
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        # Bounds check
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            return {'error': 'Invalid segment bounds'}
        
        segment = audio_data[start_sample:end_sample]
        
        # Get features
        features = self.extract_prosodic_features(segment)
        mfccs = self.extract_mfcc(segment)
        
        return {
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time,
            'prosodic_features': features,
            'mfcc_shape': mfccs.shape if mfccs is not None else None,
            'mfcc_mean': mfccs.mean(axis=1).tolist() if mfccs is not None else None
        }
    
    def get_mfcc_embedding(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Get a fixed-size MFCC embedding for an audio segment.
        Useful for comparing audio segments or as input to ML models.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            1D numpy array of shape (n_mfcc * 4,) containing mean, std, min, max of each MFCC
        """
        mfccs = self.extract_mfcc(audio_data)
        
        if mfccs is None:
            return None
        
        # Compute statistics across time for each coefficient
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        
        # Concatenate into single vector
        embedding = np.concatenate([mfcc_mean, mfcc_std, mfcc_min, mfcc_max])
        
        return embedding
    
    def detect_emphasis_regions(
        self,
        audio_data: np.ndarray,
        threshold: float = 1.5
    ) -> List[Tuple[float, float, float]]:
        """
        Detect regions of emphasis (high energy) in audio.
        
        Args:
            audio_data: Audio samples as numpy array
            threshold: Number of std deviations above mean to consider emphasis
            
        Returns:
            List of (start_time, end_time, intensity) tuples
        """
        if not self.is_ready:
            return []
            
        try:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / 32768.0
            
            # Compute RMS energy
            hop_length = 512
            rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
            
            # Find emphasis threshold
            mean_rms = np.mean(rms)
            std_rms = np.std(rms)
            emphasis_threshold = mean_rms + (threshold * std_rms)
            
            # Find regions above threshold
            emphasis_frames = rms > emphasis_threshold
            
            # Convert to time regions
            regions = []
            in_region = False
            region_start = 0
            
            for i, is_emphasis in enumerate(emphasis_frames):
                time = librosa.frames_to_time(i, sr=self.sample_rate, hop_length=hop_length)
                
                if is_emphasis and not in_region:
                    in_region = True
                    region_start = time
                elif not is_emphasis and in_region:
                    in_region = False
                    intensity = float(np.mean(rms[max(0, i-10):i]))
                    regions.append((region_start, time, intensity))
            
            # Handle case where audio ends during emphasis
            if in_region:
                end_time = librosa.frames_to_time(len(rms), sr=self.sample_rate, hop_length=hop_length)
                intensity = float(np.mean(rms[-10:]))
                regions.append((region_start, end_time, intensity))
            
            return regions
            
        except Exception as e:
            logger.error(f"Error detecting emphasis regions: {e}")
            return []


def load_audio_file(file_path: str, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """
    Utility function to load audio from file.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Audio data as numpy array or None if error
    """
    if not LIBROSA_AVAILABLE:
        logger.error("librosa required to load audio files")
        return None
        
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        return audio
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        return None
