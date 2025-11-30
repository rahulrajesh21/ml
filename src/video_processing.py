import os
from typing import List, Dict, Tuple, Optional
from moviepy import VideoFileClip, concatenate_videoclips
from src.text_analysis import RoleBasedHighlightScorer

class VideoSummarizer:
    """
    Handles video processing for generating role-based summaries.
    """
    
    def __init__(self, highlight_scorer: RoleBasedHighlightScorer):
        self.scorer = highlight_scorer
        
    def score_segments(self, segments: List[Dict], role: str) -> List[Dict]:
        """
        Score each transcription segment against the role.
        
        Args:
            segments: List of dicts with 'text', 'start', 'end', 'speaker'
            role: Target role
            
        Returns:
            List of segments with an added 'score' field
        """
        scored_segments = []
        for seg in segments:
            # Create a copy to avoid modifying original
            s = seg.copy()
            
            # Use the scorer to get a relevance score
            # We use the raw text for scoring
            score = self.scorer.score_sentence(s['text'], role)
            s['score'] = score
            scored_segments.append(s)
            
        return scored_segments

    def filter_and_smooth(
        self, 
        scored_segments: List[Dict], 
        threshold: float = 0.5, 
        min_gap: float = 2.0,
        padding: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        Filter segments by score and merge adjacent ones.
        
        Args:
            scored_segments: List of segments with scores
            threshold: Minimum score to include
            min_gap: Maximum gap (seconds) between segments to merge them
            padding: Extra time (seconds) to add before/after each clip
            
        Returns:
            List of (start, end) time ranges
        """
        # 1. Filter
        relevant_segments = [s for s in scored_segments if s['score'] >= threshold]
        
        if not relevant_segments:
            return []
            
        # Sort by start time just in case
        relevant_segments.sort(key=lambda x: x['start'])
        
        # 2. Smooth (Merge)
        merged_ranges = []
        
        if not relevant_segments:
            return []
            
        # Initialize with first segment
        current_start = max(0, relevant_segments[0]['start'] - padding)
        current_end = relevant_segments[0]['end'] + padding
        
        for i in range(1, len(relevant_segments)):
            next_seg = relevant_segments[i]
            next_start = max(0, next_seg['start'] - padding)
            next_end = next_seg['end'] + padding
            
            # Check if there is an overlap or small gap
            if next_start <= current_end + min_gap:
                # Merge
                current_end = max(current_end, next_end)
            else:
                # Finalize current range and start new one
                merged_ranges.append((current_start, current_end))
                current_start = next_start
                current_end = next_end
                
        # Append the last range
        merged_ranges.append((current_start, current_end))
        
        return merged_ranges

    def create_summary_video(
        self, 
        video_path: str, 
        time_ranges: List[Tuple[float, float]], 
        output_path: str
    ) -> str:
        """
        Cut and stitch video clips based on time ranges.
        
        Args:
            video_path: Source video path
            time_ranges: List of (start, end) tuples
            output_path: Destination path
            
        Returns:
            Status message
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Source video not found: {video_path}")
            
        if not time_ranges:
            raise ValueError("No time ranges provided for summary.")
            
        try:
            # Load source video
            # Use with block or explicit close to ensure resource cleanup
            video = VideoFileClip(video_path)
            
            clips = []
            for start, end in time_ranges:
                # Ensure we don't go out of bounds
                start = max(0, start)
                end = min(video.duration, end)
                
                if start >= end:
                    continue
                    
                # Create subclip
                clip = video.subclipped(start, end)
                clips.append(clip)
                
            if not clips:
                video.close()
                return "No valid clips generated."
                
            # Concatenate
            final_clip = concatenate_videoclips(clips)
            
            # Write output
            # Use 'libx264' codec for compatibility
            # audio_codec='aac' is standard for MP4
            final_clip.write_videofile(
                output_path, 
                codec='libx264', 
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                logger=None # Reduce noise
            )
            
            # Cleanup
            video.close()
            final_clip.close()
            for clip in clips:
                clip.close()
                
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Error processing video: {str(e)}")
