
import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

# Add src to path

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

# Standalone implementation of filter_and_smooth to avoid importing torch/src which causes crashes
def filter_and_smooth(
    scored_segments: List[Dict], 
    threshold: float = 0.5, 
    min_gap: float = 2.0,
    padding: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Filter segments by score and merge adjacent ones.
    (Copied from src/video_processing.py to avoid dependencies)
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

def calculate_overlap(ref_start, ref_end, hyp_start, hyp_end):
    """Calculate temporal overlap between two segments."""
    start = max(ref_start, hyp_start)
    end = min(ref_end, hyp_end)
    return max(0, end - start)

def calculate_der(reference_segments: List[Dict], hypothesis_segments: List[Dict]):
    """
    Calculate a simplified Diarization Error Rate (DER).
    DER = (False Alarm + Missed Detection + Confusion) / Total Speaker Time
    
    This is a simplified implementation that assumes segments are sorted and non-overlapping within the same speaker.
    """
    total_ref_time = sum(s['end'] - s['start'] for s in reference_segments)
    if total_ref_time == 0:
        return 0.0
        
    # 1. Missed Detection (Ref has speech, Hyp has none)
    # 2. False Alarm (Hyp has speech, Ref has none)
    # 3. Speaker Confusion (Ref and Hyp overlap but speaker ID differs)
    
    # For simplicity in this standalone script without pyannote, we'll approximate:
    # We'll discretize time into 100ms frames and compare labels.
    
    max_time = max(
        max(s['end'] for s in reference_segments) if reference_segments else 0,
        max(s['end'] for s in hypothesis_segments) if hypothesis_segments else 0
    )
    
    frame_step = 0.1
    num_frames = int(np.ceil(max_time / frame_step))
    
    ref_map = [None] * num_frames
    hyp_map = [None] * num_frames
    
    for s in reference_segments:
        start_idx = int(s['start'] / frame_step)
        end_idx = int(s['end'] / frame_step)
        for i in range(start_idx, min(end_idx, num_frames)):
            ref_map[i] = s['speaker']
            
    for s in hypothesis_segments:
        start_idx = int(s['start'] / frame_step)
        end_idx = int(s['end'] / frame_step)
        for i in range(start_idx, min(end_idx, num_frames)):
            hyp_map[i] = s['speaker']
            
    missed = 0
    false_alarm = 0
    confusion = 0
    correct = 0
    
    for r, h in zip(ref_map, hyp_map):
        if r is None and h is None:
            continue
        if r is not None and h is None:
            missed += 1
        elif r is None and h is not None:
            false_alarm += 1
        elif r != h:
            confusion += 1
        else:
            correct += 1
            
    # Convert frames back to time
    der = (missed + false_alarm + confusion) * frame_step / total_ref_time
    return der * 100.0 # Percentage

def calculate_role_f1(y_true, y_pred):
    """Calculate F1 score for role classification."""
    # Simple macro-F1 implementation
    labels = set(y_true) | set(y_pred)
    f1_scores = []
    
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        f1_scores.append(f1)
        
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

def simple_rouge_1(ref_text, hyp_text):
    """Calculate ROUGE-1 (unigram overlap)."""
    def get_unigrams(text):
        return Counter(text.lower().split())
        
    ref_counts = get_unigrams(ref_text)
    hyp_counts = get_unigrams(hyp_text)
    
    overlap = 0
    for token in hyp_counts:
        overlap += min(hyp_counts[token], ref_counts[token])
        
    precision = overlap / sum(hyp_counts.values()) if sum(hyp_counts.values()) > 0 else 0
    recall = overlap / sum(ref_counts.values()) if sum(ref_counts.values()) > 0 else 0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1 * 100

def simple_rouge_l(ref_text, hyp_text):
    """Calculate ROUGE-L (Longest Common Subsequence)."""
    # Simplified LCS implementation
    ref_tokens = ref_text.lower().split()
    hyp_tokens = hyp_text.lower().split()
    
    m = len(ref_tokens)
    n = len(hyp_tokens)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
    lcs_len = dp[m][n]
    
    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1 * 100

def evaluate_temporal_smoothing():
    """
    Evaluate the impact of temporal smoothing on 'watchability'.
    Metric: Number of micro-cuts (< 2 seconds).
    """
    print("\n--- 6.3.2 Impact of Temporal Smoothing ---")
    
    # 1. Create dummy scored segments (simulating raw output)
    # Pattern: [High, Low, High, Low, High] -> Fragmented
    segments = [
        {'start': 10.0, 'end': 11.5, 'text': "A", 'score': 0.8}, # 1.5s (Micro)
        {'start': 12.0, 'end': 13.0, 'text': "B", 'score': 0.2}, # Gap 0.5s
        {'start': 13.5, 'end': 14.5, 'text': "C", 'score': 0.9}, # 1.0s (Micro)
        {'start': 15.0, 'end': 16.0, 'text': "D", 'score': 0.3}, # Gap 0.5s
        {'start': 16.5, 'end': 19.0, 'text': "E", 'score': 0.85}, # 2.5s (OK)
        {'start': 20.0, 'end': 21.0, 'text': "F", 'score': 0.8}, # 1.0s (Micro)
    ]
    
    print(f"Input Segments: {len(segments)}")
    
    # 2. Without Smoothing (min_gap=0, padding=0)
    # Just filter by score
    raw_ranges = filter_and_smooth(segments, threshold=0.5, min_gap=0.0, padding=0.0)
    
    micro_cuts_raw = sum(1 for s, e in raw_ranges if (e - s) < 2.0)
    print(f"\n[Without Smoothing]")
    print(f"Selected Ranges: {raw_ranges}")
    print(f"Micro-cuts (< 2s): {micro_cuts_raw}")
    
    # 3. With Smoothing (Module 5)
    # min_gap=2.0, padding=0.5 (as per paper/code default)
    smooth_ranges = filter_and_smooth(segments, threshold=0.5, min_gap=2.0, padding=0.0)
    
    micro_cuts_smooth = sum(1 for s, e in smooth_ranges if (e - s) < 2.0)
    print(f"\n[With Temporal Smoothing (Module 5)]")
    print(f"Selected Ranges: {smooth_ranges}")
    print(f"Micro-cuts (< 2s): {micro_cuts_smooth}")
    
    reduction = micro_cuts_raw - micro_cuts_smooth
    print(f"\nResult: Temporal Smoothing reduced micro-cuts by {reduction} ({reduction/micro_cuts_raw*100:.1f}% reduction).")

def main():
    print("Running RoME Architecture Evaluation...")
    
    # 1. Temporal Smoothing (Can run without external data)
    evaluate_temporal_smoothing()
    
    # 2. Placeholders for Data-Dependent Metrics
    print("\n--- 6.2 Evaluation Metrics (Demonstration) ---")
    print("Note: Actual calculation requires AMI Corpus and Ground Truth annotations.")
    
    # Mock Data for Demonstration
    print("\n[6.2.1 Diarization Error Rate (DER)]")
    # Simulating a scenario
    ref_segments = [
        {'start': 0, 'end': 5, 'speaker': 'spk1'},
        {'start': 5, 'end': 10, 'speaker': 'spk2'}
    ]
    hyp_segments = [
        {'start': 0, 'end': 4.8, 'speaker': 'spk1'}, # Missed 0.2
        {'start': 5.2, 'end': 10, 'speaker': 'spk2'} # Missed 0.2
    ]
    der = calculate_der(ref_segments, hyp_segments)
    print(f"Demo DER: {der:.2f}% (on mock data)")
    
    print("\n[6.2.2 Role Classification Accuracy]")
    y_true = ["PM", "Dev", "Dev", "Designer", "PM"]
    y_pred = ["PM", "Dev", "PM", "Designer", "PM"] # 1 error
    f1 = calculate_role_f1(y_true, y_pred)
    print(f"Demo F1-Score: {f1:.4f} (on mock data)")
    
    print("\n[6.2.3 ROUGE Scores]")
    ref_summary = "The project manager discussed the timeline and the developer mentioned the API issues."
    hyp_summary = "Project manager talked about timeline. Developer discussed API problems."
    
    r1 = simple_rouge_1(ref_summary, hyp_summary)
    rl = simple_rouge_l(ref_summary, hyp_summary)
    
    print(f"Reference: {ref_summary}")
    print(f"Generated: {hyp_summary}")
    print(f"ROUGE-1: {r1:.2f}")
    print(f"ROUGE-L: {rl:.2f}")

if __name__ == "__main__":
    main()
