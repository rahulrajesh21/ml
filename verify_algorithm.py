import unittest
from typing import List, Dict, Tuple

# Copied logic to verify algorithm without heavy imports
def filter_and_smooth(
    scored_segments: List[Dict], 
    threshold: float = 0.5, 
    min_gap: float = 2.0,
    padding: float = 0.5
) -> List[Tuple[float, float]]:
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

class TestAlgorithm(unittest.TestCase):
    def test_filter_and_smooth_simple(self):
        # Test basic filtering
        scored = [
            {'start': 0, 'end': 5, 'score': 0.8},
            {'start': 10, 'end': 15, 'score': 0.2}, # Should be filtered out
            {'start': 20, 'end': 25, 'score': 0.9}
        ]
        
        ranges = filter_and_smooth(scored, threshold=0.5, padding=0)
        
        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0], (0, 5))
        self.assertEqual(ranges[1], (20, 25))
        
    def test_filter_and_smooth_merge(self):
        # Test merging of close segments
        scored = [
            {'start': 0, 'end': 5, 'score': 0.8},
            {'start': 6, 'end': 10, 'score': 0.8} # Gap is 1s, should merge if min_gap >= 1
        ]
        
        # min_gap=2.0, padding=0
        ranges = filter_and_smooth(scored, threshold=0.5, min_gap=2.0, padding=0)
        
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0], (0, 10))
        
    def test_filter_and_smooth_padding(self):
        # Test padding
        scored = [
            {'start': 10, 'end': 15, 'score': 0.8}
        ]
        
        # padding=1.0
        ranges = filter_and_smooth(scored, threshold=0.5, padding=1.0)
        
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0], (9.0, 16.0))

if __name__ == '__main__':
    unittest.main()
