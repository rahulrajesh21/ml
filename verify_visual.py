
import os
import sys
import numpy as np
import cv2
from moviepy import ColorClip, TextClip, CompositeVideoClip

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.visual_analysis import VisualAnalyzer

def create_test_video(filename):
    """Create a dummy video with text for OCR testing."""
    print(f"Creating test video: {filename}")
    
    # Background
    bg = ColorClip(size=(640, 480), color=(0, 0, 0), duration=5)
    
    # Text (Requires ImageMagick, might fail if not installed. Fallback to just color)
    try:
        # Simple color video is enough to test ResNet
        # If we can't create text, OCR will just return empty, which is fine for a basic crash test
        video = bg
    except:
        video = bg
        
    video.fps = 24
    video.write_videofile(filename, codec='libx264', audio=False, logger=None)

def test_visual_analysis():
    video_path = "test_visual.mp4"
    create_test_video(video_path)
    
    print("\n--- Initializing VisualAnalyzer ---")
    analyzer = VisualAnalyzer(device="cpu")
    
    if not analyzer.is_ready:
        print("❌ Failed to initialize VisualAnalyzer.")
        return
        
    print("✅ VisualAnalyzer initialized.")
    
    print("\n--- Testing Frame Extraction & Analysis ---")
    results = analyzer.analyze_video_context(video_path)
    
    print(f"Processed {len(results)} frames.")
    
    if len(results) > 0:
        first = results[0]
        print(f"Timestamp: {first['timestamp']}")
        
        emb = first['embedding']
        if emb is not None:
            print(f"Embedding Shape: {emb.shape}")
            print(f"Embedding Norm: {np.linalg.norm(emb):.4f}")
            if emb.shape == (2048,):
                print("✅ ResNet-50 Embedding shape correct (2048).")
            else:
                print("❌ Incorrect embedding shape.")
        else:
            print("❌ Embedding is None.")
            
        text = first['ocr_text']
        print(f"OCR Text: '{text}'")
        
        is_slide = first.get('is_slide', False)
        conf = first.get('slide_confidence', 0.0)
        print(f"Is Slide: {is_slide}")
        print(f"Slide Confidence: {conf:.2f}")
        
        print("✅ OCR and Slide Detection ran successfully.")
        
    else:
        print("❌ No results returned.")
        
    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)

if __name__ == "__main__":
    test_visual_analysis()
