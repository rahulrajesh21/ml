
import sys
import os
sys.path.append(os.getcwd())

from src.visual_analysis import VisualAnalyzer
import torch.nn as nn

def test_model_loading():
    print("Testing VisualAnalyzer model loading...")
    
    # Initialize
    analyzer = VisualAnalyzer(device="cpu", model_path="resnet50_meeting_context.pth")
    
    if analyzer.is_ready:
        print("✅ VisualAnalyzer initialized successfully.")
    else:
        print("❌ VisualAnalyzer failed to initialize.")
        return

    # Check components
    if analyzer.feature_extractor is not None:
        print("✅ Feature Extractor loaded.")
    else:
        print("❌ Feature Extractor missing.")

    if analyzer.classifier is not None:
        print("✅ Classifier loaded.")
        # Check output features
        if isinstance(analyzer.classifier, nn.Linear):
            print(f"   Classifier Output Features: {analyzer.classifier.out_features}")
            if analyzer.classifier.out_features == 2:
                print("✅ Classifier has correct number of classes (2).")
            else:
                print(f"❌ Classifier has wrong number of classes: {analyzer.classifier.out_features} (Expected 2)")
    else:
        print("❌ Classifier missing.")

if __name__ == "__main__":
    test_model_loading()
