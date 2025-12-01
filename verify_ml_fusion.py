import sys
import os
import torch
import numpy as np
from dataclasses import dataclass

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ml_fusion import ContextualFusionTransformer, save_model, load_model
from train_fusion import FusionTrainer
from fusion_layer import SegmentFeatures

def test_model_architecture():
    print("Testing Model Architecture...")
    model = ContextualFusionTransformer()
    
    # Dummy Input (Batch=1, Seq=10)
    text = torch.randn(1, 10, 384)
    audio = torch.randn(1, 10, 64)
    role = torch.randn(1, 10, 384)
    
    output = model(text, audio, role)
    print(f"Output Shape: {output.shape}")
    assert output.shape == (1, 10, 1)
    print("✅ Architecture Test Passed")

def test_training_loop():
    print("\nTesting Training Loop...")
    trainer = FusionTrainer(llm_summarizer=None) # No LLM for test
    
    # Dummy Segments
    segments = []
    for i in range(10):
        seg = SegmentFeatures(
            start_time=i*1.0,
            end_time=(i+1)*1.0,
            text=f"Segment {i}",
            speaker="SPEAKER_00"
        )
        seg.text_embedding = np.random.randn(384)
        seg.mfcc_embedding = np.random.randn(64)
        seg.fused_score = 0.8 if i % 2 == 0 else 0.2 # Fake heuristic score
        segments.append(seg)
        
    loss = trainer.train_step(segments, epochs=2)
    print(f"Training Loss: {loss}")
    assert os.path.exists("fusion_model.pth")
    print("✅ Training Test Passed")

def test_inference():
    print("\nTesting Inference...")
    model = load_model("fusion_model.pth")
    model.eval()
    
    text = torch.randn(1, 5, 384)
    output = model(text) # Audio/Role optional
    print(f"Inference Output: {output.squeeze()}")
    print("✅ Inference Test Passed")

if __name__ == "__main__":
    try:
        test_model_architecture()
        test_training_loop()
        test_inference()
        print("\n🎉 All ML Fusion Tests Passed!")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
