import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from src.visual_analysis import VisualAnalyzer
    print("✅ VisualAnalyzer imported successfully.")
    
    # Check dependencies
    import torchvision
    import easyocr
    print(f"✅ torchvision: {torchvision.__version__}")
    print(f"✅ easyocr: {easyocr.__version__}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
