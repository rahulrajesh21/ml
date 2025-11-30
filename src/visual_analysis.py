
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualAnalyzer:
    """
    Analyzer for extracting visual features and context from video frames.
    Implements:
    1. Fine-tuned ResNet-50 for Context Classification (Slide vs Person)
    2. Feature Extraction for Similarity (Embeddings)
    3. OCR for Text Extraction
    """
    
    def __init__(self, device: str = "cpu", model_path: str = "resnet50_meeting_context.pth"):
        self.device = device
        self.model_path = model_path
        self.feature_extractor = None
        self.classifier = None
        self.transform = None
        self.ocr_reader = None
        self.is_ready = False
        
        # Initialize immediately
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize Fine-Tuned ResNet-50 and EasyOCR."""
        try:
            logger.info(f"Initializing VisualAnalyzer on {self.device}...")
            
            # 1. Load ResNet-50
            from torchvision import models, transforms
            
            # Initialize standard ResNet-50 structure
            full_model = models.resnet50(pretrained=False) # We load custom weights
            
            # Modify the final layer to match our fine-tuning (2 classes)
            num_ftrs = full_model.fc.in_features
            full_model.fc = nn.Linear(num_ftrs, 2)
            
            # Load custom weights if available
            if os.path.exists(self.model_path):
                logger.info(f"Loading fine-tuned model from {self.model_path}")
                state_dict = torch.load(self.model_path, map_location=self.device)
                full_model.load_state_dict(state_dict)
            else:
                logger.warning(f"Fine-tuned model not found at {self.model_path}. Using random weights (Context results will be poor).")
            
            full_model.eval()
            
            # Handle device
            if self.device == "cuda" and torch.cuda.is_available():
                full_model = full_model.to("cuda")
            elif self.device == "mps" and torch.backends.mps.is_available():
                full_model = full_model.to("mps")
                
            # Split into Feature Extractor and Classifier
            # Features: All layers except the last fc
            self.feature_extractor = nn.Sequential(*list(full_model.children())[:-1])
            # Classifier: The last fc layer
            self.classifier = full_model.fc
            
            # Define transform (standard ImageNet normalization)
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # 2. Load EasyOCR
            import easyocr
            use_gpu = (self.device == "cuda")
            self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            
            self.is_ready = True
            logger.info("VisualAnalyzer initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error initializing VisualAnalyzer: {e}")
            self.is_ready = False

    def extract_frames(self, video_path: str, interval: float = 2.0) -> List[Tuple[float, np.ndarray]]:
        """Extract frames from video at a given time interval."""
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return []
            
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            return []
            
        frame_interval = int(fps * interval)
        current_frame = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame % frame_interval == 0:
                timestamp = current_frame / fps
                frames.append((timestamp, frame))
                
            current_frame += 1
            
        cap.release()
        return frames

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str, float]:
        """
        Process a single frame to get:
        1. Visual Embedding (2048-dim)
        2. Context Label (Slide/Person)
        3. Context Confidence
        """
        if not self.is_ready or self.feature_extractor is None:
            return None, "Unknown", 0.0
            
        try:
            # Preprocess
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            # Move to device
            device = next(self.feature_extractor.parameters()).device
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                # 1. Get Features (Embedding)
                # Output shape: [1, 2048, 1, 1]
                features = self.feature_extractor(input_tensor)
                embedding = features.squeeze().cpu().numpy()
                
                # Normalize embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                # 2. Get Classification (Context)
                # Flatten features for fc layer: [1, 2048]
                flat_features = features.view(features.size(0), -1)
                logits = self.classifier(flat_features)
                probs = torch.softmax(logits, dim=1)
                
                # Classes: 0=Person, 1=Slide (Based on our training order)
                # We assume alphabetical order from ImageFolder unless specified otherwise.
                # In our notebook: Person (0), Slide (1)
                slide_prob = probs[0][1].item()
                person_prob = probs[0][0].item()
                
                if slide_prob > person_prob:
                    label = "Slide"
                    conf = slide_prob
                else:
                    label = "Person"
                    conf = person_prob
                    
            return embedding, label, conf
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None, "Error", 0.0

    def extract_ocr_text(self, frame_bgr: np.ndarray) -> str:
        """Extract text from frame using EasyOCR."""
        if not self.is_ready or self.ocr_reader is None:
            return ""
            
        try:
            results = self.ocr_reader.readtext(frame_bgr, detail=0)
            return " ".join(results)
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""

    def analyze_video_context(self, video_path: str) -> List[Dict]:
        """
        Full pipeline: Extract frames -> Get Embeddings -> Classify Context -> OCR.
        """
        frames = self.extract_frames(video_path, interval=5.0) # Every 5 seconds
        results = []
        
        logger.info(f"Analyzing {len(frames)} frames from video...")
        
        for timestamp, frame in frames:
            # 1. Visual Analysis (Embedding + Context)
            embedding, label, conf = self.process_frame(frame)
            
            # 2. OCR (Only if it looks like a slide to save time, or always?)
            # Let's do it always for now to be safe, or optimize later.
            ocr_text = ""
            if label == "Slide" or conf < 0.7: # Read text if it's a slide or we are unsure
                ocr_text = self.extract_ocr_text(frame)
            
            if embedding is not None:
                results.append({
                    'timestamp': timestamp,
                    'embedding': embedding,
                    'context_label': label,
                    'context_confidence': conf,
                    'ocr_text': ocr_text,
                    'is_slide': (label == "Slide") # Backward compatibility
                })
                
        return results
