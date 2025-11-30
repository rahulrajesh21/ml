
import os
import cv2
import torch
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
    1. ResNet-50 Feature Extraction (Visual Encoder)
    2. OCR for Slide Text Extraction (Context)
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.resnet_model = None
        self.transform = None
        self.ocr_reader = None
        self.is_ready = False
        
        # Initialize immediately
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ResNet-50 and EasyOCR."""
        try:
            logger.info(f"Initializing VisualAnalyzer on {self.device}...")
            
            # 1. Load ResNet-50
            from torchvision import models, transforms
            
            # Use pretrained=True for compatibility with older torchvision versions
            self.resnet_model = models.resnet50(pretrained=True)
            
            # Remove the classification layer (fc) to get feature vectors
            # ResNet-50 fc layer input is 2048
            self.resnet_model = torch.nn.Sequential(*list(self.resnet_model.children())[:-1])
            
            self.resnet_model.eval()
            
            # Handle device
            if self.device == "cuda" and torch.cuda.is_available():
                self.resnet_model = self.resnet_model.to("cuda")
            elif self.device == "mps" and torch.backends.mps.is_available():
                self.resnet_model = self.resnet_model.to("mps")
            
            # Define transform manually since weights.transforms() might not exist
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # 2. Load EasyOCR (Lazy load recommended for speed, but we'll do it here)
            import easyocr
            # gpu=True if cuda is available, easyocr handles mps poorly sometimes so stick to cpu or cuda
            use_gpu = (self.device == "cuda")
            self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            
            self.is_ready = True
            logger.info("VisualAnalyzer initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error initializing VisualAnalyzer: {e}")
            self.is_ready = False

    def extract_frames(self, video_path: str, interval: float = 2.0) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames from video at a given time interval.
        
        Args:
            video_path: Path to video file
            interval: Time in seconds between frames
            
        Returns:
            List of (timestamp, frame_bgr) tuples
        """
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

    def get_visual_embedding(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Get ResNet-50 feature vector for a frame.
        
        Args:
            frame_bgr: BGR image from OpenCV
            
        Returns:
            2048-dim numpy array
        """
        if not self.is_ready or self.resnet_model is None:
            return None
            
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Transform
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            # Move to device
            if next(self.resnet_model.parameters()).device.type != 'cpu':
                input_tensor = input_tensor.to(next(self.resnet_model.parameters()).device)
            
            # Forward pass
            with torch.no_grad():
                output = self.resnet_model(input_tensor)
                
            # Flatten: [1, 2048, 1, 1] -> [2048]
            embedding = output.squeeze().cpu().numpy()
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting visual embedding: {e}")
            return None

    def extract_slide_info(self, frame_bgr: np.ndarray) -> Tuple[str, bool, float]:
        """
        Extract text and determine if the frame is likely a presentation slide.
        
        Args:
            frame_bgr: BGR image from OpenCV
            
        Returns:
            Tuple of (full_text, is_slide, confidence_score)
        """
        if not self.is_ready or self.ocr_reader is None:
            return "", False, 0.0
            
        try:
            # Get detailed results: List of (bbox, text, prob)
            results = self.ocr_reader.readtext(frame_bgr, detail=1)
            
            if not results:
                return "", False, 0.0
            
            # Filter low confidence text
            valid_text = []
            total_conf = 0.0
            text_area = 0.0
            
            height, width, _ = frame_bgr.shape
            frame_area = height * width
            
            for bbox, text, prob in results:
                if prob > 0.4: # Confidence threshold
                    valid_text.append(text)
                    total_conf += prob
                    
                    # Estimate area of text box
                    # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    # Approximate width/height
                    w = abs(bbox[1][0] - bbox[0][0])
                    h = abs(bbox[2][1] - bbox[1][1])
                    text_area += (w * h)
            
            full_text = " ".join(valid_text)
            
            # Heuristic for "Is this a slide?":
            # 1. Significant amount of text (at least 3 words/blocks)
            # 2. Text covers a reasonable portion of screen (optional, but good for large titles)
            # 3. High average confidence
            
            is_slide = False
            score = 0.0
            
            if len(valid_text) >= 3:
                is_slide = True
                score = total_conf / len(valid_text)
            elif len(valid_text) > 0 and (text_area / frame_area) > 0.05:
                # Large text (title slide?)
                is_slide = True
                score = total_conf / len(valid_text)
                
            return full_text, is_slide, score
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return "", False, 0.0

    def analyze_video_context(self, video_path: str) -> List[Dict]:
        """
        Full pipeline: Extract frames -> Get Embeddings -> Get OCR Text -> Detect Slide.
        
        Returns:
            List of dicts: {'timestamp': float, 'embedding': np.array, 'ocr_text': str, 'is_slide': bool}
        """
        frames = self.extract_frames(video_path, interval=5.0) # Every 5 seconds
        results = []
        
        logger.info(f"Analyzing {len(frames)} frames from video...")
        
        for timestamp, frame in frames:
            embedding = self.get_visual_embedding(frame)
            text, is_slide, conf = self.extract_slide_info(frame)
            
            # Only store if we have meaningful data
            if embedding is not None:
                results.append({
                    'timestamp': timestamp,
                    'embedding': embedding,
                    'ocr_text': text,
                    'is_slide': is_slide,
                    'slide_confidence': conf
                })
                
        return results
