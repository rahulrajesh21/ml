import logging
from typing import Optional
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMSummarizer:
    """
    Generates coherent, role-specific summaries using a local LLM.
    """
    
    def __init__(self, device: str = "cpu", model_name: str = "MBZUAI/LaMini-Flan-T5-248M"):
        """
        Initialize the LLM summarizer.
        
        Args:
            device: Device to run on ("cpu", "cuda", "mps")
            model_name: HuggingFace model name (default: LaMini-Flan-T5-248M for speed/quality balance)
        """
        self.device = -1
        if device == "cuda" and torch.cuda.is_available():
            self.device = 0
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = 0 # MPS handling varies, but pipeline often accepts 0 or device object
            
        self.model_name = model_name
        self.pipeline = None
        self.is_ready = False
        
        self._initialize_model()
        
    def _initialize_model(self):
        try:
            logger.info(f"Loading summarization model: {self.model_name}...")
            # Use text2text-generation for T5
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                device=self.device,
                max_length=512,
                truncation=True
            )
            self.is_ready = True
            logger.info("LLM Summarizer initialized!")
        except Exception as e:
            logger.error(f"Error initializing LLM Summarizer: {e}")
            self.is_ready = False

    def summarize(self, text: str, role: str, focus: str = "key decisions and action items") -> str:
        """
        Generate a role-specific summary.
        
        Args:
            text: The transcript text to summarize
            role: The target role (e.g., "Product Manager")
            focus: Specific focus area (optional)
            
        Returns:
            Generated summary string
        """
        if not self.is_ready or not text:
            return "Summarizer not available or empty text."
            
        # Construct a prompt that guides the model
        # LaMini-Flan-T5 follows instructions well
        prompt = f"""
        You are a {role}. Analyze the following meeting transcript and write a concise summary focusing on {focus}.
        
        Transcript:
        {text[:2000]} 
        
        Summary:
        """
        # Note: Truncating text to 2000 chars to fit context window if needed, 
        # but for a real app we might need chunking. For now, simple truncation.
        
        try:
            output = self.pipeline(prompt, max_length=256, do_sample=False)
            return output[0]['generated_text']
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return f"Error generating summary: {e}"
