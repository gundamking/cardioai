"""
Configuration file for Medical AI Training Project
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the project"""
    
    # Hugging Face Configuration
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    MODEL_NAME = os.getenv('MODEL_NAME', 'meta-llama/Llama-2-13b-chat-hf')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', './results/llama_finetuned')
    
    # Training Parameters
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '1'))
    MAX_LENGTH = int(os.getenv('MAX_LENGTH', '512'))
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '1'))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '2e-5'))
    
    # Data Paths
    PDF_PATH = os.getenv('PDF_PATH', './data/KDIGO-2022-Clinical-Practice-Guideline-for-Diabetes-Management-in-CKD.pdf')
    HEART_DATA_PATH = os.getenv('HEART_DATA_PATH', './data/statlog+heart/heart.dat')
    ECHO_DATA_PATH = os.getenv('ECHO_DATA_PATH', './data/echocardiogram/echocardiogram.data')
    
    # Logging
    LOG_DIR = os.getenv('LOG_DIR', './logs')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.HUGGINGFACE_TOKEN:
            raise ValueError("HUGGINGFACE_TOKEN is required. Please set it in your .env file or environment variables.")
        
        if not os.path.exists(cls.PDF_PATH):
            raise FileNotFoundError(f"PDF file not found at {cls.PDF_PATH}") 