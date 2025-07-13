"""
Medical LLM Fine-tuning for Clinical Guidelines
===============================================

This module provides functionality to fine-tune Large Language Models (LLMs) 
on medical guidelines and clinical documentation, specifically focusing on 
diabetes management in chronic kidney disease (CKD).

Author: Healthcare AI Team
Date: 2024
"""

import os
import logging
from typing import Optional, Dict, Any
import torch
from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from PyPDF2 import PdfReader
from torch.utils.data import Dataset
import sys
sys.path.append('..')
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedicalTextDataset(Dataset):
    """
    Custom Dataset class for medical text data
    """
    
    def __init__(self, texts: list, tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


class MedicalLLMTrainer:
    """
    Main class for training medical LLMs on clinical guidelines
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from PDF file
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF reading fails
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
            logger.info(f"Successfully extracted {len(text)} characters from {len(reader.pages)} pages")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def prepare_training_data(self, text: str) -> list:
        """
        Prepare training data by chunking text into manageable pieces
        
        Args:
            text (str): Raw text content
            
        Returns:
            list: List of text chunks for training
        """
        # Split text into chunks (simple approach - can be improved)
        chunk_size = 1000
        chunks = []
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if len(chunk.strip()) > 100:  # Only include substantial chunks
                chunks.append(chunk.strip())
                
        logger.info(f"Prepared {len(chunks)} training chunks")
        return chunks
    
    def load_model_and_tokenizer(self):
        """
        Load the pre-trained model and tokenizer
        
        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info(f"Loading model: {self.config.MODEL_NAME}")
            
            # Load tokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(
                self.config.MODEL_NAME,
                token=self.config.HUGGINGFACE_TOKEN
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = LlamaForCausalLM.from_pretrained(
                self.config.MODEL_NAME,
                token=self.config.HUGGINGFACE_TOKEN,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def setup_trainer(self, train_dataset: Dataset):
        """
        Setup the Hugging Face Trainer
        
        Args:
            train_dataset (Dataset): Training dataset
        """
        # Create output directory
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.config.LOG_DIR, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.OUTPUT_DIR,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            logging_dir=self.config.LOG_DIR,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_pin_memory=False,
            learning_rate=self.config.LEARNING_RATE,
            fp16=torch.cuda.is_available(),
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        logger.info("Trainer setup completed")
    
    def train(self):
        """
        Execute the training process
        """
        try:
            logger.info("Starting model fine-tuning...")
            self.trainer.train()
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def save_model(self):
        """
        Save the fine-tuned model and tokenizer
        """
        try:
            logger.info(f"Saving model to {self.config.OUTPUT_DIR}")
            self.model.save_pretrained(self.config.OUTPUT_DIR)
            self.tokenizer.save_pretrained(self.config.OUTPUT_DIR)
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def run_training_pipeline(self):
        """
        Execute the complete training pipeline
        """
        try:
            # Validate configuration
            Config.validate()
            
            # Extract text from PDF
            text_data = self.extract_text_from_pdf(self.config.PDF_PATH)
            
            # Prepare training data
            training_chunks = self.prepare_training_data(text_data)
            
            # Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Create dataset
            train_dataset = MedicalTextDataset(
                training_chunks, 
                self.tokenizer, 
                self.config.MAX_LENGTH
            )
            
            # Setup trainer
            self.setup_trainer(train_dataset)
            
            # Train the model
            self.train()
            
            # Save the model
            self.save_model()
            
            logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


def main():
    """
    Main function to run the training pipeline
    """
    try:
        # Initialize configuration
        config = Config()
        
        # Create trainer instance
        trainer = MedicalLLMTrainer(config)
        
        # Run training pipeline
        trainer.run_training_pipeline()
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 