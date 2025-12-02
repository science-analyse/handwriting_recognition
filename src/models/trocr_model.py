"""
TrOCR model for line-level handwriting recognition.
Based on Microsoft's TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models.
"""

import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    AutoTokenizer,
    AutoFeatureExtractor
)
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image


class TrOCRModel:
    """Wrapper for TrOCR model with inference and fine-tuning capabilities."""

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        device: Optional[str] = None
    ):
        """
        Initialize TrOCR model.

        Args:
            model_name: Pretrained model identifier
            device: Device to use (cuda/cpu/mps)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and processor
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = TrOCRProcessor.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def recognize_text(
        self,
        image: Image.Image,
        max_length: int = 128,
        num_beams: int = 5,
        early_stopping: bool = True
    ) -> str:
        """
        Recognize text from a single line image.

        Args:
            image: PIL Image of text line
            max_length: Maximum generated text length
            num_beams: Beam search width
            early_stopping: Whether to stop early

        Returns:
            Recognized text string
        """
        # Preprocess image
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping
            )

        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return generated_text

    def recognize_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Recognize text from multiple images.

        Args:
            images: List of PIL Images
            batch_size: Batch size for processing
            **kwargs: Additional arguments for recognize_text

        Returns:
            List of recognized text strings
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Process batch
            pixel_values = self.processor(
                images=batch,
                return_tensors="pt"
            ).pixel_values.to(self.device)

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=kwargs.get('max_length', 128),
                    num_beams=kwargs.get('num_beams', 5),
                    early_stopping=kwargs.get('early_stopping', True)
                )

            # Decode
            batch_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            results.extend(batch_texts)

        return results

    def recognize_with_confidence(
        self,
        image: Image.Image,
        max_length: int = 128,
        num_beams: int = 5
    ) -> Tuple[str, float]:
        """
        Recognize text with confidence score.

        Args:
            image: PIL Image of text line
            max_length: Maximum generated text length
            num_beams: Beam search width

        Returns:
            Tuple of (recognized text, confidence score)
        """
        # Preprocess
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        # Generate with scores
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Decode
        generated_text = self.processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True
        )[0]

        # Calculate confidence from scores
        if hasattr(outputs, 'sequences_scores'):
            confidence = torch.exp(outputs.sequences_scores[0]).item()
        else:
            # Approximate confidence from token probabilities
            scores = torch.stack(outputs.scores, dim=1)  # [batch, seq_len, vocab]
            probs = torch.softmax(scores, dim=-1)
            max_probs = probs.max(dim=-1).values
            confidence = max_probs.mean().item()

        return generated_text, confidence

    def save_model(self, save_path: str):
        """
        Save model and processor.

        Args:
            save_path: Directory to save model
        """
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, model_path: str, device: Optional[str] = None):
        """
        Load model from local path.

        Args:
            model_path: Path to saved model
            device: Device to use

        Returns:
            Loaded TrOCRModel instance
        """
        instance = cls.__new__(cls)
        instance.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        instance.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        instance.processor = TrOCRProcessor.from_pretrained(model_path)

        instance.model.to(instance.device)
        instance.model.eval()

        return instance


class TrOCRFineTuner:
    """Fine-tuning wrapper for TrOCR."""

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        learning_rate: float = 5e-5,
        device: Optional[str] = None
    ):
        """
        Initialize fine-tuner.

        Args:
            model_name: Base model identifier
            learning_rate: Learning rate
            device: Device to use
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = TrOCRProcessor.from_pretrained(model_name)

        self.model.to(self.device)

        # Set up optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

    def prepare_batch(
        self,
        images: List[Image.Image],
        texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for training.

        Args:
            images: List of PIL Images
            texts: List of ground truth texts

        Returns:
            Dictionary with model inputs
        """
        # Encode images
        pixel_values = self.processor(
            images=images,
            return_tensors="pt"
        ).pixel_values

        # Encode texts
        labels = self.processor.tokenizer(
            texts,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).input_ids

        # Replace padding token id's of the labels by -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values.to(self.device),
            "labels": labels.to(self.device)
        }

    def train_step(
        self,
        images: List[Image.Image],
        texts: List[str]
    ) -> float:
        """
        Single training step.

        Args:
            images: Batch of images
            texts: Batch of texts

        Returns:
            Loss value
        """
        self.model.train()

        # Prepare batch
        batch = self.prepare_batch(images, texts)

        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(
        self,
        images: List[Image.Image],
        texts: List[str]
    ) -> float:
        """
        Evaluate on validation data.

        Args:
            images: Validation images
            texts: Ground truth texts

        Returns:
            Validation loss
        """
        self.model.eval()

        with torch.no_grad():
            batch = self.prepare_batch(images, texts)
            outputs = self.model(**batch)
            loss = outputs.loss

        return loss.item()


def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Character Error Rate.

    Args:
        predictions: Predicted texts
        references: Ground truth texts

    Returns:
        CER score (0-1)
    """
    from jiwer import cer
    return cer(references, predictions)


def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Word Error Rate.

    Args:
        predictions: Predicted texts
        references: Ground truth texts

    Returns:
        WER score (0-1)
    """
    from jiwer import wer
    return wer(references, predictions)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Initialize model
        print("Loading TrOCR model...")
        model = TrOCRModel()

        # Recognize text
        print("Recognizing text...")
        text, confidence = model.recognize_with_confidence(image)

        print(f"\nRecognized text: {text}")
        print(f"Confidence: {confidence:.3f}")
    else:
        print("Usage: python trocr_model.py <image_path>")
