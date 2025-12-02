"""
LayoutLMv3 model for multimodal document understanding.
Combines text, layout, and visual information for structured information extraction.
"""

import torch
import torch.nn as nn
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ForSequenceClassification
)
from typing import List, Dict, Tuple, Optional
from PIL import Image
import numpy as np


class LayoutLMv3Model:
    """
    LayoutLMv3 wrapper for document information extraction.
    Supports token classification (NER) and sequence classification tasks.
    """

    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        task: str = "token_classification",
        num_labels: int = 9,
        device: Optional[str] = None
    ):
        """
        Initialize LayoutLMv3 model.

        Args:
            model_name: Pretrained model identifier
            task: Task type - 'token_classification' or 'sequence_classification'
            num_labels: Number of labels for classification
            device: Device to use
        """
        self.model_name = model_name
        self.task = task
        self.num_labels = num_labels
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load processor
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_name,
            apply_ocr=False  # We'll provide OCR results
        )

        # Load model based on task
        if task == "token_classification":
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        elif task == "sequence_classification":
            self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        self.model.to(self.device)
        self.model.eval()

        # Label mapping (example - customize based on your task)
        self.id2label = {
            0: "O",
            1: "B-NAME",
            2: "I-NAME",
            3: "B-DATE",
            4: "I-DATE",
            5: "B-NUMBER",
            6: "I-NUMBER",
            7: "B-LOCATION",
            8: "I-LOCATION"
        }
        self.label2id = {v: k for k, v in self.id2label.items()}

    def extract_entities(
        self,
        image: Image.Image,
        words: List[str],
        boxes: List[List[int]]
    ) -> List[Dict]:
        """
        Extract named entities from document.

        Args:
            image: Document image
            words: List of OCR words
            boxes: List of bounding boxes [x0, y0, x1, y1] (normalized to 1000)

        Returns:
            List of entity dicts with 'text', 'label', 'box', 'confidence'
        """
        if self.task != "token_classification":
            raise ValueError("Entity extraction requires token_classification task")

        # Prepare inputs
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            probabilities = torch.softmax(outputs.logits, dim=-1).squeeze()

        # Decode predictions
        entities = []
        current_entity = None

        for idx, (word, box, pred_id) in enumerate(zip(words, boxes, predictions)):
            # Skip special tokens
            if idx >= len(words):
                break

            label = self.id2label.get(pred_id, "O")
            confidence = probabilities[idx][pred_id].item()

            if label.startswith("B-"):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)

                current_entity = {
                    "text": word,
                    "label": label[2:],  # Remove B- prefix
                    "box": box,
                    "confidence": confidence
                }
            elif label.startswith("I-") and current_entity:
                # Continue current entity
                current_entity["text"] += " " + word
                current_entity["box"][2] = box[2]  # Extend box
                current_entity["box"][3] = max(current_entity["box"][3], box[3])
                current_entity["confidence"] = min(current_entity["confidence"], confidence)
            else:
                # O label or mismatched I- tag
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Append last entity
        if current_entity:
            entities.append(current_entity)

        return entities

    def classify_document(
        self,
        image: Image.Image,
        words: List[str],
        boxes: List[List[int]]
    ) -> Tuple[str, float]:
        """
        Classify document type.

        Args:
            image: Document image
            words: List of OCR words
            boxes: List of bounding boxes

        Returns:
            Tuple of (predicted class, confidence)
        """
        if self.task != "sequence_classification":
            raise ValueError("Document classification requires sequence_classification task")

        # Prepare inputs
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.softmax(outputs.logits, dim=-1)
            pred_class = predictions.argmax(-1).item()
            confidence = predictions[0][pred_class].item()

        predicted_label = self.id2label.get(pred_class, f"CLASS_{pred_class}")

        return predicted_label, confidence

    def save_model(self, save_path: str):
        """Save model and processor."""
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        task: str = "token_classification",
        device: Optional[str] = None
    ):
        """Load model from local path."""
        instance = cls.__new__(cls)
        instance.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        instance.task = task

        instance.processor = LayoutLMv3Processor.from_pretrained(model_path)

        if task == "token_classification":
            instance.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        else:
            instance.model = LayoutLMv3ForSequenceClassification.from_pretrained(model_path)

        instance.model.to(instance.device)
        instance.model.eval()

        return instance


def normalize_box(box: List[int], width: int, height: int) -> List[int]:
    """
    Normalize bounding box to 1000x1000 scale (required by LayoutLMv3).

    Args:
        box: [x0, y0, x1, y1] in pixel coordinates
        width: Image width
        height: Image height

    Returns:
        Normalized box [x0, y0, x1, y1]
    """
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


def denormalize_box(box: List[int], width: int, height: int) -> List[int]:
    """
    Denormalize bounding box from 1000x1000 scale to pixel coordinates.

    Args:
        box: [x0, y0, x1, y1] in normalized coordinates (0-1000)
        width: Image width
        height: Image height

    Returns:
        Box in pixel coordinates
    """
    return [
        int(box[0] * width / 1000),
        int(box[1] * height / 1000),
        int(box[2] * width / 1000),
        int(box[3] * height / 1000),
    ]


class LayoutLMv3FineTuner:
    """Fine-tuning wrapper for LayoutLMv3."""

    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        task: str = "token_classification",
        num_labels: int = 9,
        learning_rate: float = 5e-5,
        device: Optional[str] = None
    ):
        """Initialize fine-tuner."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.task = task

        self.processor = LayoutLMv3Processor.from_pretrained(
            model_name,
            apply_ocr=False
        )

        if task == "token_classification":
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        else:
            self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

    def prepare_batch(
        self,
        images: List[Image.Image],
        words_list: List[List[str]],
        boxes_list: List[List[List[int]]],
        labels_list: Optional[List[List[int]]] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch for training."""
        # Process batch
        encodings = []
        for image, words, boxes in zip(images, words_list, boxes_list):
            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
            encodings.append(encoding)

        # Stack encodings
        batch = {
            "pixel_values": torch.cat([e["pixel_values"] for e in encodings], dim=0).to(self.device),
            "input_ids": torch.cat([e["input_ids"] for e in encodings], dim=0).to(self.device),
            "attention_mask": torch.cat([e["attention_mask"] for e in encodings], dim=0).to(self.device),
            "bbox": torch.cat([e["bbox"] for e in encodings], dim=0).to(self.device),
        }

        if labels_list is not None:
            # Pad labels
            max_len = batch["input_ids"].shape[1]
            padded_labels = []

            for labels in labels_list:
                padded = labels + [-100] * (max_len - len(labels))
                padded_labels.append(padded[:max_len])

            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long).to(self.device)

        return batch

    def train_step(
        self,
        images: List[Image.Image],
        words_list: List[List[str]],
        boxes_list: List[List[List[int]]],
        labels_list: List[List[int]]
    ) -> float:
        """Single training step."""
        self.model.train()

        batch = self.prepare_batch(images, words_list, boxes_list, labels_list)

        outputs = self.model(**batch)
        loss = outputs.loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Example OCR results (you would get these from TrOCR or other OCR)
        words = ["SOCAR", "Hackathon", "2025", "Date:", "13-14", "December"]
        boxes = [
            [100, 100, 200, 150],
            [220, 100, 350, 150],
            [370, 100, 450, 150],
            [100, 200, 180, 240],
            [200, 200, 300, 240],
            [320, 200, 450, 240]
        ]

        # Normalize boxes
        width, height = image.size
        normalized_boxes = [normalize_box(box, width, height) for box in boxes]

        # Initialize model
        print("Loading LayoutLMv3 model...")
        model = LayoutLMv3Model()

        # Extract entities
        print("Extracting entities...")
        entities = model.extract_entities(image, words, normalized_boxes)

        print(f"\nExtracted {len(entities)} entities:")
        for entity in entities:
            print(f"  {entity['label']}: {entity['text']} (confidence: {entity['confidence']:.3f})")
    else:
        print("Usage: python layoutlm_model.py <image_path>")
