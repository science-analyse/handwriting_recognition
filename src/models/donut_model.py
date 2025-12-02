"""
Donut (Document understanding transformer) - OCR-free document understanding model.
Based on: https://arxiv.org/abs/2111.15664
"""

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from typing import Dict, List, Optional
import json
import re
from PIL import Image


class DonutModel:
    """
    Donut model for OCR-free document understanding.
    Directly extracts structured information from document images.
    """

    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base",
        device: Optional[str] = None
    ):
        """
        Initialize Donut model.

        Args:
            model_name: Pretrained model identifier
            device: Device to use (cuda/cpu/mps)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and processor
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def extract_fields(
        self,
        image: Image.Image,
        task_prompt: Optional[str] = None,
        max_length: int = 512
    ) -> Dict:
        """
        Extract structured fields from document image.

        Args:
            image: PIL Image of document
            task_prompt: Task-specific prompt (e.g., "<s_docvqa><s_question>Extract fields</s_question><s_answer>")
            max_length: Maximum output length

        Returns:
            Dictionary of extracted fields
        """
        # Default task prompt for field extraction
        if task_prompt is None:
            task_prompt = "<s_docvqa><s_question>Extract all fields and values</s_question><s_answer>"

        # Prepare inputs
        pixel_values = self.processor(
            image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        # Encode task prompt
        task_prompt_ids = self.processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=task_prompt_ids,
                max_length=max_length,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # Decode
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
            self.processor.tokenizer.pad_token, ""
        )

        # Parse output to structured format
        result = self._parse_output(sequence, task_prompt)

        return result

    def _parse_output(self, sequence: str, task_prompt: str) -> Dict:
        """
        Parse model output into structured dictionary.

        Args:
            sequence: Generated sequence
            task_prompt: Original task prompt

        Returns:
            Parsed dictionary
        """
        # Remove task prompt from sequence
        sequence = sequence.replace(task_prompt, "")

        # Try to parse as JSON
        try:
            # Look for JSON-like structure
            json_match = re.search(r'\{.*\}', sequence, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except json.JSONDecodeError:
            pass

        # Fallback: parse key-value pairs
        result = {}
        lines = sequence.strip().split('\n')

        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()

        return result

    def extract_with_template(
        self,
        image: Image.Image,
        template: Dict[str, str]
    ) -> Dict:
        """
        Extract fields based on a template.

        Args:
            image: Document image
            template: Template dict with field names and descriptions
                     e.g., {"name": "Full name", "date": "Date in format DD/MM/YYYY"}

        Returns:
            Extracted fields matching template
        """
        # Build task prompt from template
        fields_str = ", ".join(template.keys())
        task_prompt = f"<s_docvqa><s_question>Extract: {fields_str}</s_question><s_answer>"

        # Extract
        result = self.extract_fields(image, task_prompt)

        # Validate against template
        validated_result = {}
        for field in template.keys():
            # Try exact match
            if field in result:
                validated_result[field] = result[field]
            else:
                # Try case-insensitive match
                for key, value in result.items():
                    if key.lower() == field.lower():
                        validated_result[field] = value
                        break

        return validated_result

    def document_vqa(
        self,
        image: Image.Image,
        question: str,
        max_length: int = 256
    ) -> str:
        """
        Visual Question Answering on document.

        Args:
            image: Document image
            question: Question to answer
            max_length: Maximum answer length

        Returns:
            Answer string
        """
        # Build VQA prompt
        task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"

        # Prepare inputs
        pixel_values = self.processor(
            image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        task_prompt_ids = self.processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=task_prompt_ids,
                max_length=max_length,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode
        answer = self.processor.batch_decode(outputs)[0]
        answer = answer.replace(task_prompt, "").strip()
        answer = answer.replace(self.processor.tokenizer.eos_token, "").strip()

        return answer

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
            Loaded DonutModel instance
        """
        instance = cls.__new__(cls)
        instance.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        instance.processor = DonutProcessor.from_pretrained(model_path)
        instance.model = VisionEncoderDecoderModel.from_pretrained(model_path)

        instance.model.to(instance.device)
        instance.model.eval()

        return instance


class DonutFineTuner:
    """Fine-tuning wrapper for Donut model."""

    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base",
        learning_rate: float = 3e-5,
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

        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        self.model.to(self.device)

        # Configure model for training
        self.model.config.max_length = 512
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 1.0
        self.model.config.num_beams = 1

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

    def prepare_training_batch(
        self,
        images: List[Image.Image],
        ground_truths: List[Dict],
        task_prompt: str = "<s_docvqa><s_question>Extract fields</s_question><s_answer>"
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for training.

        Args:
            images: List of document images
            ground_truths: List of ground truth dicts
            task_prompt: Task prompt template

        Returns:
            Batch dictionary
        """
        # Process images
        pixel_values = self.processor(
            images,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        # Prepare decoder inputs (task prompt + ground truth)
        decoder_input_ids = []
        labels = []

        for gt in ground_truths:
            # Convert ground truth to string format
            gt_str = json.dumps(gt, ensure_ascii=False)
            full_sequence = f"{task_prompt}{gt_str}</s>"

            # Tokenize
            tokens = self.processor.tokenizer(
                full_sequence,
                add_special_tokens=False,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            decoder_input_ids.append(tokens.input_ids)

            # Labels are the same as decoder inputs, but shifted
            label = tokens.input_ids.clone()
            label[label == self.processor.tokenizer.pad_token_id] = -100
            labels.append(label)

        decoder_input_ids = torch.cat(decoder_input_ids, dim=0).to(self.device)
        labels = torch.cat(labels, dim=0).to(self.device)

        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels
        }

    def train_step(
        self,
        images: List[Image.Image],
        ground_truths: List[Dict]
    ) -> float:
        """
        Single training step.

        Args:
            images: Batch of images
            ground_truths: Batch of ground truth dicts

        Returns:
            Loss value
        """
        self.model.train()

        # Prepare batch
        batch = self.prepare_training_batch(images, ground_truths)

        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss

        # Backward pass
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

        # Initialize model
        print("Loading Donut model...")
        model = DonutModel()

        # Extract fields
        print("Extracting fields...")
        result = model.extract_fields(image)

        print(f"\nExtracted fields:")
        for key, value in result.items():
            print(f"  {key}: {value}")

        # Example VQA
        print("\nDocument VQA example:")
        answer = model.document_vqa(image, "What is the document type?")
        print(f"  Q: What is the document type?")
        print(f"  A: {answer}")
    else:
        print("Usage: python donut_model.py <image_path>")
