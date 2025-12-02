"""
Ensemble pipeline combining TrOCR, Donut, and LayoutLMv3 for robust handwriting recognition.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch
from dataclasses import dataclass

from .trocr_model import TrOCRModel
from .donut_model import DonutModel
from .layoutlm_model import LayoutLMv3Model, normalize_box
from ..preprocessing.image_processor import ImagePreprocessor, LayoutDetector


@dataclass
class ExtractionResult:
    """Result from document extraction."""
    fields: Dict[str, str]
    confidence: float
    method: str
    raw_text: Optional[str] = None
    entities: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None


class HybridOCRPipeline:
    """
    Hybrid OCR pipeline combining multiple approaches:
    1. TrOCR for line-level OCR
    2. Donut for OCR-free document understanding
    3. LayoutLMv3 for multimodal entity extraction
    """

    def __init__(
        self,
        use_trocr: bool = True,
        use_donut: bool = True,
        use_layoutlm: bool = True,
        ensemble_strategy: str = "voting",
        device: Optional[str] = None
    ):
        """
        Initialize hybrid pipeline.

        Args:
            use_trocr: Enable TrOCR
            use_donut: Enable Donut
            use_layoutlm: Enable LayoutLMv3
            ensemble_strategy: 'voting', 'weighted', or 'cascaded'
            device: Device to use
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.ensemble_strategy = ensemble_strategy

        # Initialize preprocessors
        self.image_preprocessor = ImagePreprocessor()
        self.layout_detector = LayoutDetector()

        # Initialize models
        self.models = {}

        if use_trocr:
            print("Loading TrOCR...")
            self.models['trocr'] = TrOCRModel(device=self.device)

        if use_donut:
            print("Loading Donut...")
            self.models['donut'] = DonutModel(device=self.device)

        if use_layoutlm:
            print("Loading LayoutLMv3...")
            self.models['layoutlm'] = LayoutLMv3Model(device=self.device)

        # Ensemble weights
        self.weights = {
            'trocr': 0.3,
            'donut': 0.4,
            'layoutlm': 0.3
        }

    def process_document(
        self,
        image: Image.Image,
        template: Optional[Dict[str, str]] = None
    ) -> ExtractionResult:
        """
        Process document with full pipeline.

        Args:
            image: Document image
            template: Optional field template for extraction

        Returns:
            ExtractionResult with extracted fields
        """
        # Preprocess image
        image_np = np.array(image)
        processed_image = self.image_preprocessor.process(image_np)

        # Detect layout
        regions = self.layout_detector.detect_text_regions(processed_image)
        lines = self.layout_detector.segment_lines(processed_image)

        # Extract with each available model
        results = []

        # 1. TrOCR approach (line-by-line)
        if 'trocr' in self.models:
            trocr_result = self._extract_with_trocr(image, lines)
            results.append(trocr_result)

        # 2. Donut approach (OCR-free)
        if 'donut' in self.models:
            donut_result = self._extract_with_donut(image, template)
            results.append(donut_result)

        # 3. LayoutLMv3 approach (multimodal)
        if 'layoutlm' in self.models and 'trocr' in self.models:
            layoutlm_result = self._extract_with_layoutlm(image, lines)
            results.append(layoutlm_result)

        # Ensemble results
        final_result = self._ensemble_results(results)

        return final_result

    def _extract_with_trocr(
        self,
        image: Image.Image,
        lines: List[Tuple[int, int, int, int]]
    ) -> ExtractionResult:
        """Extract using TrOCR line-by-line."""
        trocr_model = self.models['trocr']

        # Extract text from each line
        all_text = []
        total_confidence = 0.0

        for x, y, w, h in lines:
            # Crop line
            line_image = image.crop((x, y, x + w, y + h))

            # Recognize
            text, confidence = trocr_model.recognize_with_confidence(line_image)
            all_text.append(text)
            total_confidence += confidence

        # Join text
        raw_text = "\n".join(all_text)
        avg_confidence = total_confidence / max(len(lines), 1)

        # Parse fields from text (simple heuristic)
        fields = self._parse_fields_from_text(raw_text)

        return ExtractionResult(
            fields=fields,
            confidence=avg_confidence,
            method="trocr",
            raw_text=raw_text
        )

    def _extract_with_donut(
        self,
        image: Image.Image,
        template: Optional[Dict[str, str]] = None
    ) -> ExtractionResult:
        """Extract using Donut OCR-free approach."""
        donut_model = self.models['donut']

        if template:
            fields = donut_model.extract_with_template(image, template)
        else:
            fields = donut_model.extract_fields(image)

        # Estimate confidence (Donut doesn't provide built-in confidence)
        # Use field coverage as proxy
        confidence = min(len(fields) / 5.0, 1.0)  # Assume 5 fields is full coverage

        return ExtractionResult(
            fields=fields,
            confidence=confidence,
            method="donut"
        )

    def _extract_with_layoutlm(
        self,
        image: Image.Image,
        lines: List[Tuple[int, int, int, int]]
    ) -> ExtractionResult:
        """Extract using LayoutLMv3 with TrOCR for OCR."""
        trocr_model = self.models['trocr']
        layoutlm_model = self.models['layoutlm']

        # Get OCR results with bounding boxes
        words = []
        boxes = []
        width, height = image.size

        for x, y, w, h in lines:
            line_image = image.crop((x, y, x + w, y + h))
            text, _ = trocr_model.recognize_with_confidence(line_image)

            # Split into words (simple whitespace split)
            line_words = text.split()

            # Approximate word boxes (evenly distributed along line)
            word_width = w / max(len(line_words), 1)

            for i, word in enumerate(line_words):
                words.append(word)
                word_box = [
                    int(x + i * word_width),
                    y,
                    int(x + (i + 1) * word_width),
                    y + h
                ]
                # Normalize box
                normalized_box = normalize_box(word_box, width, height)
                boxes.append(normalized_box)

        # Extract entities
        entities = layoutlm_model.extract_entities(image, words, boxes)

        # Convert entities to fields
        fields = {}
        total_confidence = 0.0

        for entity in entities:
            label = entity['label']
            text = entity['text']
            confidence = entity['confidence']

            # Aggregate entities by label
            if label in fields:
                fields[label] += " " + text
            else:
                fields[label] = text

            total_confidence += confidence

        avg_confidence = total_confidence / max(len(entities), 1)

        return ExtractionResult(
            fields=fields,
            confidence=avg_confidence,
            method="layoutlm",
            entities=entities
        )

    def _parse_fields_from_text(self, text: str) -> Dict[str, str]:
        """
        Parse structured fields from raw text.
        Uses simple heuristics - can be improved with regex or NER.
        """
        fields = {}
        lines = text.strip().split('\n')

        for line in lines:
            # Look for key-value patterns
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    fields[key] = value

        return fields

    def _ensemble_results(
        self,
        results: List[ExtractionResult]
    ) -> ExtractionResult:
        """
        Combine results from multiple models.

        Args:
            results: List of extraction results

        Returns:
            Combined extraction result
        """
        if len(results) == 0:
            return ExtractionResult(fields={}, confidence=0.0, method="none")

        if len(results) == 1:
            return results[0]

        if self.ensemble_strategy == "voting":
            return self._voting_ensemble(results)
        elif self.ensemble_strategy == "weighted":
            return self._weighted_ensemble(results)
        elif self.ensemble_strategy == "cascaded":
            return self._cascaded_ensemble(results)
        else:
            # Default: highest confidence
            return max(results, key=lambda r: r.confidence)

    def _voting_ensemble(
        self,
        results: List[ExtractionResult]
    ) -> ExtractionResult:
        """Majority voting ensemble."""
        # Collect all fields
        all_fields = {}
        field_votes = {}

        for result in results:
            for key, value in result.fields.items():
                if key not in field_votes:
                    field_votes[key] = {}

                if value not in field_votes[key]:
                    field_votes[key][value] = 0

                field_votes[key][value] += 1

        # Select most voted value for each field
        for key, votes in field_votes.items():
            all_fields[key] = max(votes.items(), key=lambda x: x[1])[0]

        # Average confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)

        return ExtractionResult(
            fields=all_fields,
            confidence=avg_confidence,
            method="voting_ensemble"
        )

    def _weighted_ensemble(
        self,
        results: List[ExtractionResult]
    ) -> ExtractionResult:
        """Weighted ensemble based on model weights."""
        all_fields = {}
        field_scores = {}

        for result in results:
            weight = self.weights.get(result.method, 1.0)
            score = weight * result.confidence

            for key, value in result.fields.items():
                field_key = (key, value)

                if field_key not in field_scores:
                    field_scores[field_key] = 0

                field_scores[field_key] += score

        # Select highest scoring value for each field
        fields_by_key = {}
        for (key, value), score in field_scores.items():
            if key not in fields_by_key or score > fields_by_key[key][1]:
                fields_by_key[key] = (value, score)

        all_fields = {k: v[0] for k, v in fields_by_key.items()}

        # Weighted average confidence
        total_weight = sum(
            self.weights.get(r.method, 1.0) * r.confidence
            for r in results
        )
        total_weight_sum = sum(self.weights.get(r.method, 1.0) for r in results)
        avg_confidence = total_weight / total_weight_sum if total_weight_sum > 0 else 0

        return ExtractionResult(
            fields=all_fields,
            confidence=avg_confidence,
            method="weighted_ensemble"
        )

    def _cascaded_ensemble(
        self,
        results: List[ExtractionResult]
    ) -> ExtractionResult:
        """Cascaded ensemble - use higher confidence model, fall back to others."""
        # Sort by confidence
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

        # Start with highest confidence
        final_fields = sorted_results[0].fields.copy()

        # Fill in missing fields from other models
        for result in sorted_results[1:]:
            for key, value in result.fields.items():
                if key not in final_fields:
                    final_fields[key] = value

        return ExtractionResult(
            fields=final_fields,
            confidence=sorted_results[0].confidence,
            method="cascaded_ensemble"
        )


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Initialize pipeline
        print("Initializing hybrid OCR pipeline...")
        pipeline = HybridOCRPipeline(
            use_trocr=True,
            use_donut=True,
            use_layoutlm=False,  # Disable for faster demo
            ensemble_strategy="weighted"
        )

        # Process document
        print("Processing document...")
        result = pipeline.process_document(image)

        print(f"\nExtraction Result:")
        print(f"Method: {result.method}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"\nExtracted Fields:")
        for key, value in result.fields.items():
            print(f"  {key}: {value}")

        if result.raw_text:
            print(f"\nRaw Text:\n{result.raw_text}")
    else:
        print("Usage: python ensemble.py <image_path>")
