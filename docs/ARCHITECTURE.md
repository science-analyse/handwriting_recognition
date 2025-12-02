# 🏗️ Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Model Architectures](#model-architectures)
3. [Pipeline Flow](#pipeline-flow)
4. [Ensemble Strategy](#ensemble-strategy)
5. [Performance Optimization](#performance-optimization)

## System Overview

The handwriting data processing system employs a **hybrid multi-model ensemble** approach, combining three state-of-the-art models to achieve superior accuracy and robustness.

### Design Philosophy

1. **Redundancy**: Multiple models provide fail-safes
2. **Specialization**: Each model excels at different aspects
3. **Confidence-aware**: Weight results by model confidence
4. **Fallback**: Graceful degradation if models fail

### Key Components

```
┌─────────────────────────────────────────┐
│         Input Layer                      │
│  - Image Loading (PIL/OpenCV)           │
│  - Format Conversion (PDF→Image)        │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│      Preprocessing Layer                 │
│  - Deskewing                            │
│  - Denoising (FastNlMeans)              │
│  - Binarization (Sauvola)               │
│  - Layout Detection (Detectron2)        │
│  - Line Segmentation                    │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│        Model Inference Layer             │
│  ┌─────────────────────────────────┐   │
│  │  Branch 1: TrOCR                │   │
│  │  - Line-level OCR               │   │
│  │  - High accuracy on clean text  │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Branch 2: Donut                │   │
│  │  - OCR-free extraction          │   │
│  │  - Robust to noise              │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Branch 3: LayoutLMv3           │   │
│  │  - Multimodal understanding     │   │
│  │  - Context-aware extraction     │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│       Ensemble Layer                     │
│  - Result Alignment                     │
│  - Confidence Weighting                 │
│  - Conflict Resolution                  │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│     Post-processing Layer                │
│  - Spell Checking                       │
│  - Lexicon Matching                     │
│  - Confidence Filtering                 │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│         Output Layer                     │
│  - JSON Formatting                      │
│  - Visualization                        │
│  - Export (JSON/CSV/PDF)                │
└─────────────────────────────────────────┘
```

## Model Architectures

### 1. TrOCR (Transformer-based OCR)

**Paper**: [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282)

**Architecture**:
```
Input Image (384×384)
      ↓
Vision Encoder (ViT)
  - Patch embedding (16×16 patches)
  - 12 transformer layers
  - 768 hidden dimensions
      ↓
Text Decoder (RoBERTa)
  - 6 transformer layers
  - Autoregressive generation
  - Beam search decoding
      ↓
Output Text + Confidence
```

**Strengths**:
- ✅ High accuracy on clean handwriting (CER ~3-5%)
- ✅ Fast inference (~50ms per line)
- ✅ Pretrained on IAM dataset
- ✅ Character-level attention

**Limitations**:
- ❌ Requires line segmentation
- ❌ Sensitive to image quality
- ❌ No layout understanding

**Implementation Details**:
```python
# Model configuration
encoder: ViT-Base (DeiT)
decoder: RoBERTa-Base
vocab_size: 50,265
max_length: 128
beam_size: 5

# Training
optimizer: AdamW
lr: 5e-5
warmup_steps: 500
mixed_precision: FP16
```

### 2. Donut (Document Understanding Transformer)

**Paper**: [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664)

**Architecture**:
```
Input Image (1280×960)
      ↓
Vision Encoder (Swin Transformer)
  - Hierarchical features
  - Window attention
  - 4 stages
      ↓
Text Decoder (mBART)
  - Autoregressive generation
  - Task-specific prompts
  - JSON output
      ↓
Structured Fields
```

**Strengths**:
- ✅ No OCR preprocessing needed
- ✅ Direct field extraction
- ✅ Handles complex layouts
- ✅ Robust to image quality

**Limitations**:
- ❌ Large model (~200M params)
- ❌ Slower inference (~200ms)
- ❌ Requires more training data

**Implementation Details**:
```python
# Model configuration
encoder: Swin-Base
decoder: mBART
input_size: [1280, 960]
max_length: 512

# Task prompt format
task_prompt: "<s_docvqa><s_question>{question}</s_question><s_answer>"

# Training
optimizer: AdamW
lr: 3e-5
warmup_ratio: 0.1
gradient_accumulation: 16
```

### 3. LayoutLMv3 (Multimodal Document AI)

**Paper**: [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387)

**Architecture**:
```
Inputs:
  - Text tokens
  - Bounding boxes (normalized)
  - Visual features
      ↓
Multimodal Encoder
  - Text embedding
  - Layout embedding (2D position)
  - Image embedding (patch features)
  - Cross-modal attention
      ↓
Task Head
  - Token classification (NER)
  - Sequence classification
      ↓
Entity Labels + Confidence
```

**Strengths**:
- ✅ Best layout understanding
- ✅ Multimodal reasoning
- ✅ Context-aware extraction
- ✅ Strong on forms/tables

**Limitations**:
- ❌ Requires OCR input
- ❌ Complex preprocessing
- ❌ Medium inference speed

**Implementation Details**:
```python
# Model configuration
backbone: LayoutLMv3-Base
num_layers: 12
hidden_size: 768
num_attention_heads: 12

# Input format
text: List[str]  # OCR words
boxes: List[List[int]]  # [x0, y0, x1, y1] normalized to 1000
image: PIL.Image  # Original image

# Training
optimizer: AdamW
lr: 5e-5
task: Token Classification
num_labels: 9  # BIO scheme
```

## Pipeline Flow

### Detailed Processing Steps

#### 1. Preprocessing (50-100ms)

```python
def preprocess(image):
    # 1. Load and validate
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 2. Deskew (10-20ms)
    angle = detect_skew(image)
    image = rotate(image, -angle)

    # 3. Denoise (20-30ms)
    image = cv2.fastNlMeansDenoising(image, h=10)

    # 4. Enhance (10ms)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    image = clahe.apply(image)

    # 5. Binarize (10ms)
    binary = sauvola_threshold(image)

    # 6. Detect layout (10-20ms)
    regions = detect_text_regions(binary)
    lines = segment_lines(binary)

    return {
        'processed_image': binary,
        'regions': regions,
        'lines': lines
    }
```

#### 2. Model Inference (200-400ms)

**Parallel Execution**:
```python
from concurrent.futures import ThreadPoolExecutor

def parallel_inference(image, lines):
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all model tasks
        future_trocr = executor.submit(run_trocr, image, lines)
        future_donut = executor.submit(run_donut, image)
        future_layoutlm = executor.submit(run_layoutlm, image, lines)

        # Wait for results
        results = {
            'trocr': future_trocr.result(),
            'donut': future_donut.result(),
            'layoutlm': future_layoutlm.result()
        }

    return results
```

#### 3. Ensemble (10-20ms)

**Weighted Voting Algorithm**:
```python
def weighted_ensemble(results, weights):
    """
    Combine results using weighted voting.

    Score(field, value) = Σ weight_i × confidence_i × match_i
    """
    field_scores = defaultdict(lambda: defaultdict(float))

    for model_name, result in results.items():
        w = weights[model_name]
        c = result.confidence

        for field, value in result.fields.items():
            field_scores[field][value] += w * c

    # Select highest scoring value for each field
    final_fields = {}
    for field, value_scores in field_scores.items():
        final_fields[field] = max(
            value_scores.items(),
            key=lambda x: x[1]
        )[0]

    return final_fields
```

#### 4. Post-processing (10-20ms)

```python
def postprocess(fields, lexicon=None):
    """Apply corrections and validation."""

    # 1. Spell check
    for key, value in fields.items():
        if is_text_field(key):
            fields[key] = spell_correct(value)

    # 2. Lexicon matching
    if lexicon:
        for key, value in fields.items():
            if key in lexicon:
                fields[key] = fuzzy_match(value, lexicon[key])

    # 3. Format validation
    fields = validate_formats(fields)  # dates, numbers, etc.

    # 4. Confidence filtering
    fields = filter_low_confidence(fields, threshold=0.5)

    return fields
```

## Ensemble Strategy

### Comparison of Strategies

| Strategy | Description | Pros | Cons | Use Case |
|----------|-------------|------|------|----------|
| **Voting** | Majority vote | Simple, robust | Ignores confidence | Equal model quality |
| **Weighted** | Confidence-weighted | Optimal accuracy | Complex tuning | Default choice |
| **Cascaded** | Fallback chain | Fast when best works | Misses consensus | High-confidence primary |

### Weighted Ensemble Implementation

```python
class WeightedEnsemble:
    def __init__(self, weights):
        self.weights = weights

    def combine(self, results):
        """
        Weighted combination with spatial alignment.
        """
        # Step 1: Normalize confidences
        normalized_results = self._normalize_confidences(results)

        # Step 2: Align fields spatially
        aligned_fields = self._spatial_alignment(normalized_results)

        # Step 3: Weighted voting
        final_fields = {}
        for field_name in aligned_fields:
            candidates = aligned_fields[field_name]

            # Calculate weighted scores
            scores = {}
            for model, (value, conf) in candidates.items():
                score = self.weights[model] * conf
                scores[value] = scores.get(value, 0) + score

            # Select highest score
            final_fields[field_name] = max(
                scores.items(),
                key=lambda x: x[1]
            )[0]

        return final_fields
```

## Performance Optimization

### 1. Model Optimization

**Quantization** (FP32 → INT8):
```python
# Reduces model size by 4x, speeds up inference by 2-3x
from torch.quantization import quantize_dynamic

model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

**ONNX Export**:
```python
# Convert to ONNX for faster inference
import torch.onnx

dummy_input = torch.randn(1, 3, 384, 384)
torch.onnx.export(
    model,
    dummy_input,
    "trocr_optimized.onnx",
    opset_version=14
)
```

### 2. Batching

```python
def batch_process(images, batch_size=8):
    """Process multiple images in batches."""
    results = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]

        # Batch inference
        batch_results = model.recognize_batch(batch)
        results.extend(batch_results)

    return results
```

### 3. Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_inference(image_hash, model_name):
    """Cache inference results."""
    return model.predict(image_hash)

def process_with_cache(image):
    # Hash image
    image_hash = hashlib.md5(image.tobytes()).hexdigest()

    # Check cache
    return cached_inference(image_hash, 'trocr')
```

### 4. GPU Optimization

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Scalability Considerations

### Horizontal Scaling

```python
# Distribute across multiple GPUs
model = torch.nn.DataParallel(model)

# Or use distributed training
import torch.distributed as dist
model = torch.nn.parallel.DistributedDataParallel(model)
```

### Microservices Architecture

```
┌─────────────┐
│  API Gateway│
└──────┬──────┘
       │
   ┌───┴────┬────────┬────────┐
   ▼        ▼        ▼        ▼
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│TrOCR│  │Donut│  │LayoutLM│  │Post│
│Service││Service││Service│  │Proc│
└─────┘  └─────┘  └─────┘  └─────┘
```

---

## References

1. Microsoft Research - [TrOCR Paper](https://arxiv.org/abs/2109.10282)
2. Naver Clova - [Donut Paper](https://arxiv.org/abs/2111.15664)
3. Microsoft - [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)

**Last Updated**: December 2025
