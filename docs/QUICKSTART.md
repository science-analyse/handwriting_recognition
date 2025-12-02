# ⚡ Quick Start Guide

Get started with the handwriting OCR system in 5 minutes!

## 🎯 For the Impatient

```bash
# 1. Clone and setup
git clone <your-repo>
cd handwriting_data_processing
chmod +x setup.sh
./setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Run demo
python demo/app.py

# 4. Open browser
# Navigate to http://localhost:7860
```

## 📝 Step-by-Step

### 1. Prerequisites Check

```bash
# Check Python version (need 3.8+)
python --version

# Check CUDA (optional but recommended)
nvidia-smi

# Check available disk space (need ~10GB for models)
df -h
```

### 2. Installation

```bash
# Option A: Using setup script (recommended)
./setup.sh

# Option B: Manual installation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Quick Test

Test each component individually:

```bash
# Test preprocessing
python -c "
from src.preprocessing.image_processor import ImagePreprocessor
import cv2
preprocessor = ImagePreprocessor()
image = cv2.imread('test_image.jpg')
processed = preprocessor.process(image)
print('✅ Preprocessing works!')
"

# Test TrOCR
python -c "
from src.models.trocr_model import TrOCRModel
from PIL import Image
model = TrOCRModel()
print('✅ TrOCR loaded successfully!')
"
```

### 4. Process Your First Document

```python
from PIL import Image
from src.models.ensemble import HybridOCRPipeline

# Initialize pipeline (this will download models - first time only)
pipeline = HybridOCRPipeline(
    use_trocr=True,
    use_donut=True,
    use_layoutlm=False  # Disable for faster startup
)

# Load and process
image = Image.open("your_document.jpg")
result = pipeline.process_document(image)

# Print results
print(f"Confidence: {result.confidence:.2%}")
print(f"Extracted fields: {result.fields}")
```

## 🎨 Demo Interface

Launch the interactive demo:

```bash
python demo/app.py
```

Features:
- 📤 Upload handwritten documents
- ⚙️ Configure models (TrOCR, Donut, LayoutLMv3)
- 📊 See results in real-time
- 💾 Download results as JSON

## 🐛 Common Issues

### Issue: CUDA Out of Memory

**Solution 1**: Use CPU only
```python
pipeline = HybridOCRPipeline(device='cpu')
```

**Solution 2**: Use fewer models
```python
pipeline = HybridOCRPipeline(
    use_trocr=True,
    use_donut=False,
    use_layoutlm=False
)
```

### Issue: Models downloading slowly

**Solution**: Models are cached in `~/.cache/huggingface/`. First run downloads ~2GB of models.

```bash
# Check download progress
ls -lh ~/.cache/huggingface/hub/
```

### Issue: Import errors

**Solution**: Make sure you're in the right directory and virtual environment is activated

```bash
# Check current directory
pwd  # Should end in /handwriting_data_processing

# Check virtual environment
which python  # Should point to venv/bin/python

# Reinstall if needed
pip install -r requirements.txt --force-reinstall
```

## 🚀 Next Steps

1. **Fine-tune on your data**: See `docs/training.md`
2. **Optimize for production**: See `docs/deployment.md`
3. **Customize models**: Edit `configs/model_config.yaml`
4. **Add custom preprocessing**: Modify `src/preprocessing/image_processor.py`

## 📚 Resources

- Full documentation: `README.md`
- Model details: `docs/architecture.md`
- Training guide: `docs/training.md`
- API reference: `docs/api.md`

## 💬 Getting Help

During the hackathon:
- Check documentation first
- Ask mentors
- Debug with verbose logging: `export LOG_LEVEL=DEBUG`

## ✅ Verification Checklist

Before submission, verify:

- [ ] Demo launches without errors
- [ ] Can process test images
- [ ] Results are reasonable (>80% confidence)
- [ ] All team members can run the code
- [ ] README is updated with team info
- [ ] Code is committed to git

---

**Time to first result: ~5 minutes** ⏱️

Good luck! 🏆
