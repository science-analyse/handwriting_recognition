# Handwriting Recognition

Complete handwriting recognition system using CNN-BiLSTM-CTC on the IAM dataset.

## 📁 Files

### 1. **analysis.ipynb** - Dataset Analysis
- Exploratory Data Analysis (EDA)
- 5 detailed charts saved to `charts/` folder
- Run locally or on Colab (no GPU needed)

### 2. **train_colab.ipynb** - Model Training (GPU)
- **⚡ Google Colab GPU compatible**
- Full training pipeline
- CNN-BiLSTM-CTC model (~9.1M parameters)
- Automatic model saving
- Download trained model for deployment

## 🚀 Quick Start

### Option 1: Analyze Dataset (Local/Colab)
```bash
jupyter notebook analysis.ipynb
```
- No GPU needed
- Generates 5 EDA charts
- Fast (~2 minutes)

### Option 2: Train Model (Google Colab GPU)

1. **Upload `train_colab.ipynb` to Google Colab**
2. **Change runtime to GPU:**
   - Runtime → Change runtime type → GPU (T4 recommended)
3. **Run all cells**
4. **Download trained model** (last cell)

**Training Time:** ~1-2 hours for 20 epochs on T4 GPU

## 📊 Charts Generated

From `analysis.ipynb`:
1. `charts/01_sample_images.png` - 10 sample handwritten texts
2. `charts/02_text_length_distribution.png` - Text statistics
3. `charts/03_image_dimensions.png` - Image analysis
4. `charts/04_character_frequency.png` - Character distribution
5. `charts/05_summary_statistics.png` - Summary table

## 🎯 Model Details

**Architecture:**
- **CNN**: 7 convolutional blocks (feature extraction)
- **BiLSTM**: 2 layers, 256 hidden units (sequence modeling)
- **CTC Loss**: Alignment-free training

**Dataset:** Teklia/IAM-line (Hugging Face)
- Train: 6,482 samples
- Validation: 976 samples
- Test: 2,915 samples

**Metrics:**
- **CER** (Character Error Rate)
- **WER** (Word Error Rate)

## 💾 Model Files

After training in Colab:
- `best_model.pth` - Trained model weights
- `training_history.png` - Loss/CER/WER plots
- `predictions.png` - Sample predictions

## 📦 Requirements

```
torch>=2.0.0
datasets>=2.14.0
pillow>=9.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.13.0
jupyter>=1.0.0
jiwer>=3.0.0
```

## 🔧 Usage

### Load Trained Model
```python
import torch

# Load checkpoint
checkpoint = torch.load('best_model.pth')
char_mapper = checkpoint['char_mapper']

# Create model
from train_colab import CRNN  # Copy model class
model = CRNN(num_chars=len(char_mapper.chars))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
# ... (preprocessing + inference)
```

## 📝 Notes

- **GPU strongly recommended** for training (use Colab T4)
- Training on CPU will be extremely slow (~20x slower)
- Colab free tier: 12-hour limit, sufficient for 20 epochs
- Model checkpoint includes character mapper for deployment

## 🎓 Training Tips

1. **Start with fewer epochs** (5-10) to test
2. **Monitor CER/WER** - stop if not improving
3. **Increase epochs** if still improving (up to 50)
4. **Save checkpoint** before Colab disconnects
5. **Download model immediately** after training

## 📄 License

Dataset: IAM Database (research use)
