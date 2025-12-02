#!/bin/bash
# Setup script for SOCAR Hackathon - Handwriting Data Processing

echo "🚀 SOCAR Hackathon 2025 - Setup Script"
echo "======================================="

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{raw,processed,annotations}
mkdir -p experiments/{checkpoints,logs}
mkdir -p demo/examples

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch experiments/checkpoints/.gitkeep
touch experiments/logs/.gitkeep

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install Detectron2
echo "🔍 Installing Detectron2..."
pip install 'git+https://github.com/facebookresearch/detectron2.git' || echo "⚠️  Detectron2 installation failed, will skip for now"

# Download sample data (if available)
echo "📥 Setting up sample data..."
# Add commands to download sample datasets if needed

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate  # On Linux/Mac"
echo "   venv\\Scripts\\activate     # On Windows"
echo ""
echo "2. Run the demo:"
echo "   python demo/app.py"
echo ""
echo "3. Or test individual models:"
echo "   python src/models/trocr_model.py path/to/image.jpg"
echo ""
echo "Happy hacking! 🏆"
