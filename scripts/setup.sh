#!/bin/bash
# automated setup script for Stock Prediction ML project

echo "🚀 Starting Project Setup..."

# 1. Check Python version
python3 --version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Error: Python 3 not found. Please install Python 3.11+."
    exit 1
fi

# 2. Create virtual environment
if [ ! -d "myenv" ]; then
    echo "📦 Creating virtual environment (myenv)..."
    python3 -m venv myenv
else
    echo "✅ Virtual environment already exists."
fi

# 3. Install dependencies
echo "📥 Installing dependencies from requirements.txt..."
./myenv/bin/pip install --upgrade pip
./myenv/bin/pip install -r requirements.txt

# 4. Initialize .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔑 Initializing .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Action Required: Please edit .env with your Supabase and notification credentials."
else
    echo "✅ .env file already exists."
fi

# 5. Create necessary directories
echo "📁 Creating data and log directories..."
mkdir -p logs cache/stock_data output models

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo "Next Steps:"
echo "1. Edit .env file with your credentials."
echo "2. Activate environment: source myenv/bin/activate"
echo "3. Initialize Supabase: Execute supabase_schema.sql in Dashboard."
echo "4. Run application: python main.py"
echo "=========================================="
