# 🎮 DOTA - Dynamic Onscreen Tactical Assistant

![Dota 2](https://img.shields.io/badge/Dota%202-FF6B35?style=for-the-badge&logo=steampowered&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![AI/ML](https://img.shields.io/badge/AI%2FML-PyTorch%20%7C%20Transformers-red?style=for-the-badge&logo=pytorch&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-ChromaDB%20%7C%20LangChain-green?style=for-the-badge)

A powerful AI-powered system that combines **Vision Language Models (VLM)** and **Retrieval Augmented Generation (RAG)** to provide intelligent Dota 2 character selection and tactical recommendations.

## 🌟 Features

- **🎯 Intelligent Hero Selection**: Get AI-powered recommendations based on team composition and game state
- **📊 Advanced RAG System**: Leverages comprehensive Dota 2 knowledge base for tactical insights
- **👁️ Vision Language Processing**: Analyzes game screenshots and provides visual feedback
- **🧠 Context-Aware Tactics**: Real-time strategy recommendations based on current match situation
- **📈 Performance Analytics**: Track and analyze hero performance metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/minashkinvladislav/DOTA.git
   cd DOTA
   ```

2. **Install uv** (if not already installed)
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Set up the project**
   ```bash
   # Sync dependencies and create virtual environment
   uv sync

   # Activate the virtual environment
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Or use uv run to run commands directly
   uv run python your_script.py
   ```

4. **Verify installation**
   ```bash
   uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

## 📁 Project Structure

```
DOTA/
├── 📄 pyproject.toml          # Project configuration and dependencies
├── 📄 uv.lock                 # Locked dependency versions
├── 📄 .python-version         # Python version specification
├── 📄 .gitignore              # Git ignore rules
├── 📄 LICENSE                 # MIT License
├── 📄 README.md               # This file
├── 📁 .venv/                  # Virtual environment (auto-generated)
└── 📁 RAG/                    # Main application directory
    ├── 📄 VLM+RAG.ipynb       # 📓 Main Jupyter notebook with VLM + RAG workflow
    ├── 📄 vlm_functions.py    # 🔧 Vision Language Model utility functions
    ├── 📄 dota_heroes.txt     # 👥 List of Dota 2 heroes
    ├── 📄 hero_embeddings.pkl # 🧠 Pre-computed hero embeddings
    ├── 📁 dota_db_1.3/        # 🗄️ ChromaDB knowledge base
    ├── 📁 dota_hero_icons/    # 🖼️ Hero icon assets
    ├── 📁 dota_wget/          # 📥 Downloaded data files
    └── 📁 test_pic/           # 🧪 Test images for VLM processing
```

## 🛠️ Key Components

### Vision Language Model (VLM)
The VLM system processes game screenshots and visual information to understand:
- Current game state
- Hero positions and status
- Team compositions
- Map objectives and terrain

### Retrieval Augmented Generation (RAG)
Our RAG system leverages:
- **ChromaDB** for vector storage and similarity search
- **LangChain** for document processing and retrieval
- **Comprehensive Dota 2 knowledge base** with hero stats, abilities, and strategies

### Hero Analysis Engine
- **130+ Heroes supported** with complete ability descriptions
- **Embedding-based similarity matching** for hero recommendations
- **Real-time performance analysis** and meta insights

## 📚 Key Files Explained

| File | Purpose | Usage |
|------|---------|-------|
| `VLM+RAG.ipynb` | Main application workflow | Run this notebook to start the system |
| `vlm_functions.py` | VLM utility functions | Import functions for custom implementations |
| `pyproject.toml` | Project configuration | Manage dependencies and project settings |
| `hero_embeddings.pkl` | Pre-computed embeddings | Accelerate hero similarity searches |
| `dota_heroes.txt` | Hero database | Reference for all supported heroes |

## 🎮 Usage Examples

### Basic Hero Recommendation
```python
from RAG.vlm_functions import get_hero_recommendation

# Get hero recommendations based on team composition
recommendations = get_hero_recommendation(
    allied_heroes=["Pudge", "Mirana", "Lina"],
    enemy_heroes=["Anti-Mage", "Sven", "Crystal Maiden"],
    position="carry"
)
print(recommendations)
```

### VLM Image Analysis
```python
from RAG.vlm_functions import analyze_game_screenshot
import PIL.Image as Image

# Analyze a game screenshot
screenshot = Image.open("path/to/screenshot.png")
analysis = analyze_game_screenshot(screenshot)
print(f"Game state: {analysis}")
```

### RAG Knowledge Query
```python
from RAG.vlm_functions import query_dota_knowledge

# Query specific Dota 2 knowledge
answer = query_dota_knowledge("How to counter Phantom Lancer as a support?")
print(answer)
```

## 🔧 Development

### Running the Main Notebook
```bash
# Start Jupyter Lab
uv run jupyter lab

# Navigate to RAG/VLM+RAG.ipynb and run all cells
```

### Adding New Heroes
1. Update `RAG/dota_heroes.txt` with new hero data
2. Run embedding generation: `uv run python generate_embeddings.py`
3. Update the ChromaDB collection with new hero information

### Custom Model Integration
Modify `RAG/vlm_functions.py` to integrate different models:
```python
# Example: Switch to a different vision model
from transformers import Blip2ForConditionalGeneration

def initialize_vlm():
    return Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
```

## 📊 Performance

- **Hero Processing**: ~130 heroes with full metadata
- **Embedding Dimension**: 768-dimensional vectors
- **Query Response Time**: <100ms for most queries
- **Memory Usage**: ~2GB for full knowledge base

## 🎯 Future Enhancements

- [ ] **Real-time API Integration**: Direct Dota 2 client integration
- [ ] **Advanced Analytics**: Player performance tracking
- [ ] **Multi-language Support**: Localized hero names and descriptions
- [ ] **Web Interface**: Browser-based UI for easy access
- [ ] **Mobile App**: Companion app for mobile strategy planning

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Setup
```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .
```

## 📜 Dependencies

### Core ML Libraries
- **PyTorch 2.9.0+** - Deep learning framework
- **Transformers 4.57.1+** - Hugging Face transformers
- **Sentence Transformers 5.1.2+** - Sentence embeddings

### RAG & Vector Databases
- **ChromaDB 1.3.4+** - Vector database
- **LangChain 1.0.3+** - LLM application framework
- **LangChain-Chroma 1.0.0+** - LangChain integration

### Web Scraping & Text Processing
- **BeautifulSoup4 4.14.2+** - HTML parsing
- **LXML 6.0.2+** - XML/HTML processing
- **NLTK 3.9.2+** - Natural language toolkit

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Valve Corporation** for creating Dota 2
- **Hugging Face** for the excellent transformers library
- **ChromaDB** team for the powerful vector database
- **LangChain** community for the RAG framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: minashkinvladislav@gmail.com

---

<div align="center">

**Built with ❤️ for the Dota 2 community**

[⭐ Star this repo if you find it useful! ⭐](https://github.com/your-repo)

</div>
