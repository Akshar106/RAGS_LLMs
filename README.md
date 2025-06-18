# Document Chat & Meeting Notes Summarizer

A powerful Streamlit application that combines two essential AI-powered features:
1. **RAG-based Document Chat** - Upload PDF documents and chat with them using OpenAI's GPT models
2. **Meeting Notes Summarizer** - Automatically summarize meeting notes into concise bullet points using T5 transformer

## 🚀 Features

### 📄 Chat with Document using RAG
- Upload PDF documents and ask questions about their content
- Uses Retrieval Augmented Generation (RAG) with FAISS vector database
- Powered by OpenAI's GPT-3.5-turbo model
- Maintains conversation history for context-aware responses
- Efficient document chunking and embedding for better retrieval

### 📝 Meeting Notes Summarizer
- Transform lengthy meeting notes into structured bullet points
- Uses Google's T5-small transformer model
- Local processing - no external API calls required
- Automatic text preprocessing and optimization

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-name>
```

### Install Required Dependencies
```bash
pip install streamlit
pip install transformers
pip install langchain
pip install langchain-community
pip install faiss-cpu
pip install openai
pip install python-dotenv
pip install PyPDF2
pip install pypdf
pip install torch
pip install numpy
pip install pillow
```

Or install all dependencies at once:
```bash
pip install streamlit transformers langchain langchain-community faiss-cpu openai python-dotenv PyPDF2 pypdf torch numpy pillow
```

## ⚙️ Configuration

### Environment Variables Setup
Create a `.env` file in the root directory of your project:

```env
OPENAI_API_KEY=your_openai_api_key_here
API_BASE_URL=https://api.openai.com/v1
visionchat_api=your_visionchat_api_key_here
```

### Get OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and add it to your `.env` file

## 🚦 Usage

### Running the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using Chat with Document
1. Select "Chat with Document using RAG" from the sidebar
2. Upload a PDF document using the file uploader
3. Wait for the document to be processed
4. Enter your questions in the text input field
5. Click "Get Answer" to receive AI-powered responses

### Using Meeting Notes Summarizer
1. Select "Meeting Notes Summarizer" from the sidebar
2. Paste your meeting notes in the text area
3. Click "Summarize Meeting Notes"
4. View the organized bullet-point summary

## 📁 Project Structure

```
project/
│
├── app.py                 # Main Streamlit application
├── .env                   # Environment variables (create this)
├── README.md             # This file
├── requirements.txt      # Dependencies list
└── uploaded_document.pdf # Temporary file (auto-generated)
```

## 🔧 Technical Details

### Document Chat Architecture
- **Document Loading**: PyPDFLoader for PDF processing
- **Text Splitting**: RecursiveCharacterTextSplitter (500 char chunks, 50 overlap)
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Store**: FAISS for similarity search
- **LLM**: OpenAI GPT-3.5-turbo
- **Memory**: ConversationBufferMemory for chat history

### Summarizer Architecture
- **Model**: Google T5-small transformer
- **Tokenizer**: T5Tokenizer
- **Processing**: Automatic text preprocessing with "summarize:" prefix
- **Generation**: Beam search with length penalties for quality summaries

## 🐛 Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# If you get import errors, try:
pip install --upgrade langchain langchain-community
```

**2. OpenAI API Errors**
- Verify your API key is correct and active
- Check your OpenAI account has sufficient credits
- Ensure the API key has proper permissions

**3. FAISS Installation Issues**
```bash
# On some systems, you might need:
pip install faiss-gpu  # For GPU support
# or
conda install -c pytorch faiss-cpu  # Using conda
```

**4. PyTorch Issues**
```bash
# Install PyTorch based on your system:
# Visit: https://pytorch.org/get-started/locally/
```

### Performance Tips
- For large documents, consider increasing chunk size
- Adjust the number of retrieved chunks (k parameter) based on your needs
- Use GPU acceleration for faster T5 processing if available

## 📋 Requirements

### Python Packages
```
streamlit>=1.28.0
transformers>=4.21.0
langchain>=0.1.0
langchain-community>=0.0.1
faiss-cpu>=1.7.4
openai>=1.0.0
python-dotenv>=1.0.0
PyPDF2>=1.26.0
pypdf>=3.0.0
torch>=1.13.0
numpy>=1.21.0
pillow>=8.0.0
```

### System Requirements
- RAM: Minimum 4GB (8GB recommended for large documents)
- Storage: 2GB free space for model downloads
- Internet: Required for OpenAI API calls and initial model downloads

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 🙏 Acknowledgments

- OpenAI for GPT models and embeddings
- Google for T5 transformer model
- Hugging Face for the transformers library
- LangChain for RAG implementation
- Streamlit for the web interface

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Create an issue in the repository
3. Review the documentation of individual packages

---
