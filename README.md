# GPT-OSS Web Search & Chat Interface

A powerful Python implementation that combines OpenAI's GPT-OSS 20B model with real-time web search capabilities using You.com API. This project enables intelligent web browsing, information retrieval, and document analysis with a Shrek-themed creative chat interface.

## 🌟 Features

- **Real-time Web Search**: Integrated You.com API for up-to-date information retrieval
- **GPT-OSS 20B Model**: Utilizes OpenAI's open-source 20B parameter model
- **Smart Tool Usage**: Automatic web browsing and source verification
- **Document Analysis**: PDF processing capabilities for CV evaluation and document review
- **Movie Recommendation Engine**: Letterboxd integration with intelligent film selection
- **Flexible Chat System**: Customizable system prompts for various tasks
- **Async Processing**: Non-blocking web searches with proper async/await implementation

## 🛠️ Installation

### Prerequisites

```bash
# Install required dependencies
pip install git+https://github.com/huggingface/transformers triton==3.4 kernels
pip install gpt-oss openai_harmony
pip install pdfplumber pandas nest_asyncio
```

### Environment Setup

Set your You.com API key:
```python
os.environ["YDC_API_KEY"] = "your-ydc-api-key-here"
```

## 🚀 Usage

### 1. Model Initialization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda",
)
```

### 2. Web Search Function

The main `web_search()` function performs intelligent web searches:

```python
result = await web_search("What's the weather in Istanbul right now?", model)
print(result)
```

**Function Features:**
- Performs You.com web search
- Opens and analyzes reliable sources
- Provides summarized, up-to-date information
- Handles tool calls and browser interactions automatically

### 3. Intelligent Chat with Custom System Prompts

The `answer()` function provides flexible response generation with customizable system prompts:

```python
def answer(question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Choose a movie for movie night. Do NOT select any movie listed in the watched movies. Exclude them strictly. Reasoning: High."},
        {"role": "user", "content": f"{question}"},
    ]
    # Process and return clean response
```

**Key Features:**
- Customizable system content for different tasks
- Regex-based response parsing for clean output
- High reasoning effort for detailed analysis
- Token-efficient processing

### 4. Document Analysis

Process PDF documents for analysis:

```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"

evaluation = answer(f"Evaluate this CV: {text}")
```

## 📋 Core Functions

### `web_search(question, model)`
- **Purpose**: Performs intelligent web searches with source verification
- **Parameters**: 
  - `question`: The query to search for
  - `model`: The GPT-OSS model instance
- **Returns**: Processed and summarized web search results
- **Features**: Two-stage inference process for optimal accuracy

### `answer(question)`
- **Purpose**: Generates intelligent responses with customizable system prompts
- **Parameters**: `question` - The input query
- **Returns**: Processed response based on system configuration
- **Special**: Uses high reasoning effort and regex parsing for clean output
- **Customizable**: System content can be modified for different use cases

## 🎯 Example Use Cases

The code demonstrates several practical applications:

1. **Real-time Information**: Weather updates, sports scores, current events
2. **Gaming Data**: EA FC player ratings, transfer news
3. **Entertainment News**: Celebrity updates, show information
4. **Document Review**: CV evaluation, document analysis
5. **Movie Recommendations**: Intelligent film selection based on viewing history
6. **Batch Processing**: Multiple question processing from Excel/CSV files


### Intelligent Movie Selection
```python
prompt = f"""
You are a helpful assistant tasked with choosing a movie for a movie night.
Do NOT select any movie that is already in the watched movies list below.
Available movies to choose from:
- Il Treno dei Bambini (2024)
- Cast Away (Tom Hanks)
- Inception (Christopher Nolan)
- I Want You Back (2022)
- Drive (Ryan Gosling)
- Whiplash (JK Simmons)
- Instant Family (Mark Wahlberg)
- Memento

Movies already watched: {watched_movies}
Choose the best movie considering the taste of two people described below.
Only return the movie title, nothing else.
Person 1: {df}
Person 2: {df1}
"""

recommendation = answer(prompt)
```

**Features:**
- Multi-person taste analysis
- Watched movie exclusion
- Genre-based recommendations
- Batch movie selection (top 5 recommendations)

The project includes Excel file processing capabilities:

```python
import pandas as pd
df = pd.read_excel("search_questions.xlsx")
questions = df['question'].tolist()

# Process multiple questions
responses = []
for question in questions:
    response = answer(question)
    responses.append(response)
```

## ⚙️ Technical Architecture

### Two-Stage Inference Process

1. **Stage 1**: Web search and tool usage enabled
   - Performs You.com search
   - Opens reliable sources
   - Extracts relevant information

2. **Stage 2**: Final answer generation without tools
   - Processes collected information
   - Generates comprehensive response
   - Ensures no additional tool calls

### Async Implementation

Uses `nest_asyncio` for proper async handling in notebook environments:

```python
import nest_asyncio
nest_asyncio.apply()

result = asyncio.run(web_search("Your question here", model))
```

## 🔧 Configuration Options

- **Reasoning Effort**: Set to `ReasoningEffort.HIGH` for detailed analysis
- **Max Tokens**: Configurable output length (1000-3000 tokens)
- **Device Mapping**: CUDA support for GPU acceleration
- **Tool Integration**: Browser tool with You.com backend

## 🚨 Important Notes

- Requires CUDA-compatible GPU for optimal performance
- You.com API key needed for web search functionality
- Model downloads ~40GB (GPT-OSS 20B parameters)
- Async functions must be run with `asyncio.run()` or in async context

## 📈 Performance Tips

1. Use GPU acceleration for faster inference
2. Batch process multiple questions for efficiency
3. Cache frequently accessed model components
4. Monitor API rate limits for You.com searches

## 🤝 Contributing

Contributions welcome! Please consider:
- Adding more web search backends
- Improving document processing capabilities
- Expanding creative response themes
- Optimizing memory usage for larger models

## 📄 License

This project uses various open-source components. Please check individual package licenses:
- GPT-OSS: OpenAI license terms
- Transformers: Apache 2.0
- Other dependencies: Respective licenses

## 🔗 Related Projects

- [OpenAI Harmony](https://github.com/openai/openai-harmony)
- [GPT-OSS](https://github.com/openai/gpt-oss)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

---

*Built with ❤️ for intelligent web search and creative AI interactions*
