# FLAN-T5 CLI

A command-line interface for interacting with Google's FLAN-T5 models, providing an easy way to generate text using state-of-the-art language models.

## Features

- Interactive chat interface with the FLAN-T5 model
- Support for different model sizes (small, base, large, xl)
- Customizable generation parameters (temperature, top-p sampling)
- Runs on both CPU and GPU (CUDA)
- Rich text formatting in the terminal

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flan-t5-cli.git
   cd flan-t5-cli
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Usage
```bash
flan-t5-cli
```

### Specify a Different Model
```bash
flan-t5-cli --model google/flan-t5-large
```

### Force CPU Usage
```bash
flan-t5-cli --device cpu
```

## Available Commands

- Type your message and press Enter to get a response
- `/model` - Switch to a different FLAN-T5 model
- `/help` - Show help information
- `/quit` or `Ctrl+C` - Exit the application

## Available Models

- `google/flan-t5-small` (fastest, least capable)
- `google/flan-t5-base` (default, good balance)
- `google/flan-t5-large` (slower, more capable)
- `google/flan-t5-xl` (slowest, most capable)

## Generation Parameters

You can customize the text generation by modifying these parameters in the code:

- `temperature` (default: 0.7) - Controls randomness (higher = more random)
- `top_p` (default: 0.9) - Nucleus sampling parameter
- `max_length` (default: 200) - Maximum number of tokens to generate

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Rich (for terminal formatting)

## Acknowledgments

- [Google Research](https://github.com/google-research/t5x) for the FLAN-T5 models
- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
