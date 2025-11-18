# Uncertainty-Aware Reasoning

An LLM interface that uses function calling to measure and communicate uncertainty in AI responses.

## Overview

The idea is that instead of sampling the most probable next token by an LLM, we compare beforehand the logits of its answer with some "uncertain" answers. If there is uncertainty in its first answer, it should be able to communicate what it is uncertain about towards the user to improve the overall capability of the LLM.

This project implements a function-calling LLM interface where the LLM **always** uses a `measure_uncertainty` tool for every user query. The tool:

- Queries the same LLM **5 times** (configurable) with the user's prompt
- Captures **token logits** for each response
- Analyzes response diversity and confidence levels
- Provides uncertainty metrics and recommendations

## Features

- 🎯 **Automatic Uncertainty Measurement**: Every query is analyzed for uncertainty
- 📊 **Token Logit Analysis**: Captures and analyzes probability distributions
- 🔄 **Multiple Sampling**: Queries the LLM multiple times to detect inconsistencies
- 📈 **Uncertainty Metrics**: Response diversity, token confidence, and uncertainty levels
- 💬 **Function Calling Interface**: Uses OpenAI's function calling for seamless integration
- 🎨 **Interactive CLI**: Easy-to-use command-line interface

## Quick Start

### Try it Now (No API Key Required!)

```bash
python demo.py
```

See the system in action with simulated responses!

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run

```bash
# Interactive mode
python main.py

# With detailed uncertainty analysis
python main.py --verbose

# Run example
python main.py --example
```

📖 **New to the project?** Check out [QUICKSTART.md](QUICKSTART.md) for a 3-minute guide!

## How It Works

```
User: "What is the capital of France?"
    ↓
LLM Interface → Calls measure_uncertainty function
    ↓
Uncertainty Measurer:
  - Queries LLM 5 times: "What is the capital of France?"
  - Captures token logits for each response
  - Analyzes: All 5 responses say "Paris"
  - Calculates: Low diversity, high confidence
    ↓
LLM: "The capital of France is Paris. 
      [Low uncertainty - consistent across all samples]"
```

## Example Output

```
💬 User: What is quantum entanglement?

🔍 Measuring uncertainty by querying the LLM 5 times...
✓ Sample 1/5 completed
✓ Sample 2/5 completed
...

UNCERTAINTY ANALYSIS
─────────────────────────
Uncertainty Level: MEDIUM
Response Diversity: 0.600 (3/5 unique)
Average Token Confidence: 0.847

Recommendation: The model shows some uncertainty. 
You may want to verify the response or ask for clarification.

🤖 Assistant: Quantum entanglement is a quantum mechanical phenomenon...
[The model shows moderate uncertainty in explaining this complex topic, 
with variations in how it describes the phenomenon across different samples.]
```

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - 3-minute quick start guide (start here!)
- [USAGE.md](USAGE.md) - Comprehensive usage guide and API reference
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Technical implementation details
- [.env.example](.env.example) - Environment configuration template

## Project Structure

```
.
├── main.py                      # Main application entry point
├── requirements.txt             # Python dependencies
├── .env.example                # Environment configuration template
├── src/
│   ├── __init__.py
│   ├── llm_interface.py        # Function-calling LLM interface
│   └── measure_uncertainty.py   # Uncertainty measurement tool
├── README.md                    # This file
└── USAGE.md                     # Detailed usage guide
```

## Key Components

### 1. LLM Function Interface (`src/llm_interface.py`)
- Manages the main LLM that uses function calling
- Defines the `measure_uncertainty` function schema
- Orchestrates the uncertainty measurement workflow

### 2. Uncertainty Measurer (`src/measure_uncertainty.py`)
- Queries the LLM multiple times with the same prompt
- Captures and analyzes token logits
- Calculates uncertainty metrics (diversity, confidence)
- Provides recommendations based on uncertainty levels

### 3. Main Application (`main.py`)
- Interactive CLI for user queries
- Example mode for demonstration
- Verbose mode for detailed analysis

## Configuration

Environment variables (`.env`):

```bash
OPENAI_API_KEY=sk-your-key-here  # Required
OPENAI_MODEL=gpt-4               # Optional, defaults to gpt-4
```

## Use Cases

- **Educational Tools**: Help users understand when AI might be uncertain
- **Research**: Analyze LLM confidence across different domains
- **Decision Support**: Highlight when additional verification is needed
- **Debugging**: Identify prompts that produce inconsistent responses

## Requirements

- Python 3.8+
- OpenAI API key with access to GPT-4 or GPT-3.5-turbo
- Models must support function calling and logprobs

## Cost Considerations

Each user query results in approximately 7 API calls:
- 1 call for function-calling decision
- 5 calls for uncertainty measurement (default)
- 1 call for final response synthesis

## Contributing

Contributions are welcome! Areas for improvement:
- Support for additional LLM providers
- More sophisticated uncertainty metrics
- Visualization of uncertainty distributions
- Batch processing capabilities

## License

MIT License - See LICENSE file for details
