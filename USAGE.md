# Usage Guide

## Overview

This project implements an uncertainty-aware LLM interface that uses function calling to measure uncertainty in responses. The system automatically invokes a `measure_uncertainty` tool for every user query, which:

1. Calls the same LLM 5 times (configurable) with the user's prompt
2. Captures token logits for each response
3. Analyzes response diversity and confidence
4. Provides an uncertainty assessment

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4
```

## Running the Application

### Interactive Mode (Default)

```bash
python main.py
```

This starts an interactive session where you can ask questions. The LLM will automatically use the `measure_uncertainty` function for each query.

### Interactive Mode with Verbose Output

```bash
python main.py --verbose
```

or

```bash
python main.py -v
```

This shows detailed uncertainty analysis including all individual responses.

### Example Mode

```bash
python main.py --example
```

Runs a simple example to demonstrate the functionality.

## How It Works

### Architecture

1. **User Interface** (`main.py`): Entry point for user interactions
2. **LLM Function Interface** (`src/llm_interface.py`): Manages the function-calling LLM
3. **Uncertainty Measurer** (`src/measure_uncertainty.py`): Implements the core uncertainty measurement

### Function Calling Flow

```
User Query
    ↓
LLM Interface (with function calling enabled)
    ↓
LLM decides to call measure_uncertainty function
    ↓
Uncertainty Measurer:
    - Queries LLM 5 times with same prompt
    - Captures token logits
    - Analyzes response diversity
    - Calculates confidence scores
    ↓
Results returned to LLM Interface
    ↓
LLM synthesizes final response with uncertainty awareness
    ↓
Response presented to User
```

### Uncertainty Metrics

The system provides several metrics:

- **Response Diversity**: Ratio of unique responses to total samples
- **Average Token Confidence**: Mean probability across all tokens
- **Uncertainty Level**: High, Medium, or Low
- **Recommendation**: Actionable advice based on uncertainty

### Uncertainty Levels

- **Low** (diversity < 0.4): Model is confident
- **Medium** (0.4 ≤ diversity < 0.8): Some uncertainty present
- **High** (diversity ≥ 0.8): Significant uncertainty detected

## Example Usage

```python
from dotenv import load_dotenv
from src.llm_interface import LLMFunctionInterface

load_dotenv()

# Initialize the interface
interface = LLMFunctionInterface(model="gpt-4")

# Process a user query
result = interface.process_user_message("What is the capital of France?")

# Access results
print(result["assistant_response"])
print(result["formatted_results"])
```

## Programmatic Usage

### Using the Uncertainty Measurer Directly

```python
from src.measure_uncertainty import UncertaintyMeasurer

measurer = UncertaintyMeasurer(model="gpt-4")

# Measure uncertainty for a prompt
results = measurer.measure_uncertainty(
    prompt="What is quantum entanglement?",
    num_samples=5,
    temperature=0.7
)

# View analysis
print(results["uncertainty_analysis"])
```

### Customizing Parameters

```python
# Use more samples for better uncertainty estimation
results = measurer.measure_uncertainty(
    prompt="Explain artificial intelligence",
    num_samples=10,
    temperature=0.8,
    max_tokens=1000
)
```

## API Reference

### LLMFunctionInterface

#### `__init__(api_key=None, model="gpt-4", uncertainty_model=None)`
Initialize the interface.

#### `process_user_message(user_message: str) -> Dict`
Process a user message and return results including uncertainty analysis.

#### `reset_conversation()`
Clear conversation history.

### UncertaintyMeasurer

#### `measure_uncertainty(prompt: str, num_samples=5, temperature=0.7, max_tokens=500, uncertainty_threshold=1.0) -> Dict`
Measure uncertainty by querying the LLM multiple times.

Parameters:
- `prompt`: The question or prompt to analyze
- `num_samples`: Number of times to query (default: 5)
- `temperature`: Sampling temperature (default: 0.7)
- `max_tokens`: Max tokens per response (default: 500)
- `uncertainty_threshold`: Certainty ratio threshold (default: 1.0)

Returns:
- `responses`: List of response texts
- `logprobs`: Token log probabilities
- `uncertainty_analysis`: Detailed analysis including:
  - Response diversity and confidence
  - Answer mean logprob
  - Uncertainty phrase logprobs
  - Certainty ratio
  - Threshold comparison
  - Uncertainty status (is_uncertain)
- `is_uncertain`: Boolean indicating if LLM is uncertain
- `tool_response`: The response to return (answer or clarification request)

#### `format_results(results: Dict) -> str`
Format results for human-readable display.

## Tips

1. **For factual questions**: Expect low uncertainty and confident responses
2. **For opinion-based questions**: Expect higher uncertainty and possible clarification requests
3. **For ambiguous questions**: Uncertainty measurement helps identify areas needing clarification
4. **Higher temperature**: Increases response diversity, better for measuring uncertainty
5. **More samples**: Provides more accurate uncertainty estimates but increases API costs
6. **Adjusting threshold**: 
   - Lower threshold (e.g., 0.5): More conservative, requests clarification more often
   - Higher threshold (e.g., 2.0): More permissive, provides answers more readily
   - Default (1.0): Balanced approach

## Troubleshooting

### API Key Issues
- Ensure your `.env` file exists and contains a valid `OPENAI_API_KEY`
- Check that the API key has proper permissions

### Rate Limiting
- The tool makes multiple API calls (5 by default)
- Consider reducing `num_samples` if hitting rate limits

### Model Compatibility
- The system requires models that support function calling (GPT-4, GPT-3.5-turbo)
- Token logprobs are required, ensure your model/API plan supports this feature

## Cost Considerations

Each query results in:
- 1 API call to the function-calling LLM
- N API calls for uncertainty measurement (default: 5)
- 3 API calls for uncertainty phrase logprobs
- 1 API call for generating clarification message (if uncertain)
- 1 final API call for response synthesis

Total: ~10-11 API calls per user query with default settings.
