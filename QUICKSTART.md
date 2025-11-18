# Quick Start Guide

Get started with the uncertainty-aware LLM interface in 3 minutes!

## 🚀 Quick Demo (No API Key Required)

See the system in action immediately:

```bash
python demo.py
```

This runs a demonstration with simulated responses showing:
- Low uncertainty (factual questions)
- High uncertainty (subjective questions)  
- Medium uncertainty (complex questions)
- Function calling schema

## 📦 Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API key
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-key-here
```

## 🎮 Interactive Mode

```bash
python main.py
```

Example interaction:
```
👤 You: What is the capital of France?

🔧 LLM is calling function: measure_uncertainty
🔍 Measuring uncertainty by querying the LLM 5 times...
✓ Sample 1/5 completed
...

🤖 Assistant: The capital of France is Paris.
[Low uncertainty - consistent across all samples]
```

## 💻 Programmatic Usage

```python
from dotenv import load_dotenv
from src.llm_interface import LLMFunctionInterface

load_dotenv()

# Initialize
interface = LLMFunctionInterface(model="gpt-4")

# Ask a question
result = interface.process_user_message("What is quantum entanglement?")

# Get response
print(result["assistant_response"])

# Get uncertainty analysis
analysis = result["uncertainty_results"]["uncertainty_analysis"]
print(f"Uncertainty: {analysis['uncertainty_level']}")
print(f"Diversity: {analysis['response_diversity']}")
```

## 📊 What You Get

For each query, the system provides:

1. **5 independent LLM responses** (configurable)
2. **Token logits** for each response
3. **Uncertainty metrics**:
   - Response diversity (0-1)
   - Average token confidence (0-1)
   - Uncertainty level (Low/Medium/High)
4. **Recommendations** based on uncertainty

## 🔧 Common Commands

```bash
# Interactive with verbose output
python main.py --verbose

# Run examples
python main.py --example
python example.py

# Run tests
python -m unittest discover tests -v

# Run demo (no API key needed)
python demo.py
```

## 🎯 Example Output

```
UNCERTAINTY ANALYSIS
────────────────────
Uncertainty Level: MEDIUM
Response Diversity: 0.600 (3/5 unique)
Average Token Confidence: 0.847

Recommendation: The model shows some uncertainty.
You may want to verify the response or ask for clarification.
```

## 📚 Next Steps

- Read [USAGE.md](USAGE.md) for comprehensive guide
- Check [IMPLEMENTATION.md](IMPLEMENTATION.md) for technical details
- See [example.py](example.py) for more code examples
- Review [README.md](README.md) for project overview

## 🔑 Key Features

✅ Automatic uncertainty measurement for every query  
✅ Token logit analysis for confidence assessment  
✅ Function-calling interface (LLM always uses measure_uncertainty)  
✅ Interactive CLI and programmatic API  
✅ Configurable sampling parameters  
✅ Comprehensive documentation and examples  

## 💡 Tips

- Higher `temperature` → more diverse responses → better uncertainty detection
- More `num_samples` → more accurate uncertainty estimates → higher cost
- Factual questions typically show low uncertainty
- Subjective/ambiguous questions show high uncertainty

## ⚠️ Requirements

- Python 3.8+
- OpenAI API key (GPT-4 or GPT-3.5-turbo)
- Models must support function calling and logprobs

## 🆘 Troubleshooting

**"OPENAI_API_KEY environment variable not set"**
- Create `.env` file with your API key

**"No module named 'openai'"**
- Run: `pip install -r requirements.txt`

**Want to test without API key?**
- Run: `python demo.py`

## 📞 Support

For more help:
1. Check [USAGE.md](USAGE.md) for detailed documentation
2. Review examples in [example.py](example.py)
3. Run the demo: `python demo.py`
4. Check the tests: `tests/test_measure_uncertainty.py`

---

**Ready to start?** Run `python demo.py` to see it in action!
