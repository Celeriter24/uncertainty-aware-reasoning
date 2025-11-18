"""
Demo Script - Uncertainty-Aware LLM Interface

This script demonstrates the system's functionality without requiring an API key.
It uses mock responses to show how the system works.
"""

from unittest.mock import Mock, patch
from src.measure_uncertainty import UncertaintyMeasurer


def create_mock_response(text, logprob=-0.1):
    """Create a mock OpenAI API response."""
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = text
    
    # Create mock logprobs
    mock_logprobs = Mock()
    mock_content = []
    for _ in range(len(text.split())):  # One token per word for simplicity
        token_logprob = Mock()
        token_logprob.logprob = logprob
        mock_content.append(token_logprob)
    
    mock_logprobs.content = mock_content
    mock_choice.logprobs = mock_logprobs
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    return mock_response


def demo_low_uncertainty():
    """Demonstrate low uncertainty (confident response)."""
    print("=" * 80)
    print("DEMO 1: LOW UNCERTAINTY - Factual Question")
    print("=" * 80)
    print("\nQuestion: What is the capital of France?")
    print("\nSimulating 5 LLM queries...")
    
    # All responses are the same = low uncertainty
    responses = [
        "Paris is the capital of France.",
        "The capital of France is Paris.",
        "Paris is the capital city of France.",
        "France's capital is Paris.",
        "Paris is the capital of France."
    ]
    
    with patch('src.measure_uncertainty.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            create_mock_response(resp, logprob=-0.05) for resp in responses
        ]
        mock_openai.return_value = mock_client
        
        measurer = UncertaintyMeasurer(api_key="demo-key")
        results = measurer.measure_uncertainty("What is the capital of France?", num_samples=5)
        
        print("\n" + "-" * 80)
        print("RESULTS")
        print("-" * 80)
        analysis = results["uncertainty_analysis"]
        print(f"Uncertainty Level: {analysis['uncertainty_level'].upper()}")
        print(f"Response Diversity: {analysis['response_diversity']} ({analysis['unique_responses']}/{analysis['total_samples']} unique)")
        print(f"Average Token Confidence: {analysis['average_token_confidence']}")
        print(f"\nInterpretation: {analysis['recommendation']}")


def demo_high_uncertainty():
    """Demonstrate high uncertainty (controversial/ambiguous question)."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: HIGH UNCERTAINTY - Subjective Question")
    print("=" * 80)
    print("\nQuestion: What is the best programming language?")
    print("\nSimulating 5 LLM queries...")
    
    # All responses are different = high uncertainty
    responses = [
        "Python is the best programming language for beginners and data science.",
        "JavaScript is the most versatile programming language for web development.",
        "Rust is the best for systems programming with memory safety guarantees.",
        "Go is the best for building scalable concurrent applications.",
        "C++ offers the best performance for computationally intensive tasks."
    ]
    
    with patch('src.measure_uncertainty.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            create_mock_response(resp, logprob=-0.8) for resp in responses
        ]
        mock_openai.return_value = mock_client
        
        measurer = UncertaintyMeasurer(api_key="demo-key")
        results = measurer.measure_uncertainty("What is the best programming language?", num_samples=5)
        
        print("\n" + "-" * 80)
        print("RESULTS")
        print("-" * 80)
        analysis = results["uncertainty_analysis"]
        print(f"Uncertainty Level: {analysis['uncertainty_level'].upper()}")
        print(f"Response Diversity: {analysis['response_diversity']} ({analysis['unique_responses']}/{analysis['total_samples']} unique)")
        print(f"Average Token Confidence: {analysis['average_token_confidence']}")
        print(f"\nInterpretation: {analysis['recommendation']}")


def demo_medium_uncertainty():
    """Demonstrate medium uncertainty (complex question with variations)."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: MEDIUM UNCERTAINTY - Complex Question")
    print("=" * 80)
    print("\nQuestion: How does consciousness emerge from neural activity?")
    print("\nSimulating 5 LLM queries...")
    
    # Some responses are similar, some different = medium uncertainty
    responses = [
        "Consciousness emerges through complex neural network interactions and feedback loops.",
        "The integration of information across brain regions creates conscious experience.",
        "Consciousness emerges through integrated information processing in neural networks.",
        "Neural synchronization and global workspace theory explain consciousness emergence.",
        "The integration of sensory data and memory creates conscious awareness."
    ]
    
    with patch('src.measure_uncertainty.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            create_mock_response(resp, logprob=-0.4) for resp in responses
        ]
        mock_openai.return_value = mock_client
        
        measurer = UncertaintyMeasurer(api_key="demo-key")
        results = measurer.measure_uncertainty("How does consciousness emerge?", num_samples=5)
        
        print("\n" + "-" * 80)
        print("RESULTS")
        print("-" * 80)
        analysis = results["uncertainty_analysis"]
        print(f"Uncertainty Level: {analysis['uncertainty_level'].upper()}")
        print(f"Response Diversity: {analysis['response_diversity']} ({analysis['unique_responses']}/{analysis['total_samples']} unique)")
        print(f"Average Token Confidence: {analysis['average_token_confidence']}")
        print(f"\nInterpretation: {analysis['recommendation']}")


def demo_function_calling():
    """Demonstrate the function calling schema."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: FUNCTION CALLING SCHEMA")
    print("=" * 80)
    
    with patch('src.llm_interface.OpenAI'):
        from src.llm_interface import LLMFunctionInterface
        
        interface = LLMFunctionInterface(api_key="demo-key")
        schema = interface.MEASURE_UNCERTAINTY_FUNCTION
        
        print("\nFunction Schema that LLM receives:")
        print("-" * 80)
        print(f"Function Name: {schema['function']['name']}")
        print(f"Description: {schema['function']['description'][:100]}...")
        print("\nParameters:")
        for param, details in schema['function']['parameters']['properties'].items():
            required = "✓" if param in schema['function']['parameters']['required'] else "○"
            print(f"  [{required}] {param}: {details['description'][:70]}...")
        
        print("\nThe LLM is forced to use this function via:")
        print("  tool_choice={'type': 'function', 'function': {'name': 'measure_uncertainty'}}")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("UNCERTAINTY-AWARE LLM INTERFACE - DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo shows how the system measures uncertainty across different")
    print("types of questions without requiring an actual API key.")
    print("\nNote: These are simulated responses for demonstration purposes.")
    
    try:
        demo_low_uncertainty()
        demo_high_uncertainty()
        demo_medium_uncertainty()
        demo_function_calling()
        
        print("\n\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  1. Factual questions → Low uncertainty (consistent responses)")
        print("  2. Subjective questions → High uncertainty (diverse responses)")
        print("  3. Complex questions → Medium uncertainty (some variation)")
        print("  4. LLM is forced to always use the measure_uncertainty function")
        print("\nTo use with real API:")
        print("  1. Set OPENAI_API_KEY in .env file")
        print("  2. Run: python main.py")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
