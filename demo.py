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
    print("DEMO 1: CONFIDENT RESPONSE - High Certainty Ratio")
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
    
    # Mock uncertainty phrases
    uncertainty_phrases_responses = [
        "I'm not sure about that",
        "I'm insecure about this",
        "I need help with this"
    ]
    
    with patch('src.measure_uncertainty.OpenAI') as mock_openai:
        mock_client = Mock()
        # First 5 calls are for answer queries (high confidence)
        # Next 3 calls are for uncertainty phrases (low confidence)
        mock_client.chat.completions.create.side_effect = [
            create_mock_response(resp, logprob=-0.05) for resp in responses
        ] + [
            create_mock_response(phrase, logprob=-2.0) for phrase in uncertainty_phrases_responses
        ]
        mock_openai.return_value = mock_client
        
        measurer = UncertaintyMeasurer(api_key="demo-key")
        results = measurer.measure_uncertainty("What is the capital of France?", num_samples=5, uncertainty_threshold=1.0)
        
        print("\n" + "-" * 80)
        print("RESULTS")
        print("-" * 80)
        analysis = results["uncertainty_analysis"]
        print(f"Uncertainty Level: {analysis['uncertainty_level'].upper()}")
        print(f"Response Diversity: {analysis['response_diversity']} ({analysis['unique_responses']}/{analysis['total_samples']} unique)")
        print(f"Average Token Confidence: {analysis['average_token_confidence']}")
        
        if 'certainty_ratio' in analysis:
            print(f"\n🎯 NEW FEATURE - Certainty Ratio Analysis:")
            print(f"  Answer Mean Logprob: {analysis['answer_mean_logprob']}")
            print(f"  Uncertainty Phrases Mean Logprob: {analysis['uncertainty_phrase_mean_logprob']}")
            print(f"  Certainty Ratio: {analysis['certainty_ratio']}")
            print(f"  Threshold: {analysis['uncertainty_threshold']}")
            print(f"  Status: {'UNCERTAIN' if analysis['is_uncertain'] else 'CONFIDENT'}")
            print(f"\nTool Response: {results.get('tool_response', 'N/A')[:100]}...")
        
        print(f"\nInterpretation: {analysis['recommendation']}")


def demo_high_uncertainty():
    """Demonstrate high uncertainty (controversial/ambiguous question)."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: UNCERTAIN RESPONSE - Low Certainty Ratio")
    print("=" * 80)
    print("\nQuestion: What is the meaning of existence?")
    print("\nSimulating 5 LLM queries...")
    
    # All responses are different = high uncertainty
    responses = [
        "Existence may relate to consciousness and subjective experience.",
        "From a philosophical perspective, existence could involve being and nothingness.",
        "Perhaps existence is about creating meaning through our choices.",
        "The meaning might be found in relationships and connections.",
        "Existence could be an emergent property of the universe."
    ]
    
    # Mock uncertainty phrases
    uncertainty_phrases_responses = [
        "I'm not sure about that",
        "I'm insecure about this", 
        "I need help with this"
    ]
    
    with patch('src.measure_uncertainty.OpenAI') as mock_openai:
        mock_client = Mock()
        # First 5 calls are for answer queries (low confidence - similar to uncertainty phrases)
        # Next 3 calls are for uncertainty phrases
        # Last call is for generating the uncertainty message
        mock_client.chat.completions.create.side_effect = [
            create_mock_response(resp, logprob=-1.8) for resp in responses
        ] + [
            create_mock_response(phrase, logprob=-2.0) for phrase in uncertainty_phrases_responses
        ] + [
            create_mock_response("the specific aspect of existence you're asking about - whether it's philosophical, scientific, or personal meaning. Could you clarify?", logprob=-0.3)
        ]
        mock_openai.return_value = mock_client
        
        measurer = UncertaintyMeasurer(api_key="demo-key")
        results = measurer.measure_uncertainty("What is the meaning of existence?", num_samples=5, uncertainty_threshold=1.0)
        
        print("\n" + "-" * 80)
        print("RESULTS")
        print("-" * 80)
        analysis = results["uncertainty_analysis"]
        print(f"Uncertainty Level: {analysis['uncertainty_level'].upper()}")
        print(f"Response Diversity: {analysis['response_diversity']} ({analysis['unique_responses']}/{analysis['total_samples']} unique)")
        print(f"Average Token Confidence: {analysis['average_token_confidence']}")
        
        if 'certainty_ratio' in analysis:
            print(f"\n🎯 NEW FEATURE - Certainty Ratio Analysis:")
            print(f"  Answer Mean Logprob: {analysis['answer_mean_logprob']}")
            print(f"  Uncertainty Phrases Mean Logprob: {analysis['uncertainty_phrase_mean_logprob']}")
            print(f"  Certainty Ratio: {analysis['certainty_ratio']}")
            print(f"  Threshold: {analysis['uncertainty_threshold']}")
            print(f"  Status: {'UNCERTAIN' if analysis['is_uncertain'] else 'CONFIDENT'}")
            print(f"\nTool Response: {results.get('tool_response', 'N/A')[:150]}...")
        
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
            if param == 'uncertainty_threshold':
                print(f"       🎯 NEW: Controls certainty ratio threshold!")
        
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
