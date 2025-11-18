"""
Simple example demonstrating the uncertainty-aware LLM interface.

This example shows how to use the system programmatically.
"""

import os
from dotenv import load_dotenv
from src.llm_interface import LLMFunctionInterface
from src.measure_uncertainty import UncertaintyMeasurer


def example_1_basic_usage():
    """Example 1: Basic usage of the LLM function interface."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)
    
    # Initialize the interface
    interface = LLMFunctionInterface(model="gpt-4")
    
    # Ask a simple factual question
    question = "What is the capital of France?"
    print(f"\nQuestion: {question}")
    
    result = interface.process_user_message(question)
    
    print("\n" + "-" * 80)
    print("Assistant's Response:")
    print("-" * 80)
    print(result["assistant_response"])
    
    print("\n" + "-" * 80)
    print("Uncertainty Analysis:")
    print("-" * 80)
    analysis = result["uncertainty_results"]["uncertainty_analysis"]
    print(f"Uncertainty Level: {analysis['uncertainty_level']}")
    print(f"Response Diversity: {analysis['response_diversity']}")
    print(f"Recommendation: {analysis['recommendation']}")


def example_2_direct_uncertainty_measurement():
    """Example 2: Using the uncertainty measurer directly."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Direct Uncertainty Measurement")
    print("=" * 80)
    
    # Initialize the uncertainty measurer
    measurer = UncertaintyMeasurer(model="gpt-4")
    
    # Measure uncertainty for a more ambiguous question
    question = "Is artificial intelligence good or bad?"
    print(f"\nQuestion: {question}")
    
    results = measurer.measure_uncertainty(
        prompt=question,
        num_samples=5,
        temperature=0.8  # Higher temperature for more diversity
    )
    
    # Display formatted results
    print("\n" + measurer.format_results(results))


def example_3_comparing_uncertainties():
    """Example 3: Compare uncertainty across different types of questions."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Comparing Uncertainties")
    print("=" * 80)
    
    measurer = UncertaintyMeasurer(model="gpt-4")
    
    questions = [
        ("Factual", "What is 2 + 2?"),
        ("Ambiguous", "What is the best programming language?"),
        ("Complex", "How does consciousness emerge from neural activity?")
    ]
    
    for category, question in questions:
        print(f"\n{'-' * 80}")
        print(f"{category} Question: {question}")
        print('-' * 80)
        
        results = measurer.measure_uncertainty(
            prompt=question,
            num_samples=3,  # Fewer samples for this comparison
            temperature=0.7
        )
        
        analysis = results["uncertainty_analysis"]
        print(f"\nUncertainty Level: {analysis['uncertainty_level']}")
        print(f"Response Diversity: {analysis['response_diversity']}")
        print(f"Unique Responses: {analysis['unique_responses']}/{analysis['total_samples']}")


def main():
    """Run all examples."""
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please create a .env file with your OpenAI API key")
        return
    
    print("\n" + "=" * 80)
    print("UNCERTAINTY-AWARE LLM INTERFACE - EXAMPLES")
    print("=" * 80)
    print("\nNote: These examples make multiple API calls and may take a minute to run.")
    print("Each example demonstrates different aspects of the system.")
    
    try:
        # Run examples
        example_1_basic_usage()
        
        # Uncomment to run additional examples (they make more API calls)
        # example_2_direct_uncertainty_measurement()
        # example_3_comparing_uncertainties()
        
        print("\n\n" + "=" * 80)
        print("EXAMPLES COMPLETED")
        print("=" * 80)
        print("\nUncomment additional examples in example.py to see more functionality.")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
