"""
Main Application - Uncertainty-Aware LLM Interface

This is the main entry point for the uncertainty-aware reasoning system.
It demonstrates how to use the LLM function-calling interface with the
measure_uncertainty tool.
"""

import os
import sys
from dotenv import load_dotenv
from src.llm_interface import LLMFunctionInterface


def main():
    """Main application entry point."""
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please create a .env file with your OpenAI API key")
        print("Example: OPENAI_API_KEY=sk-...")
        sys.exit(1)
    
    print("=" * 80)
    print("UNCERTAINTY-AWARE LLM INTERFACE")
    print("=" * 80)
    print("\nThis interface uses function calling to measure uncertainty in LLM responses.")
    print("The LLM will automatically use the 'measure_uncertainty' tool for every query.")
    print("\nCommands:")
    print("  - Type your question to get an uncertainty-aware response")
    print("  - Type 'reset' to clear conversation history")
    print("  - Type 'quit' or 'exit' to exit")
    print("=" * 80)
    
    # Initialize the interface
    model = os.environ.get("OPENAI_MODEL", "gpt-4")
    print(f"\n🚀 Initializing LLM interface with model: {model}\n")
    
    interface = LLMFunctionInterface(model=model)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye!\n")
                break
            
            if user_input.lower() == 'reset':
                interface.reset_conversation()
                continue
            
            # Process the user message
            result = interface.process_user_message(user_input)
            
            # Optionally display the detailed uncertainty analysis
            if "--verbose" in sys.argv or "-v" in sys.argv:
                print("\n" + "=" * 80)
                print("DETAILED UNCERTAINTY ANALYSIS")
                print("=" * 80)
                print(result["formatted_results"])
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            import traceback
            traceback.print_exc()


def run_example():
    """Run a simple example without interactive mode."""
    load_dotenv()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    print("=" * 80)
    print("EXAMPLE: Uncertainty-Aware LLM Interface")
    print("=" * 80)
    
    interface = LLMFunctionInterface()
    
    # Example question
    example_question = "What is the capital of France?"
    
    print(f"\nExample question: {example_question}\n")
    
    result = interface.process_user_message(example_question)
    
    print("\n" + "=" * 80)
    print("FULL RESULTS")
    print("=" * 80)
    print(result["formatted_results"])


if __name__ == "__main__":
    if "--example" in sys.argv:
        run_example()
    else:
        main()
