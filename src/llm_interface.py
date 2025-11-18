"""
LLM Function-Calling Interface

This module implements an interface towards a function-calling LLM that always
uses the measure_uncertainty function/tool when responding to user queries.
"""

import os
import json
from typing import Optional, Dict, Any
from openai import OpenAI
from src.measure_uncertainty import UncertaintyMeasurer


class LLMFunctionInterface:
    """
    An interface for a function-calling LLM that uses the measure_uncertainty tool.
    """
    
    # Define the function schema for the measure_uncertainty tool
    MEASURE_UNCERTAINTY_FUNCTION = {
        "type": "function",
        "function": {
            "name": "measure_uncertainty",
            "description": "Measures uncertainty in LLM responses by querying the model multiple times and analyzing token logits to assess confidence and variability in answers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The user's question or prompt to analyze for uncertainty"
                    },
                    "num_samples": {
                        "type": "integer",
                        "description": "Number of times to query the LLM (default: 5)",
                        "default": 5
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature for sampling - higher values produce more diverse responses (default: 0.7)",
                        "default": 0.7
                    },
                    "uncertainty_threshold": {
                        "type": "number",
                        "description": "Threshold for uncertainty ratio (uncertainty_phrase_logprob / answer_logprob). If ratio > threshold, LLM is uncertain; otherwise certain. This value should be provided by the user based on their requirements. Typical values: 0.8-1.2 (default: 1.0)",
                        "default": 1.0
                    }
                },
                "required": ["prompt"]
            }
        }
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        uncertainty_model: Optional[str] = None
    ):
        """
        Initialize the LLM function-calling interface.
        
        Args:
            api_key: OpenAI API key
            model: Model for the main interface LLM
            uncertainty_model: Model for uncertainty measurement (defaults to same as model)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.uncertainty_model = uncertainty_model or model
        self.client = OpenAI(api_key=self.api_key)
        self.uncertainty_measurer = UncertaintyMeasurer(
            api_key=self.api_key,
            model=self.uncertainty_model
        )
        self.conversation_history = []
        self.uncertainty_threshold = 1.0  # Default threshold, can be overridden by user
        
    def process_user_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message through the function-calling LLM.
        
        The LLM is instructed to always use the measure_uncertainty function
        to analyze the user's query.
        
        Args:
            user_message: The user's input message
            
        Returns:
            Dictionary containing the conversation and uncertainty results
        """
        print(f"\n💬 User: {user_message}\n")
        
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Create a system message that instructs the LLM to use the function
        system_message = {
            "role": "system",
            "content": (
                f"You are an uncertainty-aware AI assistant. For every user query, "
                f"you MUST use the 'measure_uncertainty' function to analyze the query "
                f"and measure uncertainty in the response. "
                f"IMPORTANT: Always use uncertainty_threshold={self.uncertainty_threshold} in your function call. "
                f"This threshold has been provided by the user. "
                f"After receiving the uncertainty results, provide a comprehensive answer to the user that incorporates "
                f"the uncertainty analysis."
            )
        }
        
        # Prepare messages for the LLM
        messages = [system_message] + self.conversation_history
        
        # Call the LLM with function calling
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[self.MEASURE_UNCERTAINTY_FUNCTION],
            tool_choice={"type": "function", "function": {"name": "measure_uncertainty"}}
        )
        
        response_message = response.choices[0].message
        
        # Check if the LLM wants to call the function
        if response_message.tool_calls:
            # Add the assistant's response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response_message.tool_calls
                ]
            })
            
            # Execute the function call
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"🔧 LLM is calling function: {function_name}")
            print(f"📋 Arguments: {json.dumps(function_args, indent=2)}\n")
            
            # Call the measure_uncertainty function
            # Use user-provided threshold, not LLM-decided threshold
            uncertainty_results = self.uncertainty_measurer.measure_uncertainty(
                prompt=function_args.get("prompt", user_message),
                num_samples=function_args.get("num_samples", 5),
                temperature=function_args.get("temperature", 0.7),
                uncertainty_threshold=self.uncertainty_threshold  # Always use user-provided threshold
            )
            
            # Format the results
            formatted_results = self.uncertainty_measurer.format_results(uncertainty_results)
            
            # Prepare tool response with both analysis and the actual response
            tool_content = {
                "analysis": uncertainty_results["uncertainty_analysis"],
                "is_uncertain": uncertainty_results.get("is_uncertain", False),
                "tool_response": uncertainty_results.get("tool_response", "")
            }
            
            # Add function result to conversation history
            self.conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_content)
            })
            
            # Get the final response from the LLM
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=[system_message] + self.conversation_history
            )
            
            final_message = final_response.choices[0].message.content
            
            # Add final response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": final_message
            })
            
            print(f"\n🤖 Assistant: {final_message}\n")
            
            return {
                "user_message": user_message,
                "function_called": function_name,
                "function_args": function_args,
                "uncertainty_results": uncertainty_results,
                "formatted_results": formatted_results,
                "assistant_response": final_message
            }
        else:
            # This shouldn't happen with tool_choice set to required
            print("⚠️  LLM did not call the function as expected")
            return {
                "user_message": user_message,
                "error": "Function was not called",
                "assistant_response": response_message.content
            }
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        print("\n🔄 Conversation history reset\n")
