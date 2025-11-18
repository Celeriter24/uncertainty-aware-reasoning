"""
Measure Uncertainty Tool

This module implements the measure_uncertainty function that queries an LLM
multiple times to measure uncertainty in the responses by analyzing token logits.
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json


class UncertaintyMeasurer:
    """
    A class that measures uncertainty in LLM responses by querying the model
    multiple times and analyzing the token logits.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the UncertaintyMeasurer.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable
            model: The model to use for generating responses
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        
    def measure_uncertainty(
        self,
        prompt: str,
        num_samples: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Measure uncertainty by querying the LLM multiple times.
        
        Args:
            prompt: The user's prompt to send to the LLM
            num_samples: Number of times to query the LLM (default: 5)
            temperature: Temperature for sampling (higher = more diverse)
            max_tokens: Maximum tokens in the response
            
        Returns:
            A dictionary containing:
                - responses: List of response texts
                - logprobs: List of log probabilities for each response
                - uncertainty_score: A calculated uncertainty metric
                - analysis: Analysis of the uncertainty
        """
        responses = []
        all_logprobs = []
        
        print(f"\n🔍 Measuring uncertainty by querying the LLM {num_samples} times...\n")
        
        for i in range(num_samples):
            try:
                # Query the LLM with logprobs enabled
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=True,
                    top_logprobs=5  # Get top 5 alternative tokens at each position
                )
                
                response_text = completion.choices[0].message.content
                logprobs_data = completion.choices[0].logprobs
                
                responses.append(response_text)
                all_logprobs.append(logprobs_data)
                
                print(f"✓ Sample {i+1}/{num_samples} completed")
                
            except Exception as e:
                print(f"✗ Error in sample {i+1}: {str(e)}")
                responses.append(None)
                all_logprobs.append(None)
        
        # Analyze uncertainty
        analysis = self._analyze_uncertainty(responses, all_logprobs)
        
        return {
            "prompt": prompt,
            "num_samples": num_samples,
            "responses": responses,
            "logprobs": all_logprobs,
            "uncertainty_analysis": analysis
        }
    
    def _analyze_uncertainty(
        self,
        responses: List[str],
        logprobs_data: List[Any]
    ) -> Dict[str, Any]:
        """
        Analyze uncertainty across multiple responses.
        
        Args:
            responses: List of response texts
            logprobs_data: List of logprobs data from the API
            
        Returns:
            Dictionary with uncertainty analysis
        """
        # Filter out None responses
        valid_responses = [r for r in responses if r is not None]
        valid_logprobs = [lp for lp in logprobs_data if lp is not None]
        
        if len(valid_responses) == 0:
            return {
                "error": "No valid responses received",
                "uncertainty_level": "unknown"
            }
        
        # Check response diversity
        unique_responses = len(set(valid_responses))
        response_diversity = unique_responses / len(valid_responses)
        
        # Calculate average token confidence from logprobs
        avg_confidence = self._calculate_average_confidence(valid_logprobs)
        
        # Determine uncertainty level
        if response_diversity >= 0.8:
            uncertainty_level = "high"
            recommendation = "The model is highly uncertain. Consider reformulating the question or providing more context."
        elif response_diversity >= 0.4:
            uncertainty_level = "medium"
            recommendation = "The model shows some uncertainty. You may want to verify the response or ask for clarification."
        else:
            uncertainty_level = "low"
            recommendation = "The model appears confident in its response."
        
        return {
            "unique_responses": unique_responses,
            "total_samples": len(valid_responses),
            "response_diversity": round(response_diversity, 3),
            "average_token_confidence": round(avg_confidence, 3),
            "uncertainty_level": uncertainty_level,
            "recommendation": recommendation
        }
    
    def _calculate_average_confidence(self, logprobs_data: List[Any]) -> float:
        """
        Calculate average confidence from logprobs data.
        
        Args:
            logprobs_data: List of logprobs data
            
        Returns:
            Average confidence score (0-1)
        """
        if not logprobs_data:
            return 0.0
        
        total_confidence = 0.0
        total_tokens = 0
        
        for logprobs in logprobs_data:
            if logprobs and logprobs.content:
                for token_logprob in logprobs.content:
                    if token_logprob.logprob is not None:
                        # Convert log probability to probability
                        prob = 2 ** token_logprob.logprob
                        total_confidence += prob
                        total_tokens += 1
        
        if total_tokens == 0:
            return 0.0
        
        return total_confidence / total_tokens
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """
        Format the uncertainty measurement results for display.
        
        Args:
            results: Results dictionary from measure_uncertainty
            
        Returns:
            Formatted string representation
        """
        output = []
        output.append("=" * 80)
        output.append("UNCERTAINTY MEASUREMENT RESULTS")
        output.append("=" * 80)
        output.append(f"\nPrompt: {results['prompt']}")
        output.append(f"\nNumber of samples: {results['num_samples']}")
        
        analysis = results['uncertainty_analysis']
        
        output.append("\n" + "-" * 80)
        output.append("UNCERTAINTY ANALYSIS")
        output.append("-" * 80)
        
        if 'error' in analysis:
            output.append(f"\nError: {analysis['error']}")
        else:
            output.append(f"\nUncertainty Level: {analysis['uncertainty_level'].upper()}")
            output.append(f"Response Diversity: {analysis['response_diversity']} ({analysis['unique_responses']}/{analysis['total_samples']} unique)")
            output.append(f"Average Token Confidence: {analysis['average_token_confidence']}")
            output.append(f"\nRecommendation: {analysis['recommendation']}")
        
        output.append("\n" + "-" * 80)
        output.append("INDIVIDUAL RESPONSES")
        output.append("-" * 80)
        
        for i, response in enumerate(results['responses'], 1):
            if response is not None:
                output.append(f"\nResponse {i}:")
                output.append(f"{response[:200]}..." if len(response) > 200 else response)
        
        output.append("\n" + "=" * 80)
        
        return "\n".join(output)
