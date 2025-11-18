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
        max_tokens: int = 500,
        uncertainty_threshold: float = 1.0
    ) -> Dict[str, Any]:
        """
        Measure uncertainty by querying the LLM multiple times.
        
        Args:
            prompt: The user's prompt to send to the LLM
            num_samples: Number of times to query the LLM (default: 5)
            temperature: Temperature for sampling (higher = more diverse)
            max_tokens: Maximum tokens in the response
            uncertainty_threshold: Threshold for ratio comparison (default: 1.0)
            
        Returns:
            A dictionary containing:
                - responses: List of response texts
                - logprobs: List of log probabilities for each response
                - uncertainty_score: A calculated uncertainty metric
                - analysis: Analysis of the uncertainty
                - is_uncertain: Boolean indicating if LLM is uncertain
                - tool_response: Response based on certainty level
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
        
        # Calculate mean logprob of answers
        answer_mean_logprob = self._calculate_mean_logprob(all_logprobs)
        
        # Calculate mean logprob of uncertainty phrases
        print("\n📊 Calculating logprobs for uncertainty phrases...")
        uncertainty_phrases = ["I'm not sure", "I'm insecure", "I need help"]
        phrase_logprobs = self._get_phrase_logprobs(uncertainty_phrases)
        
        # Calculate mean of uncertainty phrase logprobs
        uncertainty_phrase_mean = sum(phrase_logprobs.values()) / len(phrase_logprobs) if phrase_logprobs else 0.0
        
        # Calculate ratio (avoiding division by zero)
        # Ratio = uncertainty_phrase_logprob / answer_logprob
        # If ratio > threshold, then uncertain; otherwise certain
        if answer_mean_logprob != 0:
            uncertainty_ratio = uncertainty_phrase_mean / answer_mean_logprob
        else:
            uncertainty_ratio = float('inf') if uncertainty_phrase_mean > 0 else 0.0
        
        # Determine if LLM is uncertain based on threshold
        # If ratio of uncertainty to answer logprobs is high, it means uncertain
        is_uncertain = uncertainty_ratio > uncertainty_threshold
        
        print(f"📈 Answer mean logprob: {answer_mean_logprob:.4f}")
        print(f"📉 Uncertainty phrases mean logprob: {uncertainty_phrase_mean:.4f}")
        print(f"⚖️  Uncertainty ratio: {uncertainty_ratio:.4f} (threshold: {uncertainty_threshold})")
        print(f"{'❓ LLM is UNCERTAIN (ratio > threshold)' if is_uncertain else '✅ LLM is CONFIDENT (ratio <= threshold)'}")
        
        # Generate appropriate response based on certainty
        if is_uncertain:
            # Generate "I'm unsure about" message
            tool_response = self._generate_uncertainty_message(prompt, responses)
        else:
            # Return the most common or first valid response
            valid_responses = [r for r in responses if r is not None]
            tool_response = valid_responses[0] if valid_responses else "Unable to generate response."
        
        # Analyze uncertainty
        analysis = self._analyze_uncertainty(responses, all_logprobs)
        analysis["answer_mean_logprob"] = round(answer_mean_logprob, 4)
        analysis["uncertainty_phrase_logprobs"] = {k: round(v, 4) for k, v in phrase_logprobs.items()}
        analysis["uncertainty_phrase_mean_logprob"] = round(uncertainty_phrase_mean, 4)
        analysis["uncertainty_ratio"] = round(uncertainty_ratio, 4)
        analysis["uncertainty_threshold"] = uncertainty_threshold
        analysis["is_uncertain"] = is_uncertain
        
        return {
            "prompt": prompt,
            "num_samples": num_samples,
            "responses": responses,
            "logprobs": all_logprobs,
            "uncertainty_analysis": analysis,
            "is_uncertain": is_uncertain,
            "tool_response": tool_response
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
    
    def _calculate_mean_logprob(self, logprobs_data: List[Any]) -> float:
        """
        Calculate mean log probability from logprobs data.
        
        Args:
            logprobs_data: List of logprobs data
            
        Returns:
            Mean log probability
        """
        if not logprobs_data:
            return 0.0
        
        total_logprob = 0.0
        total_tokens = 0
        
        for logprobs in logprobs_data:
            if logprobs and logprobs.content:
                for token_logprob in logprobs.content:
                    if token_logprob.logprob is not None:
                        total_logprob += token_logprob.logprob
                        total_tokens += 1
        
        if total_tokens == 0:
            return 0.0
        
        return total_logprob / total_tokens
    
    def _get_phrase_logprobs(self, phrases: List[str]) -> Dict[str, float]:
        """
        Get mean logprobs for specific phrases by querying the LLM.
        
        Args:
            phrases: List of phrases to get logprobs for
            
        Returns:
            Dictionary mapping phrases to their mean logprobs
        """
        phrase_logprobs = {}
        
        for phrase in phrases:
            try:
                # Query the LLM with just the phrase
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": "Complete this sentence: " + phrase}
                    ],
                    temperature=0.0,
                    max_tokens=10,
                    logprobs=True,
                    top_logprobs=5
                )
                
                logprobs_data = completion.choices[0].logprobs
                mean_logprob = self._calculate_mean_logprob([logprobs_data])
                phrase_logprobs[phrase] = mean_logprob
                
            except Exception as e:
                print(f"✗ Error getting logprobs for '{phrase}': {str(e)}")
                phrase_logprobs[phrase] = 0.0
        
        return phrase_logprobs
    
    def _generate_uncertainty_message(self, prompt: str, responses: List[str]) -> str:
        """
        Generate an "I'm unsure about..." message when LLM is uncertain.
        
        Args:
            prompt: The original user prompt
            responses: List of responses from the LLM
            
        Returns:
            A message explaining what the LLM is unsure about
        """
        try:
            # Get valid responses
            valid_responses = [r for r in responses if r is not None]
            context = ""
            if valid_responses:
                # Use first response as context
                context = f"\n\nContext: {valid_responses[0][:200]}"
            
            # Ask LLM to explain what it's unsure about
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are helping to identify uncertainty. Complete the following statement about what aspects of the question are unclear or require more information."},
                    {"role": "user", "content": f"Question: {prompt}{context}\n\nI'm unsure about"}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            uncertainty_detail = completion.choices[0].message.content.strip()
            return f"I'm unsure about {uncertainty_detail}\n\nCould you please provide more information or clarify your question?"
            
        except Exception as e:
            print(f"✗ Error generating uncertainty message: {str(e)}")
            return "I'm unsure about how to answer this question accurately. Could you please provide more information or clarify your question?"
    
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
            
            # Add new ratio analysis
            if 'uncertainty_ratio' in analysis:
                output.append(f"\nAnswer Mean Logprob: {analysis['answer_mean_logprob']}")
                output.append(f"Uncertainty Phrases Mean Logprob: {analysis['uncertainty_phrase_mean_logprob']}")
                output.append(f"Uncertainty Ratio: {analysis['uncertainty_ratio']}")
                output.append(f"Threshold: {analysis['uncertainty_threshold']}")
                output.append(f"Status: {'UNCERTAIN (ratio > threshold)' if analysis['is_uncertain'] else 'CONFIDENT (ratio <= threshold)'}")
            
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
