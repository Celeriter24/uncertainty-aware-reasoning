"""
Unit tests for the measure_uncertainty module.

These tests verify the core logic without requiring API calls.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from src.measure_uncertainty import UncertaintyMeasurer


class TestUncertaintyMeasurer(unittest.TestCase):
    """Test cases for UncertaintyMeasurer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test-api-key"
        self.model = "gpt-4"
    
    @patch('src.measure_uncertainty.OpenAI')
    def test_initialization(self, mock_openai):
        """Test UncertaintyMeasurer initialization."""
        measurer = UncertaintyMeasurer(api_key=self.api_key, model=self.model)
        
        self.assertEqual(measurer.api_key, self.api_key)
        self.assertEqual(measurer.model, self.model)
        mock_openai.assert_called_once_with(api_key=self.api_key)
    
    def test_calculate_average_confidence(self):
        """Test calculation of average confidence from logprobs."""
        measurer = UncertaintyMeasurer(api_key=self.api_key, model=self.model)
        
        # Create mock logprobs data
        mock_logprobs = []
        for _ in range(3):
            mock_content = []
            for _ in range(5):  # 5 tokens
                token_logprob = Mock()
                token_logprob.logprob = -0.1  # High confidence (close to 0)
                mock_content.append(token_logprob)
            
            mock_lp = Mock()
            mock_lp.content = mock_content
            mock_logprobs.append(mock_lp)
        
        confidence = measurer._calculate_average_confidence(mock_logprobs)
        
        # Confidence should be high (close to 1) since logprob is close to 0
        self.assertGreater(confidence, 0.9)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_average_confidence_empty(self):
        """Test average confidence calculation with empty data."""
        measurer = UncertaintyMeasurer(api_key=self.api_key, model=self.model)
        
        confidence = measurer._calculate_average_confidence([])
        self.assertEqual(confidence, 0.0)
    
    def test_analyze_uncertainty_low(self):
        """Test uncertainty analysis with low diversity (confident responses)."""
        measurer = UncertaintyMeasurer(api_key=self.api_key, model=self.model)
        
        # All responses are identical - low uncertainty
        responses = ["Paris"] * 5
        
        # Mock logprobs
        mock_logprobs = [Mock(content=[Mock(logprob=-0.1)]) for _ in range(5)]
        
        analysis = measurer._analyze_uncertainty(responses, mock_logprobs)
        
        self.assertEqual(analysis["unique_responses"], 1)
        self.assertEqual(analysis["total_samples"], 5)
        self.assertEqual(analysis["uncertainty_level"], "low")
        self.assertIn("confident", analysis["recommendation"].lower())
    
    def test_analyze_uncertainty_high(self):
        """Test uncertainty analysis with high diversity (uncertain responses)."""
        measurer = UncertaintyMeasurer(api_key=self.api_key, model=self.model)
        
        # All responses are different - high uncertainty
        responses = [f"Response {i}" for i in range(5)]
        
        # Mock logprobs
        mock_logprobs = [Mock(content=[Mock(logprob=-1.0)]) for _ in range(5)]
        
        analysis = measurer._analyze_uncertainty(responses, mock_logprobs)
        
        self.assertEqual(analysis["unique_responses"], 5)
        self.assertEqual(analysis["total_samples"], 5)
        self.assertEqual(analysis["uncertainty_level"], "high")
        self.assertIn("uncertain", analysis["recommendation"].lower())
    
    def test_analyze_uncertainty_medium(self):
        """Test uncertainty analysis with medium diversity."""
        measurer = UncertaintyMeasurer(api_key=self.api_key, model=self.model)
        
        # Some responses are the same - medium uncertainty
        responses = ["Answer A", "Answer A", "Answer B", "Answer C", "Answer A"]
        
        # Mock logprobs
        mock_logprobs = [Mock(content=[Mock(logprob=-0.5)]) for _ in range(5)]
        
        analysis = measurer._analyze_uncertainty(responses, mock_logprobs)
        
        self.assertEqual(analysis["unique_responses"], 3)
        self.assertEqual(analysis["total_samples"], 5)
        self.assertEqual(analysis["uncertainty_level"], "medium")
    
    def test_analyze_uncertainty_with_none_responses(self):
        """Test uncertainty analysis when some responses are None."""
        measurer = UncertaintyMeasurer(api_key=self.api_key, model=self.model)
        
        # Some responses failed (None)
        responses = ["Answer A", None, "Answer A", None, "Answer B"]
        mock_logprobs = [Mock(content=[Mock(logprob=-0.1)]), None, 
                         Mock(content=[Mock(logprob=-0.1)]), None,
                         Mock(content=[Mock(logprob=-0.1)])]
        
        analysis = measurer._analyze_uncertainty(responses, mock_logprobs)
        
        self.assertEqual(analysis["total_samples"], 3)  # Only valid responses
        self.assertEqual(analysis["unique_responses"], 2)
    
    def test_analyze_uncertainty_all_failed(self):
        """Test uncertainty analysis when all responses failed."""
        measurer = UncertaintyMeasurer(api_key=self.api_key, model=self.model)
        
        responses = [None, None, None]
        mock_logprobs = [None, None, None]
        
        analysis = measurer._analyze_uncertainty(responses, mock_logprobs)
        
        self.assertIn("error", analysis)
        self.assertEqual(analysis["uncertainty_level"], "unknown")
    
    def test_format_results(self):
        """Test formatting of results."""
        measurer = UncertaintyMeasurer(api_key=self.api_key, model=self.model)
        
        results = {
            "prompt": "Test prompt",
            "num_samples": 5,
            "responses": ["Response 1", "Response 2"],
            "logprobs": [Mock(), Mock()],
            "uncertainty_analysis": {
                "unique_responses": 2,
                "total_samples": 2,
                "response_diversity": 1.0,
                "average_token_confidence": 0.95,
                "uncertainty_level": "high",
                "recommendation": "Test recommendation"
            }
        }
        
        formatted = measurer.format_results(results)
        formatted_upper = formatted.upper()
        
        # Check that key information is in the formatted output
        self.assertIn("TEST PROMPT", formatted_upper)
        self.assertIn("HIGH", formatted_upper)
        self.assertIn("RESPONSE 1", formatted_upper)
    
    def test_calculate_mean_logprob(self):
        """Test calculation of mean log probability."""
        measurer = UncertaintyMeasurer(api_key=self.api_key, model=self.model)
        
        # Create mock logprobs data
        mock_logprobs = []
        for _ in range(3):
            mock_content = []
            for _ in range(5):  # 5 tokens
                token_logprob = Mock()
                token_logprob.logprob = -0.5
                mock_content.append(token_logprob)
            
            mock_lp = Mock()
            mock_lp.content = mock_content
            mock_logprobs.append(mock_lp)
        
        mean_logprob = measurer._calculate_mean_logprob(mock_logprobs)
        
        # Mean should be -0.5
        self.assertAlmostEqual(mean_logprob, -0.5, places=5)
    
    def test_calculate_mean_logprob_empty(self):
        """Test mean logprob calculation with empty data."""
        measurer = UncertaintyMeasurer(api_key=self.api_key, model=self.model)
        
        mean_logprob = measurer._calculate_mean_logprob([])
        self.assertEqual(mean_logprob, 0.0)
    
    @patch('src.measure_uncertainty.OpenAI')
    def test_uncertainty_ratio_logic(self, mock_openai):
        """Test the uncertainty ratio calculation and decision logic."""
        # Create mock responses
        def create_mock_response(logprob_value):
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = "Test response"
            
            mock_logprobs = Mock()
            mock_content = []
            for _ in range(5):  # 5 tokens
                token_logprob = Mock()
                token_logprob.logprob = logprob_value
                mock_content.append(token_logprob)
            
            mock_logprobs.content = mock_content
            mock_choice.logprobs = mock_logprobs
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response
        
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        measurer = UncertaintyMeasurer(api_key=self.api_key, model=self.model)
        
        # Test Case 1: High confidence answer (-0.05) vs low confidence uncertainty phrases (-2.0)
        # Ratio = -2.0 / -0.05 = 40, which is > 1.0, so should be UNCERTAIN
        mock_client.chat.completions.create.side_effect = [
            create_mock_response(-0.05) for _ in range(5)  # Answer queries
        ] + [
            create_mock_response(-2.0) for _ in range(3)  # Uncertainty phrase queries
        ] + [
            create_mock_response(-0.1)  # Uncertainty message generation
        ]
        
        results = measurer.measure_uncertainty("Test question", num_samples=5, uncertainty_threshold=1.0)
        analysis = results["uncertainty_analysis"]
        
        # With corrected logic: ratio = uncertainty / answer = -2.0 / -0.05 = 40
        # 40 > 1.0, so should be uncertain
        self.assertAlmostEqual(analysis["uncertainty_ratio"], 40.0, places=1)
        self.assertTrue(analysis["is_uncertain"], "High ratio should indicate uncertainty")
        
        # Test Case 2: Similar confidence for both (-1.8 vs -2.0)
        # Ratio = -2.0 / -1.8 = 1.11, which is > 1.0, so should be UNCERTAIN
        mock_client.chat.completions.create.side_effect = [
            create_mock_response(-1.8) for _ in range(5)  # Answer queries
        ] + [
            create_mock_response(-2.0) for _ in range(3)  # Uncertainty phrase queries
        ] + [
            create_mock_response(-0.1)  # Uncertainty message generation
        ]
        
        results = measurer.measure_uncertainty("Test question", num_samples=5, uncertainty_threshold=1.0)
        analysis = results["uncertainty_analysis"]
        
        # With corrected logic: ratio = uncertainty / answer = -2.0 / -1.8 ≈ 1.11
        # 1.11 > 1.0, so should be uncertain
        self.assertGreater(analysis["uncertainty_ratio"], 1.0)
        self.assertTrue(analysis["is_uncertain"], "Ratio > 1.0 should indicate uncertainty")


class TestFunctionSchema(unittest.TestCase):
    """Test the function schema for measure_uncertainty."""
    
    @patch('src.llm_interface.OpenAI')
    def test_function_schema_structure(self, mock_openai):
        """Test that the function schema is properly structured."""
        from src.llm_interface import LLMFunctionInterface
        
        interface = LLMFunctionInterface(api_key="test-key")
        schema = interface.MEASURE_UNCERTAINTY_FUNCTION
        
        # Verify schema structure
        self.assertEqual(schema["type"], "function")
        self.assertIn("function", schema)
        
        func = schema["function"]
        self.assertEqual(func["name"], "measure_uncertainty")
        self.assertIn("description", func)
        self.assertIn("parameters", func)
        
        # Verify parameters
        params = func["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("properties", params)
        self.assertIn("required", params)
        
        # Verify required parameters
        self.assertIn("prompt", params["required"])
        
        # Verify parameter properties
        props = params["properties"]
        self.assertIn("prompt", props)
        self.assertIn("num_samples", props)
        self.assertIn("temperature", props)


if __name__ == "__main__":
    unittest.main()
