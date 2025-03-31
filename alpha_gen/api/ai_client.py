"""
AI client for generating, polishing, and analyzing alpha expressions.
Unified interface for OpenRouter-based AI model interaction.
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Union, Any
import requests
from requests.exceptions import RequestException, Timeout

# Configure module logger
logger = logging.getLogger(__name__)

class AIClientError(Exception):
    """Base exception class for AI client errors."""
    pass

class AIRequestError(AIClientError):
    """Raised when a request to the AI model fails."""
    pass

class AIResponseError(AIClientError):
    """Raised when the AI model returns an invalid response."""
    pass

class AIClient:
    """
    Client for AI model interactions through OpenRouter.
    
    This client handles all AI-powered operations:
    - Alpha expression generation
    - Alpha expression refinement/polishing
    - Alpha expression analysis
    """
    
    # OpenRouter API details
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 90
    ):
        """
        Initialize AI client with API key and model configuration.
        
        Args:
            api_key: OpenRouter API key (default: from OPENROUTER_API_KEY env var)
            model: Model identifier (default: from OPENROUTER_MODEL env var or fallback)
            site_url: Site URL for OpenRouter attribution
            site_name: Site name for OpenRouter attribution
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise AIClientError(
                "OpenRouter API key not provided. Either pass it as a parameter "
                "or set the OPENROUTER_API_KEY environment variable."
            )
        
        # Default to environment variable, then fallback to Gemini 2.5 Pro
        self.model = model or os.environ.get(
            "OPENROUTER_MODEL", 
            "google/gemini-2.5-pro-exp-03-25:free"
        )
        
        # Attribution information for OpenRouter
        self.site_url = site_url or os.environ.get("OPENROUTER_SITE_URL", "http://localhost")
        self.site_name = site_name or os.environ.get("OPENROUTER_SITE_NAME", "WorldQuantAlphaGen")
        
        self.max_retries = max_retries
        self.timeout = timeout
        
        logger.info(f"AI Client initialized with model: {self.model}")
    
    def _make_request(
        self, 
        prompt: str, 
        temperature: float = 0.5,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Make a request to the OpenRouter API.
        
        Args:
            prompt: The prompt to send to the model
            temperature: The sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Model response as text
            
        Raises:
            AIRequestError: If the request fails
            AIResponseError: If the response cannot be parsed
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making request to OpenRouter (attempt {attempt + 1}/{self.max_retries})")
                
                response = requests.post(
                    self.BASE_URL,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    error_msg = f"OpenRouter API error (status: {response.status_code})"
                    try:
                        error_details = response.json()
                        error_msg += f", details: {error_details}"
                    except ValueError:
                        error_msg += f", response: {response.text[:200]}"
                    
                    if attempt < self.max_retries - 1:
                        logger.warning(f"{error_msg}. Retrying...")
                        continue
                    else:
                        raise AIRequestError(error_msg)
                
                try:
                    response_data = response.json()
                    if "choices" not in response_data or not response_data["choices"]:
                        raise AIResponseError("No choices in response")
                    
                    content = response_data["choices"][0]["message"]["content"]
                    logger.debug("Successfully received AI response")
                    return content
                    
                except (ValueError, KeyError) as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Could not parse response: {str(e)}. Retrying...")
                        continue
                    else:
                        raise AIResponseError(f"Could not parse AI response: {str(e)}")
                
            except Timeout:
                if attempt < self.max_retries - 1:
                    logger.warning("Request timed out. Retrying...")
                    continue
                else:
                    raise AIRequestError("Request timed out after multiple attempts")
                    
            except RequestException as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Request failed: {str(e)}. Retrying...")
                    continue
                else:
                    raise AIRequestError(f"Request failed: {str(e)}")
    
    def _clean_expression(self, text: str) -> str:
        """
        Clean an alpha expression extracted from AI response.
        
        Args:
            text: Raw text from AI response
            
        Returns:
            Cleaned alpha expression
        """
        # Remove code block markers if present
        text = re.sub(r'^```(fast)?expr\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'```$', '', text).strip()
        
        # Remove leading numbering or bullets
        text = re.sub(r'^\d+\.\s*', '', text)
        text = re.sub(r'^-\s*', '', text)
        
        # If there are multiple lines, try to extract just the expression
        lines = text.split('\n')
        if len(lines) > 1:
            # Try to find the line most likely to be the expression
            for line in lines:
                line = line.strip()
                if '(' in line and ')' in line and len(line.split()) > 1:
                    return line
            
            # If we couldn't identify a single line, return the last non-empty line
            for line in reversed(lines):
                line = line.strip()
                if line:
                    return line
        
        return text.strip()
    
    def _extract_expressions(self, text: str) -> List[str]:
        """
        Extract multiple alpha expressions from AI response.
        
        Args:
            text: Raw text from AI response
            
        Returns:
            List of cleaned alpha expressions
        """
        # Remove code block markers
        text = re.sub(r'```(fast)?expr\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'```', '', text)
        
        # Split by lines and clean each line
        lines = text.split('\n')
        expressions = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and explanatory text
            if not line or line.startswith('#') or not ('(' in line and ')' in line):
                continue
                
            # Remove line numbering and bullet points
            line = re.sub(r'^\d+\.\s*', '', line)
            line = re.sub(r'^-\s*', '', line)
            
            if line:
                expressions.append(line)
        
        return expressions
    
    def generate_alpha(
        self, 
        operators: List[Dict],
        data_fields: List[Dict],
        strategy_type: Optional[str] = None,
        data_field_focus: Optional[List[str]] = None,
        complexity: Optional[str] = None,
        count: int = 5
    ) -> List[str]:
        """
        Generate alpha expressions using AI.
        
        Args:
            operators: List of available operators
            data_fields: List of available data fields
            strategy_type: Optional type of strategy to focus on
            data_field_focus: Optional list of data fields to focus on
            complexity: Optional complexity level ('simple', 'moderate', 'complex')
            count: Number of alpha expressions to generate
            
        Returns:
            List of generated alpha expressions
        """
        # Prepare operator information for the prompt
        operator_by_category = {}
        for op in operators:
            category = op.get('category', 'Uncategorized')
            if category not in operator_by_category:
                operator_by_category[category] = []
            operator_by_category[category].append({
                'name': op.get('name', 'N/A'),
                'type': op.get('type', 'SCALAR'),
                'definition': op.get('definition', 'N/A'),
                'description': op.get('description', 'N/A')
            })
        
        # Prepare data field information
        field_ids = [field.get('id', 'N/A') for field in data_fields[:min(30, len(data_fields))]]
        
        # Prepare focus fields if provided
        focus_fields_text = ""
        if data_field_focus:
            focus_fields_text = f"\nPlease focus on using these data fields: {', '.join(data_field_focus)}"
        
        # Prepare strategy type text if provided
        strategy_text = ""
        if strategy_type:
            strategy_text = f"\nPlease create alphas focusing on {strategy_type} strategies."
        
        # Prepare complexity text if provided
        complexity_text = ""
        if complexity:
            complexity_text = f"\nThe complexity level should be {complexity}."
        
        # Build prompt for the AI model
        prompt = f"""Generate {count} unique alpha factor expressions using the available operators and data fields for the WorldQuant Brain platform (FASTEXPR language). Return ONLY the expressions, one per line, with no comments, explanations, or markdown formatting like backticks.

Available Data Fields (sample):
{field_ids}

Available Operators by Category (sample):
"""
        
        # Add operator information for each category
        for category, ops in operator_by_category.items():
            if len(ops) > 10:
                ops = ops[:10]  # Limit number of operators per category to keep prompt size manageable
            
            prompt += f"\n{category}:\n"
            for op in ops:
                prompt += f"- {op['name']} ({op['type']}): {op['description']}\n"
        
        # Add requirements
        prompt += f"""
Requirements:
1. Create potentially profitable alpha factors.
2. Use the provided operators and data fields, respecting operator types (SCALAR, VECTOR, MATRIX).
3. Combine multiple operators (ts_, rank, zscore, arithmetic, logical, vector, group, etc.).
4. Ensure expressions are syntactically valid for FASTEXPR.{strategy_text}{focus_fields_text}{complexity_text}

Tips:
- Common fields: 'open', 'high', 'low', 'close', 'volume', 'returns', 'vwap', 'cap'.
- Use 'rank' or 'zscore' for normalization.
- Use time series operators like 'ts_mean', 'ts_std_dev', 'ts_rank', 'ts_delta' with lookback windows (e.g., 5, 10, 20, 60).
- Use 'ts_corr' or 'ts_covariance' for relationship-based factors.

Generate {count} distinct FASTEXPR expressions now:
"""
        
        logger.info(f"Generating {count} alpha expressions...")
        
        try:
            response = self._make_request(prompt, temperature=0.7)
            expressions = self._extract_expressions(response)
            
            logger.info(f"Successfully generated {len(expressions)} expressions")
            return expressions
            
        except (AIRequestError, AIResponseError) as e:
            logger.error(f"Error generating alpha expressions: {str(e)}")
            raise
    
    def polish_alpha(
        self, 
        expression: str,
        user_requirements: Optional[str] = None,
        operators: Optional[List[Dict]] = None
    ) -> str:
        """
        Polish an existing alpha expression.
        
        Args:
            expression: The alpha expression to polish
            user_requirements: Optional specific requirements for improvement
            operators: Optional list of available operators
            
        Returns:
            Polished alpha expression
        """
        # Prepare operator information if provided
        operators_text = ""
        if operators:
            # Group operators by category
            operator_by_category = {}
            for op in operators[:100]:  # Limit to avoid excessive prompt size
                category = op.get('category', 'Uncategorized')
                if category not in operator_by_category:
                    operator_by_category[category] = []
                operator_by_category[category].append({
                    'name': op.get('name', 'N/A'),
                    'type': op.get('type', 'SCALAR'),
                    'definition': op.get('definition', 'N/A'),
                    'description': op.get('description', 'N/A')
                })
            
            operators_text = "Available Operators by Category (sample):\n"
            for category, ops in operator_by_category.items():
                operators_text += f"\n{category}:\n"
                for op in ops[:5]:  # Limit operators per category
                    operators_text += f"- {op['name']} ({op['type']}): {op['description']}\n"
        
        # Prepare user requirements text if provided
        requirements_text = ""
        if user_requirements:
            requirements_text = f"\nConsider these specific requirements:\n{user_requirements}"
        
        # Build prompt for the AI model
        prompt = f"""You are an expert quantitative analyst specializing in improving WorldQuant Brain alpha expressions (FASTEXPR language).

{operators_text}

Please carefully polish the following WorldQuant alpha expression to potentially improve its performance (e.g., Sharpe ratio, fitness, IR) while maintaining its core strategic idea.

Original Expression:
{expression}

Make thoughtful changes, such as:
1. Adjusting parameters (lookback periods, thresholds)
2. Adding normalization (rank, zscore) to improve stability
3. Applying smoothing (ts_mean) to reduce noise
4. Handling outliers (winsorize)
5. Combining with complementary factors (volume, volatility)
6. Improving cross-sectional ranking if appropriate

Return ONLY the single, complete, polished FASTEXPR expression. Do not include explanations, comments, backticks, markdown formatting, or any other text.
"""
        
        logger.info(f"Polishing alpha expression: {expression[:100]}...")
        
        try:
            response = self._make_request(prompt, temperature=0.4)
            polished_expression = self._clean_expression(response)
            
            logger.info(f"Successfully polished expression: {polished_expression[:100]}")
            return polished_expression
            
        except (AIRequestError, AIResponseError) as e:
            logger.error(f"Error polishing alpha expression: {str(e)}")
            raise
    
    def analyze_alpha(
        self, 
        expression: str,
        operators: Optional[List[Dict]] = None,
        metrics: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Analyze an alpha expression and provide insights.
        
        Args:
            expression: The alpha expression to analyze
            operators: Optional list of available operators
            metrics: Optional performance metrics for the alpha
            
        Returns:
            Dictionary with analysis sections
        """
        # Prepare operator information if provided
        operators_text = ""
        if operators:
            operator_by_category = {}
            for op in operators[:100]:  # Limit to avoid excessive prompt size
                category = op.get('category', 'Uncategorized')
                if category not in operator_by_category:
                    operator_by_category[category] = []
                operator_by_category[category].append(op.get('name', 'N/A'))
            
            operators_text = "Available WorldQuant operators by category:\n"
            for category, ops in operator_by_category.items():
                operators_text += f"\n{category}: {', '.join(ops[:10])}"  # Limit operators per category
        
        # Prepare metrics text if provided
        metrics_text = ""
        if metrics:
            metrics_text = "\nPerformance metrics for this alpha:\n"
            for key, value in metrics.items():
                metrics_text += f"- {key}: {value}\n"
        
        # Build prompt for the AI model
        prompt = f"""You are an expert quantitative analyst specializing in WorldQuant Brain alpha expressions (FASTEXPR language).

{operators_text}

Please analyze this WorldQuant alpha expression:

{expression}

{metrics_text}

Provide a concise analysis covering these points:
1. **Strategy/Inefficiency:** What market pattern, anomaly, or strategy does this alpha likely try to capture?
2. **Key Components:** Break down the expression. What is the role of each main operator and data field?
3. **Potential Strengths:** What might make this alpha perform well under certain market conditions?
4. **Potential Risks/Limitations:** What are the potential downsides, risks, or scenarios where it might fail?
5. **Validity Check:** Is the expression syntactically valid? Are there any obvious errors?
6. **Improvement Suggestion:** Offer one specific, actionable suggestion for potentially improving the alpha.

Keep the analysis clear and focused on these points.
"""
        
        logger.info(f"Analyzing alpha expression: {expression[:100]}...")
        
        try:
            response = self._make_request(prompt, temperature=0.3)
            
            # Extract analysis sections
            analysis = {
                "strategy": "",
                "components": "",
                "strengths": "",
                "risks": "",
                "validity": "",
                "improvement": "",
                "full_text": response
            }
            
            # Try to extract sections using pattern matching
            strategy_match = re.search(r'(?:Strategy/Inefficiency:?|This alpha (?:likely )?(?:aims to|captures|tries to)):?\s*(.*?)(?:\n\n|\n\d\.|\n\*\*)', response, re.DOTALL | re.IGNORECASE)
            if strategy_match:
                analysis["strategy"] = strategy_match.group(1).strip()
            
            components_match = re.search(r'(?:Key Components:?|Components:?|Breakdown:?):?\s*(.*?)(?:\n\n|\n\d\.|\n\*\*)', response, re.DOTALL | re.IGNORECASE)
            if components_match:
                analysis["components"] = components_match.group(1).strip()
            
            strengths_match = re.search(r'(?:Potential Strengths:?|Strengths:?):?\s*(.*?)(?:\n\n|\n\d\.|\n\*\*)', response, re.DOTALL | re.IGNORECASE)
            if strengths_match:
                analysis["strengths"] = strengths_match.group(1).strip()
            
            risks_match = re.search(r'(?:Potential Risks/Limitations:?|Risks:?|Limitations:?):?\s*(.*?)(?:\n\n|\n\d\.|\n\*\*)', response, re.DOTALL | re.IGNORECASE)
            if risks_match:
                analysis["risks"] = risks_match.group(1).strip()
            
            validity_match = re.search(r'(?:Validity Check:?|Syntax Check:?):?\s*(.*?)(?:\n\n|\n\d\.|\n\*\*)', response, re.DOTALL | re.IGNORECASE)
            if validity_match:
                analysis["validity"] = validity_match.group(1).strip()
            
            improvement_match = re.search(r'(?:Improvement Suggestion:?|Suggestions?:?):?\s*(.*?)(?:\n\n|\n\d\.|\n\*\*|$)', response, re.DOTALL | re.IGNORECASE)
            if improvement_match:
                analysis["improvement"] = improvement_match.group(1).strip()
            
            logger.info("Successfully analyzed alpha expression")
            return analysis
            
        except (AIRequestError, AIResponseError) as e:
            logger.error(f"Error analyzing alpha expression: {str(e)}")
            raise