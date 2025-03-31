"""
Alpha generation core functionality.
Provides tools for creating and testing alpha expressions.
"""

import os
import json
import logging
import time
import random
from typing import Dict, List, Optional, Tuple, Set, Any
import concurrent.futures
from datetime import datetime

from alpha_gen.api.wq_client import WorldQuantClient, WorldQuantError
from alpha_gen.api.ai_client import AIClient, AIClientError
from alpha_gen.models.alpha import Alpha, SimulationResult, SimulationSettings
from alpha_gen.utils.validators import (
    validate_alpha_expression,
    extract_symbols_from_expression,
    extract_parameters_from_expression,
    create_expression_variant
)

logger = logging.getLogger(__name__)

class AlphaGeneratorError(Exception):
    """Base exception for alpha generator errors."""
    pass

class AlphaGenerator:
    """
    Core class for generating and testing alpha expressions.
    
    Combines AI-based generation with WorldQuant Brain testing.
    """
    
    def __init__(
        self,
        wq_client: WorldQuantClient,
        ai_client: AIClient,
        output_dir: str = "./output",
        max_concurrent_simulations: int = 5
    ):
        """
        Initialize alpha generator.
        
        Args:
            wq_client: WorldQuant API client
            ai_client: AI client for expression generation
            output_dir: Directory for saving results
            max_concurrent_simulations: Maximum number of concurrent simulations
        """
        self.wq_client = wq_client
        self.ai_client = ai_client
        self.output_dir = output_dir
        self.max_concurrent_simulations = max_concurrent_simulations
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Cache for operators and data fields
        self._operators_cache = None
        self._data_fields_cache = {}  # Keyed by (region, universe)
        
        logger.info("Alpha Generator initialized")
    
    def get_operators(self) -> List[Dict]:
        """
        Get available operators from WorldQuant Brain.
        Uses caching to avoid repeated API calls.
        
        Returns:
            List of operator objects
        """
        if self._operators_cache is None:
            try:
                logger.info("Fetching operators from WorldQuant Brain")
                self._operators_cache = self.wq_client.get_operators()
                logger.info(f"Fetched {len(self._operators_cache)} operators")
            except WorldQuantError as e:
                logger.error(f"Failed to fetch operators: {str(e)}")
                self._operators_cache = []
        
        return self._operators_cache
    
    def get_data_fields(
        self,
        region: str = 'USA',
        universe: str = 'TOP3000',
        refresh: bool = False
    ) -> List[Dict]:
        """
        Get available data fields from WorldQuant Brain.
        Uses caching to avoid repeated API calls.
        
        Args:
            region: Region code
            universe: Universe name
            refresh: Force refresh from API
            
        Returns:
            List of data field objects
        """
        cache_key = (region, universe)
        
        if refresh or cache_key not in self._data_fields_cache:
            try:
                logger.info(f"Fetching data fields for {region}/{universe}")
                self._data_fields_cache[cache_key] = self.wq_client.get_data_fields(
                    region=region,
                    universe=universe
                )
                logger.info(f"Fetched {len(self._data_fields_cache[cache_key])} data fields")
            except WorldQuantError as e:
                logger.error(f"Failed to fetch data fields: {str(e)}")
                self._data_fields_cache[cache_key] = []
        
        return self._data_fields_cache[cache_key]
    
    def generate_expressions(
        self,
        region: str = 'USA',
        universe: str = 'TOP3000',
        strategy_type: Optional[str] = None,
        data_field_focus: Optional[List[str]] = None,
        complexity: Optional[str] = None,
        count: int = 5
    ) -> List[Alpha]:
        """
        Generate alpha expressions using AI.
        
        Args:
            region: Region code
            universe: Universe name
            strategy_type: Optional type of strategy to focus on
            data_field_focus: Optional list of data fields to focus on
            complexity: Optional complexity level
            count: Number of expressions to generate
            
        Returns:
            List of Alpha objects
        """
        # Get operators and data fields
        operators = self.get_operators()
        data_fields = self.get_data_fields(region, universe)
        
        if not operators:
            logger.error("No operators available for generation")
            raise AlphaGeneratorError("No operators available")
        
        if not data_fields:
            logger.error(f"No data fields available for {region}/{universe}")
            raise AlphaGeneratorError(f"No data fields available for {region}/{universe}")
        
        try:
            # Generate expressions
            logger.info(f"Generating {count} expressions with AI")
            expressions = self.ai_client.generate_alpha(
                operators=operators,
                data_fields=data_fields,
                strategy_type=strategy_type,
                data_field_focus=data_field_focus,
                complexity=complexity,
                count=count
            )
            
            # Create Alpha objects
            alphas = []
            for expr in expressions:
                try:
                    # Validate expression
                    is_valid, error = validate_alpha_expression(expr)
                    if not is_valid:
                        logger.warning(f"Invalid expression generated: {expr} - {error}")
                        continue
                    
                    # Create Alpha object
                    alpha = Alpha(
                        expression=expr,
                        settings=Alpha.SimulationSettings(
                            universe=universe
                        )
                    )
                    alphas.append(alpha)
                    
                except Exception as e:
                    logger.error(f"Error creating Alpha object for {expr}: {str(e)}")
            
            logger.info(f"Generated {len(alphas)} valid expressions")
            return alphas
            
        except AIClientError as e:
            logger.error(f"AI generation failed: {str(e)}")
            raise AlphaGeneratorError(f"AI generation failed: {str(e)}")
    
    def test_expressions(
        self,
        alphas: List[Alpha],
        save_results: bool = True,
        exclude_failures: bool = True
    ) -> List[Tuple[Alpha, Dict]]:
        """
        Test multiple alpha expressions by simulation.
        
        Args:
            alphas: List of Alpha objects to test
            save_results: Whether to save results to disk
            exclude_failures: Whether to exclude failed simulations from results
            
        Returns:
            List of (Alpha, result) tuples
        """
        if not alphas:
            logger.warning("No alphas provided for testing")
            return []
        
        logger.info(f"Testing {len(alphas)} expressions")
        results = []
        failed = []
        
        # Create a thread pool for concurrent simulations
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_simulations) as executor:
            # Submit all simulations
            future_to_alpha = {
                executor.submit(self.test_expression, alpha): alpha for alpha in alphas
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_alpha):
                alpha = future_to_alpha[future]
                try:
                    result = future.result()
                    if result is None:
                        failed.append(alpha)
                        continue
                    
                    results.append((alpha, result))
                    logger.info(f"Completed simulation for {alpha.expression[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Error in simulation for {alpha.expression[:50]}...: {str(e)}")
                    failed.append(alpha)
        
        # Save results if requested
        if save_results and results:
            timestamp = int(time.time())
            results_file = os.path.join(self.output_dir, f"alpha_results_{timestamp}.json")
            
            try:
                result_data = []
                for alpha, result in results:
                    # Combine alpha and result data
                    entry = {
                        "expression": alpha.expression,
                        "simulation_result": result
                    }
                    
                    # Add metrics if available
                    if "alpha_details" in result and result["alpha_details"]:
                        entry["metrics"] = result["alpha_details"].get("is", {})
                    
                    result_data.append(entry)
                
                with open(results_file, 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                logger.info(f"Saved results to {results_file}")
                
            except Exception as e:
                logger.error(f"Failed to save results: {str(e)}")
        
        # Report failures
        if failed:
            logger.warning(f"{len(failed)} simulations failed")
        
        return results
    
    def test_expression(self, alpha: Alpha) -> Optional[Dict]:
        """
        Test a single alpha expression by simulation.
        
        Args:
            alpha: Alpha object to test
            
        Returns:
            Simulation result dictionary or None if failed
        """
        try:
            # Convert alpha settings to API format
            settings = alpha.settings.to_api_format()
            
            # Run simulation
            result = self.wq_client.simulate_alpha(
                expression=alpha.expression,
                settings=settings
            )
            
            # Update alpha with results if available
            if "alpha_details" in result and result["alpha_details"]:
                alpha_details = result["alpha_details"]
                alpha.id = alpha_details.get("id")
                alpha.status = alpha_details.get("status", "UNSUBMITTED")
                alpha.grade = alpha_details.get("grade", "UNKNOWN")
                
                # Update metrics if available
                if "is" in alpha_details and alpha_details["is"]:
                    metrics_data = alpha_details["is"]
                    alpha.metrics = alpha.metrics_class.from_api_format(metrics_data)
            
            return result
            
        except WorldQuantError as e:
            logger.error(f"Simulation failed for {alpha.expression[:50]}...: {str(e)}")
            return None
    
    def generate_parameter_variations(
        self,
        base_expression: str,
        value_range_percent: float = 0.5,  # ±50%
        min_values_per_param: int = 3,
        max_values_per_param: int = 5,
        max_variations: int = 20
    ) -> List[str]:
        """
        Generate variations of an alpha expression by varying parameters.
        
        Args:
            base_expression: Original expression
            value_range_percent: Range around original values (as percentage)
            min_values_per_param: Minimum number of values to try per parameter
            max_values_per_param: Maximum number of values to try per parameter
            max_variations: Maximum total variations to generate
            
        Returns:
            List of expression variations
        """
        # Extract numeric parameters
        parameters = extract_parameters_from_expression(base_expression)
        
        if not parameters:
            logger.warning(f"No numeric parameters found in expression: {base_expression}")
            return [base_expression]
        
        logger.info(f"Found {len(parameters)} parameters in expression")
        
        # Generate variations for each parameter
        param_values = []
        for value, _, _ in parameters:
            # Calculate range
            min_value = max(1, int(value * (1 - value_range_percent)))
            max_value = int(value * (1 + value_range_percent))
            
            # Generate more values for small original values
            num_values = min_values_per_param
            if value > 20:
                num_values = max_values_per_param
            
            # Create evenly spaced values
            if max_value - min_value >= num_values:
                step = max(1, (max_value - min_value) // (num_values - 1))
                values = list(range(min_value, max_value + 1, step))
            else:
                values = list(range(min_value, max_value + 1))
            
            # Ensure original value is included
            if value not in values:
                values.append(value)
                values.sort()
            
            param_values.append(values)
        
        # Generate combinations
        positions = [(start, end) for _, start, end in parameters]
        variation_count = 1
        for values in param_values:
            variation_count *= len(values)
        
        # Limit number of variations if too many
        if variation_count > max_variations:
            logger.warning(f"Too many variations ({variation_count}), limiting to {max_variations}")
            
            # Reduce number of values per parameter
            while variation_count > max_variations and any(len(values) > 2 for values in param_values):
                # Find parameter with most values
                max_index = max(range(len(param_values)), key=lambda i: len(param_values[i]))
                
                # Remove a value (not the original)
                original_value = parameters[max_index][0]
                values = param_values[max_index]
                if len(values) > 2:
                    # Remove value furthest from original that isn't original
                    non_original = [v for v in values if v != original_value]
                    if non_original:
                        furthest = max(non_original, key=lambda v: abs(v - original_value))
                        values.remove(furthest)
                        param_values[max_index] = values
                
                # Recalculate variation count
                variation_count = 1
                for values in param_values:
                    variation_count *= len(values)
        
        # Generate variations
        variations = []
        
        # Add original expression
        variations.append(base_expression)
        
        # Function to recursively generate variations
        def generate_variations_recursive(params_so_far, param_index):
            if param_index >= len(param_values):
                # Generate expression with these parameters
                if params_so_far != [parameters[i][0] for i in range(len(parameters))]:
                    variation = create_expression_variant(base_expression, positions, params_so_far)
                    variations.append(variation)
                return
            
            for value in param_values[param_index]:
                new_params = params_so_far + [value]
                generate_variations_recursive(new_params, param_index + 1)
        
        # Start recursive generation
        generate_variations_recursive([], 0)
        
        # Apply limit if still too many
        if len(variations) > max_variations:
            variations = variations[:max_variations]
        
        logger.info(f"Generated {len(variations)} variations")
        return variations
    
    def save_alphas(self, alphas: List[Alpha], filename: Optional[str] = None) -> str:
        """
        Save alpha objects to a JSON file.
        
        Args:
            alphas: List of Alpha objects to save
            filename: Optional filename (default: generated based on timestamp)
            
        Returns:
            Path to saved file
        """
        if not alphas:
            logger.warning("No alphas to save")
            return ""
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"alphas_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            data = []
            for alpha in alphas:
                alpha_data = json.loads(alpha.to_json())
                data.append(alpha_data)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(alphas)} alphas to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save alphas: {str(e)}")
            return ""
    
    def load_alphas(self, filepath: str) -> List[Alpha]:
        """
        Load alpha objects from a JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of Alpha objects
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return []
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            alphas = []
            for item in data:
                try:
                    # Convert to JSON string and parse with Alpha.from_json
                    alpha_json = json.dumps(item)
                    alpha = Alpha.from_json(alpha_json)
                    alphas.append(alpha)
                except Exception as e:
                    logger.error(f"Failed to parse alpha: {str(e)}")
            
            logger.info(f"Loaded {len(alphas)} alphas from {filepath}")
            return alphas
            
        except Exception as e:
            logger.error(f"Failed to load alphas: {str(e)}")
            return []