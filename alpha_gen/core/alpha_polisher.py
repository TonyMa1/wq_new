"""
Alpha polishing core functionality.
Provides tools for refining and improving alpha expressions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import time

from alpha_gen.api.wq_client import WorldQuantClient, WorldQuantError
from alpha_gen.api.ai_client import AIClient, AIClientError
# MODIFIED IMPORT: Added AlphaMetrics
from alpha_gen.models.alpha import Alpha, SimulationResult, AlphaMetrics
from alpha_gen.utils.validators import validate_alpha_expression

logger = logging.getLogger(__name__)

class AlphaPolisherError(Exception):
    """Base exception for alpha polisher errors."""
    pass

class AlphaPolisher:
    """
    Core class for polishing and refining alpha expressions.

    Combines AI-based refinement with WorldQuant Brain testing.
    """

    def __init__(
        self,
        wq_client: WorldQuantClient,
        ai_client: AIClient
    ):
        """
        Initialize alpha polisher.

        Args:
            wq_client: WorldQuant API client
            ai_client: AI client for expression refinement
        """
        self.wq_client = wq_client
        self.ai_client = ai_client

        # Cache for operators
        self._operators_cache = None

        logger.info("Alpha Polisher initialized")

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

    def polish_alpha(
        self,
        alpha: Alpha,
        user_requirements: Optional[str] = None
    ) -> Tuple[Alpha, Dict]:
        """
        Polish an alpha expression and test the result.

        Args:
            alpha: Alpha to polish
            user_requirements: Optional specific requirements for improvement

        Returns:
            Tuple of (polished Alpha, comparison results)
        """
        logger.info(f"Polishing alpha: {alpha.expression[:100]}...")

        # Get operators for context
        operators = self.get_operators()

        try:
            # Run the initial simulation on the original alpha
            logger.info("Testing original alpha")
            original_result = self.wq_client.simulate_alpha(
                expression=alpha.expression,
                settings=alpha.settings.to_api_format()
            )

            # Extract metrics from original result
            original_metrics = None
            if original_result and "alpha_details" in original_result and original_result["alpha_details"]:
                original_metrics = original_result["alpha_details"].get("is", {})

            # Polish the expression
            polished_expression = self.ai_client.polish_alpha(
                expression=alpha.expression,
                user_requirements=user_requirements,
                operators=operators
            )

            # Validate the polished expression
            is_valid, error = validate_alpha_expression(polished_expression)
            if not is_valid:
                logger.warning(f"Invalid polished expression: {error}")
                raise AlphaPolisherError(f"Invalid polished expression: {error}")

            # Create a new Alpha object for the polished expression
            polished_alpha = Alpha(
                expression=polished_expression,
                settings=alpha.settings
            )

            # Test the polished alpha
            logger.info("Testing polished alpha")
            polished_result = self.wq_client.simulate_alpha(
                expression=polished_expression,
                settings=alpha.settings.to_api_format()
            )

            # Extract metrics from polished result
            polished_metrics = None
            if polished_result and "alpha_details" in polished_result and polished_result["alpha_details"]:
                polished_metrics = polished_result["alpha_details"].get("is", {})

                # Update polished alpha with results
                alpha_details = polished_result["alpha_details"]
                polished_alpha.id = alpha_details.get("id")
                polished_alpha.status = alpha_details.get("status", "UNSUBMITTED")
                polished_alpha.grade = alpha_details.get("grade", "UNKNOWN")

                # Update metrics if available
                if "is" in alpha_details and alpha_details["is"]:
                    metrics_data = alpha_details["is"]
                    # MODIFIED: Call AlphaMetrics directly
                    polished_alpha.metrics = AlphaMetrics.from_api_format(metrics_data)

            # Create comparison results
            comparison = {
                "original": {
                    "expression": alpha.expression,
                    "result": original_result,
                    "metrics": original_metrics
                },
                "polished": {
                    "expression": polished_expression,
                    "result": polished_result,
                    "metrics": polished_metrics
                },
                "improvements": self._calculate_improvements(original_metrics, polished_metrics)
            }

            logger.info("Polishing complete")
            return polished_alpha, comparison

        except (WorldQuantError, AIClientError) as e:
            logger.error(f"Error polishing alpha: {str(e)}")
            raise AlphaPolisherError(f"Error polishing alpha: {str(e)}")

    def analyze_alpha(
        self,
        alpha: Alpha,
        include_metrics: bool = True
    ) -> Dict[str, str]:
        """
        Analyze an alpha expression and provide insights.

        Args:
            alpha: Alpha to analyze
            include_metrics: Whether to include metrics in analysis

        Returns:
            Dictionary with analysis sections
        """
        logger.info(f"Analyzing alpha: {alpha.expression[:100]}...")

        # Get operators for context
        operators = self.get_operators()

        # Get metrics if requested and not already available
        metrics = None
        if include_metrics:
            if alpha.metrics:
                # Use existing metrics
                metrics = {
                    "sharpe": alpha.metrics.sharpe,
                    "fitness": alpha.metrics.fitness,
                    "turnover": alpha.metrics.turnover,
                    "returns": alpha.metrics.returns,
                    "long_count": alpha.metrics.long_count,
                    "short_count": alpha.metrics.short_count
                }
            else:
                # Get metrics from simulation
                try:
                    logger.info("Running simulation to get metrics")
                    result = self.wq_client.simulate_alpha(
                        expression=alpha.expression,
                        settings=alpha.settings.to_api_format()
                    )

                    if result and "alpha_details" in result and result["alpha_details"]:
                        alpha_details = result["alpha_details"]
                        metrics_data = alpha_details.get("is", {})

                        metrics = {
                            "sharpe": metrics_data.get("sharpe", 0),
                            "fitness": metrics_data.get("fitness"),
                            "turnover": metrics_data.get("turnover", 0),
                            "returns": metrics_data.get("returns", 0),
                            "long_count": metrics_data.get("longCount", 0),
                            "short_count": metrics_data.get("shortCount", 0)
                        }

                        # Update alpha with results
                        alpha.id = alpha_details.get("id")
                        alpha.status = alpha_details.get("status", "UNSUBMITTED")
                        alpha.grade = alpha_details.get("grade", "UNKNOWN")

                        # Update metrics
                        # MODIFIED: Call AlphaMetrics directly
                        alpha.metrics = AlphaMetrics.from_api_format(metrics_data)

                except WorldQuantError as e:
                    logger.warning(f"Failed to get metrics: {str(e)}")

        try:
            # Analyze the expression
            analysis = self.ai_client.analyze_alpha(
                expression=alpha.expression,
                operators=operators,
                metrics=metrics
            )

            logger.info("Analysis complete")
            return analysis

        except AIClientError as e:
            logger.error(f"Error analyzing alpha: {str(e)}")
            raise AlphaPolisherError(f"Error analyzing alpha: {str(e)}")

    def _calculate_improvements(
        self,
        original_metrics: Optional[Dict],
        polished_metrics: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Calculate improvements between original and polished alphas.

        Args:
            original_metrics: Metrics from original alpha
            polished_metrics: Metrics from polished alpha

        Returns:
            Dictionary of improvement metrics
        """
        if not original_metrics or not polished_metrics:
            return {}

        improvements = {}

        # Calculate improvements for key metrics
        for key in ['sharpe', 'fitness', 'turnover', 'returns']:
            if key in original_metrics and key in polished_metrics:
                original_value = original_metrics.get(key)
                polished_value = polished_metrics.get(key)

                if original_value is not None and polished_value is not None:
                    try:
                        # Attempt float conversion for calculation
                        orig_f = float(original_value)
                        pol_f = float(polished_value)

                        if abs(orig_f) > 1e-9: # Avoid division by zero or near-zero
                             pct_change = ((pol_f - orig_f) / abs(orig_f)) * 100
                             improvements[f"{key}_change_pct"] = pct_change

                        improvements[f"{key}_change"] = pol_f - orig_f

                        # Simplified improvement check logic
                        improved = False
                        if key in ['sharpe', 'fitness', 'returns']:
                            improved = pol_f > orig_f
                        elif key == 'turnover':
                            # Consider turnover improved if it moves into or stays within the ideal range [0.01, 0.7]
                            # Also handle edge cases where one value might be None/invalid after float conversion
                            orig_in_range = 0.01 <= orig_f <= 0.7
                            pol_in_range = 0.01 <= pol_f <= 0.7
                            improved = (not orig_in_range and pol_in_range) or \
                                       (orig_in_range and pol_in_range and abs(pol_f - orig_f) < 0.1) # Minor changes within range ok
                        improvements[f"{key}_improved"] = improved

                    except (ValueError, TypeError, ZeroDivisionError) as calc_err:
                        logger.debug(f"Could not calculate improvement for '{key}': {calc_err}")
                        pass # Ignore if conversion or calculation fails


        # Calculate overall improvement based on successfully calculated individual improvements
        improvements["overall_improved"] = (
            improvements.get("sharpe_improved", False) or
            improvements.get("fitness_improved", False) # Add other key metrics if needed
            # improvements.get("turnover_improved", False) # Turnover improvement is subjective
        )

        return improvements