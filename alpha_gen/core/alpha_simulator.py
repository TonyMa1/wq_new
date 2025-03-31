"""
Alpha simulation management.
Provides tools for batch simulation and monitoring.
"""

import logging
import time
import os
import json
from typing import Dict, List, Optional, Tuple, Any
import concurrent.futures
from datetime import datetime

from alpha_gen.api.wq_client import WorldQuantClient, WorldQuantError
# MODIFIED IMPORT: Added AlphaMetrics
from alpha_gen.models.alpha import Alpha, SimulationResult, SimulationSettings, AlphaMetrics

logger = logging.getLogger(__name__)

class AlphaSimulatorError(Exception):
    """Base exception for alpha simulator errors."""
    pass

class AlphaSimulator:
    """
    Manages batch simulation of alpha expressions.

    Handles parallel simulation, monitoring, and result aggregation.
    """

    def __init__(
        self,
        wq_client: WorldQuantClient,
        output_dir: str = "./output",
        max_concurrent_simulations: int = 5
    ):
        """
        Initialize alpha simulator.

        Args:
            wq_client: WorldQuant API client
            output_dir: Directory for saving results
            max_concurrent_simulations: Maximum number of concurrent simulations
        """
        self.wq_client = wq_client
        self.output_dir = output_dir
        self.max_concurrent_simulations = max_concurrent_simulations

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info("Alpha Simulator initialized")

    def simulate_batch(
        self,
        alphas: List[Alpha],
        save_results: bool = True,
        filename_prefix: str = "batch"
    ) -> List[Tuple[Alpha, Dict]]:
        """
        Simulate a batch of alphas in parallel.

        Args:
            alphas: List of Alpha objects to simulate
            save_results: Whether to save results to disk
            filename_prefix: Prefix for result files

        Returns:
            List of (Alpha, result) tuples
        """
        if not alphas:
            logger.warning("No alphas provided for simulation")
            return []

        logger.info(f"Simulating batch of {len(alphas)} alphas")
        results = []

        # Create a thread pool for concurrent simulations
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_simulations) as executor:
            # Submit all simulations
            future_to_alpha = {
                executor.submit(self._simulate_alpha, alpha): alpha for alpha in alphas
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_alpha):
                alpha = future_to_alpha[future]
                try:
                    result = future.result()
                    if result:
                        results.append((alpha, result))
                        logger.info(f"Completed simulation for alpha {alpha.id or 'unknown'}")
                    else:
                        logger.warning(f"Failed simulation for alpha {alpha.id or 'unknown'}")

                except Exception as e:
                    logger.error(f"Error in simulation for alpha {alpha.id or 'unknown'}: {str(e)}")

        # Save results if requested
        if save_results and results:
            timestamp = int(time.time())
            results_file = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.json")

            try:
                result_data = []
                for alpha, result in results:
                    # Combine alpha and result data
                    entry = {
                        "alpha_id": alpha.id,
                        "expression": alpha.expression,
                        "simulation_result": result
                    }

                    # Add metrics if available
                    if alpha.metrics:
                         entry["metrics"] = { # Convert metrics dataclass to dict
                            'sharpe': alpha.metrics.sharpe,
                            'fitness': alpha.metrics.fitness,
                            'turnover': alpha.metrics.turnover,
                            'returns': alpha.metrics.returns,
                            'drawdown': alpha.metrics.drawdown,
                            'margin': alpha.metrics.margin,
                            'long_count': alpha.metrics.long_count,
                            'short_count': alpha.metrics.short_count,
                        }
                    elif "alpha_details" in result and result["alpha_details"]:
                        entry["metrics"] = result["alpha_details"].get("is", {})

                    result_data.append(entry)

                with open(results_file, 'w') as f:
                    json.dump(result_data, f, indent=2)

                logger.info(f"Saved batch results to {results_file}")

            except Exception as e:
                logger.error(f"Failed to save batch results: {str(e)}")

        logger.info(f"Completed batch simulation ({len(results)}/{len(alphas)} successful)")
        return results

    def _simulate_alpha(self, alpha: Alpha) -> Optional[Dict]:
        """
        Simulate a single alpha.

        Args:
            alpha: Alpha object to simulate

        Returns:
            Simulation result or None if failed
        """
        try:
            # Convert settings to API format
            settings = alpha.settings.to_api_format()

            # Run simulation
            result = self.wq_client.simulate_alpha(
                expression=alpha.expression,
                settings=settings
            )

            # Check if simulation returned a result before proceeding
            if result is None:
                 logger.error(f"Simulation returned None for {alpha.expression[:50]}..., likely failed internally.")
                 return None

            # Update alpha with results
            if "alpha_details" in result and result["alpha_details"]:
                alpha_details = result["alpha_details"]
                alpha.id = alpha_details.get("id")
                alpha.status = alpha_details.get("status", "UNSUBMITTED")
                alpha.grade = alpha_details.get("grade", "UNKNOWN")

                # Update metrics
                if "is" in alpha_details and alpha_details["is"]:
                    metrics_data = alpha_details["is"]
                    # MODIFIED: Call AlphaMetrics directly
                    alpha.metrics = AlphaMetrics.from_api_format(metrics_data)

            return result

        except WorldQuantError as e:
            logger.error(f"Simulation failed for {alpha.expression[:50]}... (WorldQuantError): {str(e)}")
            return None
        except Exception as e:
             logger.exception(f"Unexpected error during _simulate_alpha for {alpha.expression[:50]}...: {str(e)}")
             return None


    def simulate_multiple_regions(
        self,
        alphas: List[Alpha],
        regions: List[str],
        save_results: bool = True,
        filename_prefix: str = "multi_region"
    ) -> Dict[str, List[Tuple[Alpha, Dict]]]:
        """
        Simulate alphas across multiple regions.

        Args:
            alphas: List of Alpha objects to simulate
            regions: List of region codes to test
            save_results: Whether to save results to disk
            filename_prefix: Prefix for result files

        Returns:
            Dictionary mapping regions to result lists
        """
        if not alphas:
            logger.warning("No alphas provided for multi-region simulation")
            return {}

        if not regions:
            logger.warning("No regions provided for multi-region simulation")
            return {}

        logger.info(f"Simulating {len(alphas)} alphas across {len(regions)} regions")
        region_results = {}

        for region in regions:
            logger.info(f"Simulating for region: {region}")

            # Create region-specific alphas with adjusted settings
            region_alphas = []
            for alpha in alphas:
                # Create a copy with updated region
                region_alpha = Alpha(
                    expression=alpha.expression,
                    id=alpha.id, # Keep original ID if needed for reference? Or should it be None?
                    name=alpha.name,
                    settings=SimulationSettings(
                        instrument_type=alpha.settings.instrument_type,
                        region=region, # Set the new region
                        universe=alpha.settings.universe,
                        delay=alpha.settings.delay,
                        decay=alpha.settings.decay,
                        neutralization=alpha.settings.neutralization,
                        truncation=alpha.settings.truncation,
                        pasteurization=alpha.settings.pasteurization,
                        unit_handling=alpha.settings.unit_handling,
                        nan_handling=alpha.settings.nan_handling,
                        language=alpha.settings.language,
                        visualization=alpha.settings.visualization
                    )
                    # Note: Metrics are not copied, they will be region-specific
                )
                region_alphas.append(region_alpha)

            # Simulate for this region
            results = self.simulate_batch(
                alphas=region_alphas,
                save_results=save_results, # Save per-region results if desired
                filename_prefix=f"{filename_prefix}_{region.lower()}"
            )

            region_results[region] = results

        # Aggregate results across regions if needed
        if save_results:
            timestamp = int(time.time())
            aggregated_file = os.path.join(self.output_dir, f"{filename_prefix}_aggregated_{timestamp}.json")

            try:
                aggregated_data = {}
                for region, region_specific_results in region_results.items():
                    region_data = []
                    for alpha, result in region_specific_results:
                        entry = {
                            "alpha_id": alpha.id, # This might be None or the ID from the *last* region simulated
                            "expression": alpha.expression,
                            "region": region, # Add region info
                            "simulation_result": result
                        }

                        # Add metrics if available
                        if alpha.metrics:
                            entry["metrics"] = { # Convert metrics dataclass to dict
                                'sharpe': alpha.metrics.sharpe,
                                'fitness': alpha.metrics.fitness,
                                'turnover': alpha.metrics.turnover,
                                'returns': alpha.metrics.returns,
                                'drawdown': alpha.metrics.drawdown,
                                'margin': alpha.metrics.margin,
                                'long_count': alpha.metrics.long_count,
                                'short_count': alpha.metrics.short_count,
                            }
                        elif "alpha_details" in result and result["alpha_details"]:
                             entry["metrics"] = result["alpha_details"].get("is", {})

                        region_data.append(entry)

                    aggregated_data[region] = region_data

                with open(aggregated_file, 'w') as f:
                    json.dump(aggregated_data, f, indent=2)

                logger.info(f"Saved aggregated multi-region results to {aggregated_file}")

            except Exception as e:
                logger.error(f"Failed to save aggregated results: {str(e)}")

        logger.info(f"Completed multi-region simulation for {len(regions)} regions")
        return region_results