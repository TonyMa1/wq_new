"""
Alpha submission handling.
Provides tools for submitting and tracking alphas.
"""

import logging
import time
import os
import json
from typing import Dict, List, Optional, Tuple, Any
import concurrent.futures
from datetime import datetime, timedelta
import re

from alpha_gen.api.wq_client import WorldQuantClient, WorldQuantError
from alpha_gen.models.alpha import Alpha, AlphaMetrics

logger = logging.getLogger(__name__)

class AlphaSubmitterError(Exception):
    """Base exception for alpha submitter errors."""
    pass

class AlphaSubmitter:
    """
    Handles submission of alphas to WorldQuant Brain.
    
    Provides tools for validation, submission, and tracking.
    """
    
    def __init__(
        self,
        wq_client: WorldQuantClient,
        output_dir: str = "./output",
        max_concurrent_submissions: int = 3
    ):
        """
        Initialize alpha submitter.
        
        Args:
            wq_client: WorldQuant API client
            output_dir: Directory for saving results
            max_concurrent_submissions: Maximum number of concurrent submissions
        """
        self.wq_client = wq_client
        self.output_dir = output_dir
        self.max_concurrent_submissions = max_concurrent_submissions
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info("Alpha Submitter initialized")
    
    def find_successful_alphas(
        self,
        sharpe_threshold: float = 1.25,
        fitness_threshold: float = 1.0,
        max_turnover: float = 0.7,
        min_turnover: float = 0.01,
        max_results: int = 100,
        max_age_days: int = 30
    ) -> List[Alpha]:
        """
        Find successful alphas that meet submission criteria.
        
        Args:
            sharpe_threshold: Minimum Sharpe ratio
            fitness_threshold: Minimum fitness value
            max_turnover: Maximum turnover
            min_turnover: Minimum turnover
            max_results: Maximum number of results to return
            max_age_days: Maximum age in days
            
        Returns:
            List of successful Alpha objects
        """
        logger.info(f"Finding successful alphas (sharpe >= {sharpe_threshold}, fitness >= {fitness_threshold})")
        
        # Calculate date cutoff
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
        
        try:
            # Fetch alphas in batches
            all_alphas = []
            offset = 0
            limit = 50  # Batch size
            
            while len(all_alphas) < max_results:
                logger.info(f"Fetching alphas batch (offset={offset}, limit={limit})")
                
                response = self.wq_client.get_submitted_alphas(
                    limit=limit,
                    offset=offset,
                    status="UNSUBMITTED",
                    order="-dateCreated"
                )
                
                results = response.get("results", [])
                if not results:
                    break
                
                # Filter alphas that meet criteria
                for alpha_data in results:
                    # Skip alphas older than cutoff
                    date_created = alpha_data.get("dateCreated")
                    if date_created and date_created < cutoff_str:
                        continue
                    
                    # Check metrics
                    is_data = alpha_data.get("is", {})
                    sharpe = is_data.get("sharpe", 0)
                    fitness = is_data.get("fitness", 0)
                    turnover = is_data.get("turnover", 0)
                    
                    if (abs(sharpe) >= sharpe_threshold and
                        abs(fitness) >= fitness_threshold and
                        min_turnover <= turnover <= max_turnover):
                        
                        # Convert to Alpha object
                        try:
                            alpha = Alpha.from_api_format(alpha_data)
                            all_alphas.append(alpha)
                            
                            if len(all_alphas) >= max_results:
                                break
                        except Exception as e:
                            logger.error(f"Failed to parse alpha {alpha_data.get('id')}: {str(e)}")
                
                # Break if no more results or reached limit
                if len(results) < limit or len(all_alphas) >= max_results:
                    break
                
                # Increment offset
                offset += limit
            
            logger.info(f"Found {len(all_alphas)} successful alphas")
            return all_alphas
            
        except WorldQuantError as e:
            logger.error(f"Failed to find successful alphas: {str(e)}")
            raise AlphaSubmitterError(f"Failed to find successful alphas: {str(e)}")
    
    def validate_alpha_for_submission(self, alpha: Alpha) -> Tuple[bool, Optional[str]]:
        """
        Validate if an alpha meets submission criteria.
        
        Args:
            alpha: Alpha to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Ensure alpha has metrics
        if not alpha.metrics:
            return False, "No metrics available"
        
        # Check Sharpe ratio
        if abs(alpha.metrics.sharpe) < 1.25:
            return False, f"Sharpe ratio too low: {alpha.metrics.sharpe}"
        
        # Check fitness
        if not alpha.metrics.fitness or abs(alpha.metrics.fitness) < 1.0:
            return False, f"Fitness too low: {alpha.metrics.fitness}"
        
        # Check turnover
        if alpha.metrics.turnover < 0.01:
            return False, f"Turnover too low: {alpha.metrics.turnover}"
        if alpha.metrics.turnover > 0.7:
            return False, f"Turnover too high: {alpha.metrics.turnover}"
        
        # Check checks
        if not alpha.metrics.checks:
            return False, "No check results available"
        
        for check in alpha.metrics.checks:
            if check.name in ["LOW_SHARPE", "LOW_FITNESS", "LOW_TURNOVER", "HIGH_TURNOVER", "CONCENTRATED_WEIGHT"]:
                if check.result == "FAIL":
                    return False, f"Check failed: {check.name}"
        
        return True, None
    
    def submit_alphas(
        self,
        alphas: List[Alpha],
        validate: bool = True,
        save_results: bool = True
    ) -> List[Tuple[Alpha, Dict]]:
        """
        Submit multiple alphas for WorldQuant review.
        
        Args:
            alphas: List of Alpha objects to submit
            validate: Whether to validate alphas before submission
            save_results: Whether to save results to disk
            
        Returns:
            List of (Alpha, result) tuples
        """
        if not alphas:
            logger.warning("No alphas provided for submission")
            return []
        
        logger.info(f"Submitting {len(alphas)} alphas")
        results = []
        
        # Validate alphas if requested
        if validate:
            valid_alphas = []
            for alpha in alphas:
                is_valid, error = self.validate_alpha_for_submission(alpha)
                if is_valid:
                    valid_alphas.append(alpha)
                else:
                    logger.warning(f"Skipping invalid alpha {alpha.id}: {error}")
            
            if not valid_alphas:
                logger.error("No valid alphas to submit")
                return []
            
            alphas = valid_alphas
            logger.info(f"{len(alphas)} alphas passed validation")
        
        # Create a thread pool for concurrent submissions
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_submissions) as executor:
            # Submit all alphas
            future_to_alpha = {
                executor.submit(self._submit_alpha, alpha): alpha for alpha in alphas
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_alpha):
                alpha = future_to_alpha[future]
                try:
                    result = future.result()
                    if result:
                        results.append((alpha, result))
                        logger.info(f"Successfully submitted alpha {alpha.id}")
                    else:
                        logger.warning(f"Failed to submit alpha {alpha.id}")
                        
                except Exception as e:
                    logger.error(f"Error submitting alpha {alpha.id}: {str(e)}")
        
        # Save results if requested
        if save_results and results:
            timestamp = int(time.time())
            results_file = os.path.join(self.output_dir, f"submission_results_{timestamp}.json")
            
            try:
                result_data = []
                for alpha, result in results:
                    entry = {
                        "alpha_id": alpha.id,
                        "expression": alpha.expression,
                        "submission_result": result
                    }
                    result_data.append(entry)
                
                with open(results_file, 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                logger.info(f"Saved submission results to {results_file}")
                
            except Exception as e:
                logger.error(f"Failed to save submission results: {str(e)}")
        
        logger.info(f"Completed submission of {len(results)}/{len(alphas)} alphas")
        return results
    
    def _submit_alpha(self, alpha: Alpha) -> Optional[Dict]:
        """
        Submit a single alpha.
        
        Args:
            alpha: Alpha to submit
            
        Returns:
            Submission result or None if failed
        """
        if not alpha.id:
            logger.error("Alpha has no ID")
            return None
        
        try:
            # Submit alpha
            result = self.wq_client.submit_alpha(alpha.id)
            
            # Update alpha status
            alpha.status = "SUBMITTED"
            
            return result
            
        except WorldQuantError as e:
            logger.error(f"Submission failed: {str(e)}")
            return None
    
    def tag_alpha(
        self,
        alpha: Alpha,
        tags: List[str],
        name: Optional[str] = None,
        color: Optional[str] = None,
        description: Optional[str] = None
    ) -> bool:
        """
        Add tags and properties to an alpha.
        
        Args:
            alpha: Alpha to tag
            tags: List of tags to add
            name: Optional new name
            color: Optional color
            description: Optional description
            
        Returns:
            Success status
        """
        if not alpha.id:
            logger.error("Alpha has no ID")
            return False
        
        try:
            # Set alpha properties
            self.wq_client.set_alpha_properties(
                alpha_id=alpha.id,
                name=name,
                color=color,
                tags=tags,
                description=description
            )
            
            # Update alpha object
            if name:
                alpha.name = name
            if tags:
                alpha.tags = tags
            if color:
                alpha.color = color
            if description:
                alpha.description = description
            
            logger.info(f"Successfully tagged alpha {alpha.id}")
            return True
            
        except WorldQuantError as e:
            logger.error(f"Failed to tag alpha {alpha.id}: {str(e)}")
            return False