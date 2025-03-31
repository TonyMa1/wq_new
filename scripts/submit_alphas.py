 #!/usr/bin/env python
"""
Script to submit high-performing alphas to WorldQuant.

This script provides a command-line interface for finding, validating,
and submitting successful alphas to WorldQuant Brain.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Optional
import time
import json

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alpha_gen.api.wq_client import WorldQuantClient, WorldQuantError
from alpha_gen.core.alpha_submitter import AlphaSubmitter, AlphaSubmitterError
from alpha_gen.models.alpha import Alpha, SimulationResult, SimulationSettings
from alpha_gen.utils.logging import setup_logging, get_logger
from alpha_gen.utils.config import Config

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Submit high-performing alphas to WorldQuant')
    
    # Mode options
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'find', 'submit'],
                       help='Operation mode (default: auto)')
    
    # Find options
    parser.add_argument('--sharpe-threshold', type=float, default=1.25,
                       help='Minimum Sharpe ratio (default: 1.25)')
    parser.add_argument('--fitness-threshold', type=float, default=1.0,
                       help='Minimum fitness value (default: 1.0)')
    parser.add_argument('--max-turnover', type=float, default=0.7,
                       help='Maximum turnover (default: 0.7)')
    parser.add_argument('--min-turnover', type=float, default=0.01,
                       help='Minimum turnover (default: 0.01)')
    parser.add_argument('--max-age-days', type=int, default=30,
                       help='Maximum age in days (default: 30)')
    parser.add_argument('--max-results', type=int, default=20,
                       help='Maximum number of results (default: 20)')
    
    # Submit options
    parser.add_argument('--input', type=str, default=None,
                       help='Input file with alphas to submit (required for submit mode)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation of alphas before submission')
    parser.add_argument('--tags', type=str, default=None,
                       help='Comma-separated list of tags to add to submitted alphas')
    parser.add_argument('--max-concurrent', type=int, default=3,
                       help='Maximum concurrent submissions (default: 3)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory (default: ./output)')
    
    # Logging options
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'submit' and not args.input:
        parser.error("--input is required for submit mode")
    
    return args

def load_alphas(input_path):
    """Load alphas from input file."""
    if not os.path.exists(input_path):
        raise ValueError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    alphas = []
    
    if isinstance(data, list):
        # List of alphas or alpha dictionaries
        for item in data:
            if isinstance(item, dict):
                if 'id' in item:
                    # Alpha with ID
                    settings = item.get('settings', {})
                    alphas.append(Alpha(
                        id=item['id'],
                        expression=item.get('expression', ''),
                        settings=SimulationSettings.from_api_format(settings)
                    ))
                elif 'alpha_id' in item:
                    # Alpha result with ID
                    alphas.append(Alpha(
                        id=item['alpha_id'],
                        expression=item.get('expression', '')
                    ))
            elif isinstance(item, str):
                # Alpha ID
                alphas.append(Alpha(id=item, expression=''))
    elif isinstance(data, dict):
        # Dictionary of alphas or results
        for key, item in data.items():
            if isinstance(item, dict):
                if 'id' in item:
                    # Alpha with ID
                    settings = item.get('settings', {})
                    alphas.append(Alpha(
                        id=item['id'],
                        expression=item.get('expression', ''),
                        settings=SimulationSettings.from_api_format(settings)
                    ))
                elif 'alpha_id' in item:
                    # Alpha result with ID
                    alphas.append(Alpha(
                        id=item['alpha_id'],
                        expression=item.get('expression', '')
                    ))
    
    return alphas

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        log_dir='./logs'
    )
    logger.info("Starting alpha submission script")
    
    try:
        # Load configuration
        logger.info("Loading configuration")
        config = Config.load()
        
        # Create API client
        logger.info("Creating WorldQuant API client")
        wq_client = WorldQuantClient(
            username=config.wq.username,
            password=config.wq.password,
            max_retries=config.wq.max_retries,
            retry_delay=config.wq.retry_delay,
            timeout=config.wq.timeout
        )
        
        # Create alpha submitter
        logger.info("Creating alpha submitter")
        submitter = AlphaSubmitter(
            wq_client=wq_client,
            output_dir=args.output_dir,
            max_concurrent_submissions=args.max_concurrent
        )
        
        # Process according to mode
        if args.mode == 'auto' or args.mode == 'find':
            # Find successful alphas
            logger.info("Finding successful alphas")
            alphas = submitter.find_successful_alphas(
                sharpe_threshold=args.sharpe_threshold,
                fitness_threshold=args.fitness_threshold,
                max_turnover=args.max_turnover,
                min_turnover=args.min_turnover,
                max_age_days=args.max_age_days,
                max_results=args.max_results
            )
            
            logger.info(f"Found {len(alphas)} successful alphas")
            
            # Save to file
            timestamp = int(time.time())
            alphas_file = os.path.join(args.output_dir, f"successful_alphas_{timestamp}.json")
            
            os.makedirs(args.output_dir, exist_ok=True)
            with open(alphas_file, 'w') as f:
                alpha_data = []
                for alpha in alphas:
                    alpha_data.append({
                        "id": alpha.id,
                        "expression": alpha.expression,
                        "date_created": alpha.date_created.isoformat() if alpha.date_created else None,
                        "status": alpha.status,
                        "grade": alpha.grade,
                        "sharpe": alpha.metrics.sharpe if alpha.metrics else None,
                        "fitness": alpha.metrics.fitness if alpha.metrics else None,
                        "turnover": alpha.metrics.turnover if alpha.metrics else None
                    })
                json.dump(alpha_data, f, indent=2)
            
            logger.info(f"Saved successful alphas to {alphas_file}")
            
            # Submit in auto mode
            if args.mode == 'auto' and alphas:
                # Prepare tags if provided
                tags = None
                if args.tags:
                    tags = [tag.strip() for tag in args.tags.split(',')]
                
                # Submit alphas
                logger.info(f"Submitting {len(alphas)} alphas")
                results = submitter.submit_alphas(
                    alphas=alphas,
                    validate=not args.skip_validation,
                    save_results=True
                )
                
                logger.info(f"Submitted {len(results)} alphas")
                
                # Tag if requested
                if tags:
                    logger.info(f"Tagging submitted alphas with tags: {tags}")
                    for alpha, _ in results:
                        submitter.tag_alpha(
                            alpha=alpha,
                            tags=tags
                        )
        
        elif args.mode == 'submit':
            # Load alphas from file
            logger.info(f"Loading alphas from {args.input}")
            alphas = load_alphas(args.input)
            
            if not alphas:
                logger.error("No alphas found in input file")
                return 1
            
            logger.info(f"Loaded {len(alphas)} alphas")
            
            # Prepare tags if provided
            tags = None
            if args.tags:
                tags = [tag.strip() for tag in args.tags.split(',')]
            
            # Submit alphas
            logger.info(f"Submitting {len(alphas)} alphas")
            results = submitter.submit_alphas(
                alphas=alphas,
                validate=not args.skip_validation,
                save_results=True
            )
            
            logger.info(f"Submitted {len(results)} alphas")
            
            # Tag if requested
            if tags:
                logger.info(f"Tagging submitted alphas with tags: {tags}")
                for alpha, _ in results:
                    submitter.tag_alpha(
                        alpha=alpha,
                        tags=tags
                    )
        
        logger.info("Alpha submission completed successfully")
        return 0
        
    except WorldQuantError as e:
        logger.error(f"WorldQuant API error: {str(e)}")
        return 1
    except AlphaSubmitterError as e:
        logger.error(f"Error: {str(e)}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())