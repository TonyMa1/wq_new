"""
WorldQuant Brain API client with robust error handling and session management.
This unified client replaces multiple duplicate implementations across the codebase.
"""

import os
import time
import logging
import json
from typing import Dict, List, Optional, Tuple, Union, Any
import requests
from requests.exceptions import RequestException, Timeout

# Configure module logger
logger = logging.getLogger(__name__)

class WorldQuantError(Exception):
    """Base exception class for WorldQuant API errors."""
    pass

class AuthenticationError(WorldQuantError):
    """Raised when authentication with WorldQuant Brain fails."""
    pass

class SimulationError(WorldQuantError):
    """Raised when a simulation fails unexpectedly."""
    pass

class RateLimitError(WorldQuantError):
    """Raised when rate limits are hit."""
    pass

class WorldQuantClient:
    """
    Unified client for interacting with WorldQuant Brain API.
    
    This client manages authentication, session handling, retries,
    and provides a clean interface for all WorldQuant API operations.
    """
    
    # API Endpoints
    BASE_URL = "https://api.worldquantbrain.com"
    AUTH_ENDPOINT = "/authentication"
    SIMULATIONS_ENDPOINT = "/simulations"
    ALPHAS_ENDPOINT = "/alphas"
    DATA_FIELDS_ENDPOINT = "/data-fields"
    OPERATORS_ENDPOINT = "/operators"
    
    # Default simulation settings
    DEFAULT_SIMULATION_SETTINGS = {
        'instrumentType': 'EQUITY',
        'region': 'USA',
        'universe': 'TOP3000',
        'delay': 1,
        'decay': 0,
        'neutralization': 'INDUSTRY',
        'truncation': 0.08,
        'pasteurization': 'ON',
        'unitHandling': 'VERIFY',
        'nanHandling': 'OFF',
        'language': 'FASTEXPR',
        'visualization': False,
    }
    
    def __init__(
        self, 
        username: Optional[str] = None, 
        password: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        timeout: int = 30
    ):
        """
        Initialize client with credentials from env or parameters.
        
        Args:
            username: WorldQuant Brain username (default: from WQ_USERNAME env var)
            password: WorldQuant Brain password (default: from WQ_PASSWORD env var)
            max_retries: Maximum number of retries for failed requests
            retry_delay: Base delay between retries in seconds (exponential backoff)
            timeout: Request timeout in seconds
        """
        self.username = username or os.environ.get("WQ_USERNAME")
        self.password = password or os.environ.get("WQ_PASSWORD")
        
        if not self.username or not self.password:
            raise AuthenticationError(
                "WorldQuant credentials not provided. Either pass them as parameters "
                "or set WQ_USERNAME and WQ_PASSWORD environment variables."
            )
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.session = requests.Session()
        self.login()
    
    def login(self) -> None:
        """
        Authenticate and establish a session with WorldQuant Brain.
        
        Raises:
            AuthenticationError: If authentication fails
        """
        logger.info("Authenticating with WorldQuant Brain...")
        
        for attempt in range(self.max_retries):
            try:
                self.session = requests.Session()
                self.session.auth = (self.username, self.password)
                response = self.session.post(
                    f"{self.BASE_URL}{self.AUTH_ENDPOINT}", 
                    timeout=self.timeout
                )
                
                if response.status_code != 201:
                    error_msg = f"Authentication failed (status: {response.status_code})"
                    try:
                        error_details = response.json()
                        error_msg += f", details: {error_details}"
                    except ValueError:
                        error_msg += f", response: {response.text[:200]}"
                    
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(f"{error_msg}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise AuthenticationError(error_msg)
                
                # Set common headers for future requests
                self.session.headers.update({
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                })
                
                logger.info("Authentication successful")
                return
                
            except Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Authentication timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise AuthenticationError("Authentication timed out after multiple attempts")
                    
            except RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Authentication request failed: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise AuthenticationError(f"Authentication request failed: {str(e)}")
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        handle_retry_after: bool = True
    ) -> requests.Response:
        """
        Make a request to the WorldQuant API with retry logic.
        
        Args:
            method: HTTP method ('get', 'post', 'patch', etc.)
            endpoint: API endpoint to call
            json_data: JSON data to include in the request
            params: URL parameters to include
            handle_retry_after: Whether to handle Retry-After headers
            
        Returns:
            Response object
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limits are hit
            WorldQuantError: For other API errors
        """
        if endpoint.startswith(('http://', 'https://')):
            url = endpoint  # Use the endpoint directly if it's absolute
        else:
            url = f"{self.BASE_URL}{endpoint}"  # Otherwise, prepend the base URL
            
        request_func = getattr(self.session, method.lower())
        
        for attempt in range(self.max_retries):
            try:
                # Use the correctly determined url variable
                response = request_func(
                    url,
                    json=json_data,
                    params=params,
                    timeout=self.timeout
                )
                
                # Handle authentication failures
                if response.status_code == 401:
                    if attempt < self.max_retries - 1:
                        logger.warning("Session expired, re-authenticating...")
                        self.login()
                        continue
                    else:
                        raise AuthenticationError("Failed to re-authenticate after multiple attempts")
                
                # Handle rate limiting
                if response.status_code == 429 and handle_retry_after:
                    retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                return response
                
            except Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Request timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise WorldQuantError(f"Request to {endpoint} timed out after multiple attempts")
                    
            except RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise WorldQuantError(f"Request to {endpoint} failed: {str(e)}")
    
    def get_data_fields(
        self,
        instrument_type: str = 'EQUITY',
        region: str = 'USA',
        delay: int = 1,
        universe: str = 'TOP3000',
        dataset_id: str = '',
        search: str = '',
        limit: int = 50
    ) -> List[Dict]:
        """
        Fetch available data fields from WorldQuant Brain.
        
        Args:
            instrument_type: Instrument type (EQUITY, etc.)
            region: Region code (USA, JAPAN, etc.)
            delay: Delay value
            universe: Universe name (TOP3000, etc.)
            dataset_id: Optional dataset ID to filter by
            search: Optional search term
            limit: Maximum fields to return per request
            
        Returns:
            List of data field objects
        """
        params = {
            'instrumentType': instrument_type,
            'region': region,
            'delay': delay,
            'universe': universe,
            'limit': limit
        }
        
        if dataset_id:
            params['dataset.id'] = dataset_id
        
        if search:
            params['search'] = search
        
        logger.info(f"Fetching data fields with params: {params}")
        
        # First get the total count
        response = self._make_request('get', self.DATA_FIELDS_ENDPOINT, params=params)
        if response.status_code != 200:
            raise WorldQuantError(f"Failed to fetch data fields: {response.text}")
        
        data = response.json()
        total_count = data.get('count', 0)
        results = data.get('results', [])
        
        # Fetch remaining pages if needed
        for offset in range(limit, total_count, limit):
            params['offset'] = offset
            response = self._make_request('get', self.DATA_FIELDS_ENDPOINT, params=params)
            if response.status_code == 200:
                page_data = response.json()
                results.extend(page_data.get('results', []))
            else:
                logger.warning(f"Failed to fetch data fields page at offset {offset}")
        
        logger.info(f"Successfully fetched {len(results)} data fields")
        return results
    
    def get_operators(self) -> List[Dict]:
        """
        Fetch available operators from WorldQuant Brain.
        
        Returns:
            List of operator objects
        """
        logger.info("Fetching operators")
        
        response = self._make_request('get', self.OPERATORS_ENDPOINT)
        if response.status_code != 200:
            raise WorldQuantError(f"Failed to fetch operators: {response.text}")
        
        data = response.json()
        operators = data if isinstance(data, list) else data.get('results', [])
        
        logger.info(f"Successfully fetched {len(operators)} operators")
        return operators
    
    def submit_simulation(
        self, 
        expression: str,
        settings: Optional[Dict] = None
    ) -> str:
        """
        Submit an alpha expression for simulation.
        
        Args:
            expression: Alpha expression to simulate
            settings: Optional simulation settings (defaults used if not provided)
            
        Returns:
            URL for monitoring simulation progress
            
        Raises:
            SimulationError: If simulation submission fails
        """
        sim_settings = self.DEFAULT_SIMULATION_SETTINGS.copy()
        if settings:
            sim_settings.update(settings)
            
        simulation_data = {
            'type': 'REGULAR',
            'settings': sim_settings,
            'regular': expression
        }
        
        logger.info(f"Submitting simulation for expression: {expression[:100]}...")
        
        response = self._make_request('post', self.SIMULATIONS_ENDPOINT, json_data=simulation_data)
        
        if response.status_code != 201:
            error_msg = f"Simulation submission failed (status: {response.status_code})"
            try:
                error_details = response.json()
                error_msg += f", details: {error_details}"
            except ValueError:
                error_msg += f", response: {response.text[:200]}"
            raise SimulationError(error_msg)
        
        progress_url = response.headers.get('Location')
        if not progress_url:
            raise SimulationError("No progress URL in simulation response")
        
        logger.info(f"Simulation submitted successfully, progress URL: {progress_url}")
        return progress_url
    
    def monitor_simulation(
        self, 
        progress_url: str,
        max_attempts: int = 60,
        poll_interval: int = 5
    ) -> Dict:
        """
        Monitor a simulation until completion.
        
        Args:
            progress_url: Progress URL from simulation submission
            max_attempts: Maximum number of polling attempts
            poll_interval: Time between polls in seconds
            
        Returns:
            Simulation result data
            
        Raises:
            SimulationError: If simulation fails or times out
        """
        logger.info(f"Monitoring simulation: {progress_url}")
        
        for attempt in range(max_attempts):
            # Set handle_retry_after=False as we will handle it explicitly here
            response = self._make_request('get', progress_url, handle_retry_after=False)

            # --- Start Modification ---
            # First, check if the response object itself is valid
            if response is None:
                logger.warning(f"Monitoring request failed for {progress_url}. Retrying after {poll_interval}s.")
                time.sleep(poll_interval)
                continue

            # Handle rate limiting specifically within monitoring
            if response.status_code == 429 and 'Retry-After' in response.headers:
                retry_after_value = response.headers.get('Retry-After')
                try:
                    # Convert to float, sleep uses float seconds
                    retry_after_seconds = float(retry_after_value)
                    logger.warning(f"Rate limited during monitoring. Waiting {retry_after_seconds:.1f} seconds (Retry-After: {retry_after_value})...")
                    time.sleep(retry_after_seconds)
                    continue # Retry the request
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse Retry-After header value: {retry_after_value}. Using default poll interval {poll_interval}s.")
                    time.sleep(poll_interval)
                    continue # Retry the request

            # Handle unexpected status codes other than rate limit
            if response.status_code != 200 and response.status_code != 202: # 202 Accepted might also mean processing
                 # Check if simulation is still initializing based on empty response
                 if not response.text.strip() and response.status_code == 204: # Check specifically for 204 No Content
                     logger.debug(f"Simulation still initializing (status {response.status_code}, attempt {attempt+1}/{max_attempts})")
                     time.sleep(poll_interval)
                     continue
                 # Log other unexpected statuses before trying to parse JSON
                 logger.warning(f"Unexpected status {response.status_code} while monitoring {progress_url}. Content: {response.text[:200]}. Retrying...")
                 time.sleep(poll_interval)
                 continue
            # --- End Modification ---

            # Handle empty response (simulation still initializing - keep this check too)
            if not response.text.strip():
                logger.debug(f"Simulation still initializing (empty response, attempt {attempt+1}/{max_attempts})")
                time.sleep(poll_interval)
                continue
            
            # Handle empty response (simulation still initializing)
            if not response.text.strip():
                logger.debug(f"Simulation still initializing (attempt {attempt+1}/{max_attempts})")
                time.sleep(poll_interval)
                continue
            
            try:
                result = response.json()
                status = result.get('status')
                
                if status == 'COMPLETE':
                    logger.info("Simulation completed successfully")
                    return result
                elif status in ['FAILED', 'ERROR']:
                    error_msg = f"Simulation failed with status {status}"
                    if 'message' in result:
                        error_msg += f": {result['message']}"
                    raise SimulationError(error_msg)
                else:
                    logger.debug(f"Simulation status: {status}, waiting...")
                    time.sleep(poll_interval)
                    
            except ValueError:
                logger.warning(f"Could not parse simulation response as JSON: {response.text[:200]}")
                time.sleep(poll_interval)
        
        raise SimulationError(f"Simulation monitoring timed out after {max_attempts} attempts")
    
    def simulate_alpha(
        self, 
        expression: str,
        settings: Optional[Dict] = None,
        get_alpha_details: bool = True
    ) -> Dict:
        """
        Submit alpha for simulation and wait for results.
        
        Args:
            expression: Alpha expression to simulate
            settings: Optional simulation settings
            get_alpha_details: Whether to fetch alpha details after simulation
            
        Returns:
            Dictionary with simulation results and (optionally) alpha details
            
        Raises:
            SimulationError: If simulation fails
        """
        progress_url = self.submit_simulation(expression, settings)
        simulation_result = self.monitor_simulation(progress_url)
        
        result = {
            'simulation': simulation_result,
            'expression': expression
        }
        
        if get_alpha_details and 'alpha' in simulation_result:
            alpha_id = simulation_result['alpha']
            try:
                alpha_details = self.get_alpha_details(alpha_id)
                result['alpha_details'] = alpha_details
            except WorldQuantError as e:
                logger.warning(f"Could not fetch alpha details: {str(e)}")
                result['alpha_details'] = None
        
        return result
    
    def get_alpha_details(self, alpha_id: str) -> Dict:
        """
        Get detailed information about an alpha.
        
        Args:
            alpha_id: Alpha ID
            
        Returns:
            Alpha details dictionary
            
        Raises:
            WorldQuantError: If the request fails
        """
        logger.info(f"Fetching details for alpha {alpha_id}")
        
        response = self._make_request('get', f"{self.ALPHAS_ENDPOINT}/{alpha_id}")
        
        if response.status_code != 200:
            raise WorldQuantError(f"Failed to fetch alpha details: {response.text}")
        
        return response.json()
    
    def get_submitted_alphas(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str = None,
        order: str = "-dateCreated"
    ) -> Dict:
        """
        Fetch submitted alphas.
        
        Args:
            limit: Maximum number of results
            offset: Result offset
            status: Filter by status (SUBMITTED, UNSUBMITTED, etc.)
            order: Sort order
            
        Returns:
            Dictionary with count, results, etc.
        """
        params = {
            'limit': limit,
            'offset': offset,
            'order': order,
            'hidden': 'false'
        }
        
        if status:
            params['status'] = status
        
        logger.info(f"Fetching submitted alphas with params: {params}")
        
        response = self._make_request('get', "/users/self/alphas", params=params)
        
        if response.status_code != 200:
            raise WorldQuantError(f"Failed to fetch submitted alphas: {response.text}")
        
        return response.json()
    
    def submit_alpha(self, alpha_id: str) -> Dict:
        """
        Submit an alpha for consideration by WorldQuant.
        
        Args:
            alpha_id: Alpha ID to submit
            
        Returns:
            Submission response data
            
        Raises:
            WorldQuantError: If submission fails
        """
        logger.info(f"Submitting alpha {alpha_id}")
        
        # Start submission
        response = self._make_request('post', f"{self.ALPHAS_ENDPOINT}/{alpha_id}/submit")
        
        if response.status_code != 201:
            raise WorldQuantError(f"Alpha submission failed: {response.text}")
        
        # Monitor submission progress
        for attempt in range(30):  # Maximum wait of 5 minutes
            check_response = self._make_request('get', f"{self.ALPHAS_ENDPOINT}/{alpha_id}/submit")
            
            # When submission is done, we'll get a JSON response
            if check_response.text.strip():
                try:
                    result = check_response.json()
                    logger.info(f"Alpha {alpha_id} submission complete")
                    return result
                except ValueError:
                    pass
            
            time.sleep(10)
        
        raise WorldQuantError(f"Alpha submission monitoring timed out for {alpha_id}")
    
    def set_alpha_properties(
        self,
        alpha_id: str,
        name: Optional[str] = None,
        color: Optional[str] = None,
        tags: List[str] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Update alpha properties.
        
        Args:
            alpha_id: Alpha ID to update
            name: New name
            color: New color
            tags: List of tags
            description: Description text
        """
        params = {}
        if name is not None:
            params["name"] = name
        if color is not None:
            params["color"] = color
        if tags is not None:
            params["tags"] = tags
        if description is not None:
            params["regular"] = {"description": description}
        
        logger.info(f"Updating properties for alpha {alpha_id}")
        
        response = self._make_request('patch', f"{self.ALPHAS_ENDPOINT}/{alpha_id}", json_data=params)
        
        if response.status_code != 200:
            raise WorldQuantError(f"Failed to update alpha properties: {response.text}")
        
        logger.info(f"Successfully updated properties for alpha {alpha_id}")