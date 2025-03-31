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
    ) -> Optional[requests.Response]: # MODIFIED: Allow returning None defensively
        """
        Make a request to the WorldQuant API with retry logic.

        Args:
            method: HTTP method ('get', 'post', 'patch', etc.)
            endpoint: API endpoint to call
            json_data: JSON data to include in the request
            params: URL parameters to include
            handle_retry_after: Whether to handle Retry-After headers

        Returns:
            Response object or None if a critical error occurs before raising.
            (Note: Ideally should only return Response or raise Exception)

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
        response = None # Initialize response to None

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
                    # Try parsing Retry-After as float for more precision
                    retry_after_str = response.headers.get('Retry-After')
                    try:
                        retry_after = float(retry_after_str) if retry_after_str else self.retry_delay
                    except (ValueError, TypeError):
                        retry_after = self.retry_delay
                        logger.warning(f"Could not parse Retry-After header: {retry_after_str}. Using default {retry_after}s.")

                    logger.warning(f"Rate limited. Waiting {retry_after:.1f} seconds...")
                    time.sleep(retry_after)
                    continue # Continue the loop to retry the request

                # If successful or unhandled status code, return the response
                return response

            except Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Request timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request to {endpoint} timed out after {self.max_retries} attempts")
                    raise WorldQuantError(f"Request to {endpoint} timed out after multiple attempts")

            except RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request to {endpoint} failed after {self.max_retries} attempts: {str(e)}")
                    raise WorldQuantError(f"Request to {endpoint} failed: {str(e)}")
            except Exception as e: # Catch unexpected errors during request attempt
                 logger.exception(f"Unexpected error during request attempt {attempt + 1} to {url}: {e}")
                 if attempt >= self.max_retries - 1:
                     raise WorldQuantError(f"Unexpected error during final request attempt to {endpoint}: {e}")
                 wait_time = self.retry_delay * (2 ** attempt)
                 time.sleep(wait_time) # Wait before retrying on unexpected error too

        # If loop finishes without returning or raising (shouldn't happen ideally)
        logger.error(f"Request loop finished unexpectedly for {endpoint}")
        return response # Return whatever the last response was, possibly None

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
        if response is None or response.status_code != 200:
            raise WorldQuantError(f"Failed to fetch data fields count: {response.text if response else 'No response'}")

        data = response.json()
        total_count = data.get('count', 0)
        results = data.get('results', [])

        # Fetch remaining pages if needed
        current_fetched = len(results)
        while current_fetched < total_count:
            offset = current_fetched
            params['offset'] = offset
            logger.debug(f"Fetching data fields page at offset {offset}")
            response = self._make_request('get', self.DATA_FIELDS_ENDPOINT, params=params)
            if response and response.status_code == 200:
                page_data = response.json()
                page_results = page_data.get('results', [])
                results.extend(page_results)
                current_fetched += len(page_results)
                if not page_results: # Break if a page returns empty results
                    logger.warning(f"Received empty page at offset {offset}, stopping pagination.")
                    break
            else:
                logger.warning(f"Failed to fetch data fields page at offset {offset}. Status: {response.status_code if response else 'No Response'}. Stopping pagination.")
                break # Stop pagination on error

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
        if response is None or response.status_code != 200:
            raise WorldQuantError(f"Failed to fetch operators: {response.text if response else 'No response'}")

        data = response.json()
        operators = data if isinstance(data, list) else data.get('results', [])

        logger.info(f"Successfully fetched {len(operators)} operators")
        return operators

    def submit_simulation(
        self,
        expression: str,
        settings: Optional[Dict] = None
    ) -> Optional[str]: # MODIFIED: Return Optional[str] as it might fail
        """
        Submit an alpha expression for simulation.

        Args:
            expression: Alpha expression to simulate
            settings: Optional simulation settings (defaults used if not provided)

        Returns:
            URL for monitoring simulation progress, or None if submission fails critically.

        Raises:
            SimulationError: If simulation submission fails with an API error message.
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

        # --- Start Modification ---
        # Add a check for None response before accessing attributes
        if response is None:
            error_msg = f"Simulation submission failed: Did not receive a valid response from the API for expression {expression[:100]}..."
            logger.error(error_msg)
            # Raise SimulationError consistent with other failures here.
            raise SimulationError(error_msg)
            # Alternatively, could return None here if preferred by the caller logic:
            # return None
        # --- End Modification ---

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
            # This case should ideally be covered by status_code != 201 check, but added defensively
            raise SimulationError("No progress URL in simulation response header, although status was 201.")

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
                logger.warning(f"Monitoring request failed for {progress_url} (attempt {attempt+1}/{max_attempts}). Retrying after {poll_interval}s.")
                time.sleep(poll_interval)
                continue

            # Handle rate limiting specifically within monitoring
            if response.status_code == 429 and 'Retry-After' in response.headers:
                retry_after_value = response.headers.get('Retry-After')
                try:
                    # Convert to float, sleep uses float seconds
                    retry_after_seconds = float(retry_after_value)
                    logger.warning(f"Rate limited during monitoring (attempt {attempt+1}/{max_attempts}). Waiting {retry_after_seconds:.1f} seconds (Retry-After: {retry_after_value})...")
                    time.sleep(retry_after_seconds)
                    continue # Retry the request
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse Retry-After header value: {retry_after_value} (attempt {attempt+1}/{max_attempts}). Using default poll interval {poll_interval}s.")
                    time.sleep(poll_interval)
                    continue # Retry the request

            # Handle specific non-error codes indicating progress
            if response.status_code == 202: # Accepted (still processing)
                logger.debug(f"Simulation status: Accepted (still processing, attempt {attempt+1}/{max_attempts}), waiting...")
                time.sleep(poll_interval)
                continue
            if response.status_code == 204: # No Content (still initializing)
                logger.debug(f"Simulation still initializing (status 204, attempt {attempt+1}/{max_attempts}), waiting...")
                time.sleep(poll_interval)
                continue

            # Handle unexpected status codes *other* than 200
            if response.status_code != 200:
                 logger.warning(f"Unexpected status {response.status_code} while monitoring {progress_url} (attempt {attempt+1}/{max_attempts}). Content: {response.text[:200]}. Retrying...")
                 time.sleep(poll_interval)
                 continue
            # --- End Modification ---

            # We expect status 200 now if it's ready or has info
            # Handle case where response is 200 but body is empty (unlikely but possible)
            if not response.text.strip():
                logger.debug(f"Simulation status 200 but empty response (attempt {attempt+1}/{max_attempts}), waiting...")
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
                else: # PENDING, RUNNING, etc.
                    logger.debug(f"Simulation status: {status} (attempt {attempt+1}/{max_attempts}), waiting...")
                    time.sleep(poll_interval)

            except ValueError: # JSONDecodeError inherits from ValueError
                logger.warning(f"Could not parse simulation response as JSON (attempt {attempt+1}/{max_attempts}): {response.text[:200]}")
                time.sleep(poll_interval) # Wait and retry

        raise SimulationError(f"Simulation monitoring timed out after {max_attempts} attempts for {progress_url}")

    def simulate_alpha(
        self,
        expression: str,
        settings: Optional[Dict] = None,
        get_alpha_details: bool = True
    ) -> Optional[Dict]: # MODIFIED: Can return None if simulation fails
        """
        Submit alpha for simulation and wait for results.

        Args:
            expression: Alpha expression to simulate
            settings: Optional simulation settings
            get_alpha_details: Whether to fetch alpha details after simulation

        Returns:
            Dictionary with simulation results and (optionally) alpha details,
            or None if the simulation submission or monitoring fails critically.

        Raises:
            SimulationError: If simulation completes with FAILED/ERROR status.
        """
        try:
            progress_url = self.submit_simulation(expression, settings)
            # If submit_simulation failed critically and raised SimulationError, propagate it
            if progress_url is None:
                 # This case should now be less likely due to the check in submit_simulation
                 logger.error("simulate_alpha: progress_url was None, submission failed critically.")
                 return None

            simulation_result = self.monitor_simulation(progress_url)
            # monitor_simulation raises SimulationError on timeout or FAILED/ERROR status

            result = {
                'simulation': simulation_result,
                'expression': expression,
                'alpha_details': None # Initialize
            }

            if get_alpha_details and 'alpha' in simulation_result:
                alpha_id = simulation_result['alpha']
                if alpha_id: # Ensure alpha ID is not None or empty
                    try:
                        alpha_details = self.get_alpha_details(alpha_id)
                        result['alpha_details'] = alpha_details
                    except WorldQuantError as e:
                        logger.warning(f"Could not fetch alpha details for {alpha_id}: {str(e)}")
                        # Keep result['alpha_details'] as None
                else:
                     logger.warning(f"Simulation result for {expression[:50]} completed but missing alpha ID.")

            return result

        except SimulationError as e:
             # Re-raise SimulationError if monitor_simulation indicated failure/error status
             # or if submit_simulation failed with a specific error message
             logger.error(f"Simulation failed for {expression[:50]}...: {str(e)}")
             raise # Propagate SimulationError (indicates failed sim status or timeout)
        except WorldQuantError as e:
             # Catch other potential WorldQuant errors during the process
             logger.error(f"WorldQuant API error during simulation of {expression[:50]}...: {str(e)}")
             return None # Indicate failure without raising if it wasn't a specific SimError
        except Exception as e:
             # Catch unexpected errors
             logger.exception(f"Unexpected error during simulation of {expression[:50]}...: {str(e)}")
             return None # Indicate failure


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

        if response is None or response.status_code != 200:
            raise WorldQuantError(f"Failed to fetch alpha details for {alpha_id}: {response.text if response else 'No response'}")

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
            'hidden': 'false' # Typically want to see non-hidden alphas
        }

        if status:
            params['status'] = status

        logger.info(f"Fetching submitted alphas with params: {params}")

        # Endpoint is specific to the user
        response = self._make_request('get', "/users/self/alphas", params=params)

        if response is None or response.status_code != 200:
            raise WorldQuantError(f"Failed to fetch submitted alphas: {response.text if response else 'No response'}")

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
        submit_url = f"{self.ALPHAS_ENDPOINT}/{alpha_id}/submit"
        response = self._make_request('post', submit_url)

        if response is None or response.status_code != 201:
             # Check for specific error messages if possible
            error_details = ""
            if response:
                 try:
                     error_details = f" Details: {response.json()}"
                 except ValueError:
                     error_details = f" Response: {response.text[:200]}"
            raise WorldQuantError(f"Alpha submission POST failed for {alpha_id}. Status: {response.status_code if response else 'No Response'}.{error_details}")

        # Monitor submission progress by GETting the same endpoint
        # WQ Brain API uses GET on submit endpoint to check status
        # It returns 204 while processing, and 200 with JSON body on completion/failure
        monitoring_attempts = 30 # ~5 minutes
        poll_interval_submit = 10 # seconds

        for attempt in range(monitoring_attempts):
            logger.debug(f"Checking submission status for {alpha_id} (attempt {attempt+1}/{monitoring_attempts})")
            check_response = self._make_request('get', submit_url)

            if check_response is None:
                 logger.warning(f"Failed to get submission status for {alpha_id} on attempt {attempt+1}. Retrying...")
                 time.sleep(poll_interval_submit)
                 continue

            if check_response.status_code == 200: # Submission check complete (success or failure info)
                try:
                    result = check_response.json()
                    # Check for success/failure within the result body if needed, based on API spec
                    logger.info(f"Alpha {alpha_id} submission status check returned 200 OK.")
                    return result # Return the final status info
                except ValueError:
                     logger.error(f"Alpha {alpha_id} submission status check returned 200 OK but failed to parse JSON: {check_response.text[:200]}")
                     # Treat as failure, maybe raise error?
                     raise WorldQuantError(f"Alpha submission status check for {alpha_id} failed: Invalid JSON response.")
            elif check_response.status_code == 204: # Still processing
                 logger.debug(f"Alpha {alpha_id} submission still processing (status 204). Waiting {poll_interval_submit}s...")
                 time.sleep(poll_interval_submit)
            else: # Unexpected status during monitoring
                 logger.warning(f"Unexpected status {check_response.status_code} while checking submission for {alpha_id}. Retrying...")
                 time.sleep(poll_interval_submit)

        raise WorldQuantError(f"Alpha submission monitoring timed out after {monitoring_attempts} attempts for {alpha_id}")


    def set_alpha_properties(
        self,
        alpha_id: str,
        name: Optional[str] = None,
        color: Optional[str] = None,
        tags: Optional[List[str]] = None, # Changed to Optional
        description: Optional[str] = None
    ) -> None:
        """
        Update alpha properties. Only sends parameters that are not None.

        Args:
            alpha_id: Alpha ID to update
            name: New name (if not None)
            color: New color (if not None)
            tags: List of tags (replaces existing tags if not None)
            description: Description text (if not None)
        """
        params = {}
        regular_params = {} # For nested 'regular' properties like description

        if name is not None:
            params["name"] = name
        if color is not None:
            params["color"] = color
        if tags is not None: # Note: This usually replaces existing tags entirely
            params["tags"] = tags
        if description is not None:
             # Description is nested under 'regular' in the PATCH request body
             regular_params["description"] = description

        if regular_params:
            params["regular"] = regular_params

        if not params:
            logger.warning(f"No properties provided to update for alpha {alpha_id}. Skipping PATCH.")
            return

        logger.info(f"Updating properties for alpha {alpha_id} with data: {params}")

        response = self._make_request('patch', f"{self.ALPHAS_ENDPOINT}/{alpha_id}", json_data=params)

        if response is None or response.status_code != 200:
            raise WorldQuantError(f"Failed to update alpha properties for {alpha_id}: {response.text if response else 'No response'}")

        logger.info(f"Successfully updated properties for alpha {alpha_id}")