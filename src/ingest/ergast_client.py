from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class ErgastConfig:
    base_url: str = "https://api.jolpi.ca/ergast/f1"
    timeout_sec: int = 30
    max_retries: int = 3
    backoff_sec: float = 1.5
    page_limit: int = 1000  # Ergast supports pagination with limit/offset


class ErgastClient:
    """
    Client for interacting with the Ergast-compatible F1 API.

    Responsibilities:
    - Handle HTTP requests
    - Retry on transient failures
    - Handle pagination (limit/offset)
    """

    def __init__(self, config: ErgastConfig = ErgastConfig(), session: Optional[requests.Session] = None):
        self.config = config
        self.session = session or requests.Session()

    def _get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a GET request and return parsed JSON.
        Retries up to max_retries on failure.
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.config.timeout_sec)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                # basic exponential-ish backoff
                time.sleep(self.config.backoff_sec * attempt)
        raise RuntimeError(f"Ergast request failed after retries: url={url} params={params}") from last_err

    def fetch_all(self, path: str, extra_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Fetch all pages for a given endpoint path, returning a list of "Race" dicts (or similar),
        based on the MRData structure.

        Example paths:
          - "2010/results.json"
          - "2010.json"  (races)
          - "drivers.json"
        """
        extra_params = extra_params or {}
        url = f"{self.config.base_url}/{path}"

        all_items: List[Dict[str, Any]] = []
        offset = 0
        limit = self.config.page_limit

        while True:
            params = {"limit": limit, "offset": offset, **extra_params}
            payload = self._get_json(url, params=params)

            mrdata = payload.get("MRData", {})
            total = int(mrdata.get("total", "0") or 0)

            # IMPORTANT: the API may cap the limit (e.g., to 100) even if we request 1000
            returned_limit = int(mrdata.get("limit", str(limit)) or limit)
            returned_offset = int(mrdata.get("offset", str(offset)) or offset)

            race_table = mrdata.get("RaceTable", {})
            races = race_table.get("Races", [])

            if races:
                all_items.extend(races)

                # Move to next page using what the server actually returned
                offset = returned_offset + returned_limit

                if offset >= total:
                    break
            else:
                break

        return all_items

    def fetch_raw(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch a single JSON payload without pagination handling assumptions.
        Useful for drivers/constructors/circuits endpoints.
        """
        params = params or {"limit": self.config.page_limit, "offset": 0}
        url = f"{self.config.base_url}/{path}"
        return self._get_json(url, params=params)
