#!/usr/bin/env python3
"""
Template-based update_services.py for SambaNova.

Yields model dictionaries that are rendered using Jinja2 templates.

Usage: python scripts/update_services.py
"""

import os
import sys
from pathlib import Path
from typing import Iterator

import httpx

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.template_populate import populate_from_iterator

# Provider Configuration
PROVIDER_NAME = "sambanova"
PROVIDER_DISPLAY_NAME = "SambaNova"
API_BASE_URL = "https://api.sambanova.ai/v1"
ENV_API_KEY_NAME = "SAMBANOVA_API_KEY"

SCRIPT_DIR = Path(__file__).parent


class ModelSource:
    """Fetches models and yields template dictionaries."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_fetcher = ModelDataFetcher()
        self.litellm_data = None

    def iter_models(self) -> Iterator[dict]:
        """Yield model dictionaries for template rendering."""
        # Fetch LiteLLM data once
        self.litellm_data = self.data_fetcher.fetch_litellm_model_data()

        print(f"Fetching models from {PROVIDER_DISPLAY_NAME} API...")
        try:
            r = httpx.get(
                f"{API_BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            r.raise_for_status()
            models = r.json().get("data", [])
            print(f"Found {len(models)} models\n")
        except Exception as e:
            print(f"Error listing models: {e}")
            return

        for i, model_info in enumerate(models, 1):
            model_id = model_info.get("id", "")
            print(f"[{i}/{len(models)}] {model_id}")

            # Build template variables
            template_vars = self._build_template_vars(model_id, model_info)
            if template_vars:
                yield template_vars
                print("  OK")

    def _build_template_vars(self, model_id: str, model_info: dict) -> dict:
        """Build template variables for a model."""
        service_type = self._determine_service_type(model_id)
        display_name = model_id.replace("-", " ").replace("_", " ").title()

        # Build details from LiteLLM data and model info
        details = {}
        model_data = ModelDataLookup.lookup_model_details(
            model_id, self.litellm_data or {})

        if model_data:
            for field in [
                    "max_tokens", "max_input_tokens", "max_output_tokens",
                    "mode"
            ]:
                if field in model_data:
                    details[field] = model_data[field]
            if "litellm_provider" in model_data:
                details["litellm_provider"] = model_data["litellm_provider"]

        # Function-calling capability.  ``ModelDataLookup`` returns the
        # first ``*/<model_id>`` row it sees, which can be a non-sambanova
        # provider with optimistic flags (e.g. gemma's deepinfra entry
        # claims tool support but SambaNova's hosted gemma rejects
        # ``tools`` with 400).  Prefer the explicit ``sambanova/`` row,
        # then apply a denylist for models LiteLLM-elsewhere marks
        # tool-capable that SambaNova nonetheless rejects.  Drop entries
        # when SambaNova adds upstream support.  Same pattern as Nebius.
        sambanova_specific = (self.litellm_data or {}).get(
            f"sambanova/{model_id}", model_data
        )
        supports_function_calling = bool(
            (sambanova_specific or {}).get("supports_function_calling")
        )
        if model_id in self._FC_DENYLIST:
            supports_function_calling = False

        if "owned_by" in model_info:
            details["owned_by"] = model_info["owned_by"]
        if "object" in model_info:
            details["object"] = model_info["object"]

        # Canonical (snake_case) metadata required by the platform validator
        # for LLM offerings.  Both keys must be present; null asserts
        # "unknown".  Claude models are closed-source so parameter_count
        # is permanently null per the canonical helper.  metadata_sources
        # records provenance so reviewers can triage stale-value reports.
        canonical = ModelDataLookup.get_canonical_metadata(
            model_id,
            fetcher=self.data_fetcher,
        )
        details["context_length"] = canonical["context_length"]
        details["parameter_count"] = canonical["parameter_count"]
        if canonical["sources"]:
            details["metadata_sources"] = canonical["sources"]

        # Extract upstream pricing for description, but set prices to 0 for BYOK
        pricing = None
        if model_data:
            if "input_cost_per_token" in model_data and "output_cost_per_token" in model_data:
                input_price = round(float(
                    model_data["input_cost_per_token"]) * 1_000_000, 4)
                output_price = round(float(
                    model_data["output_cost_per_token"]) * 1_000_000, 4)
                price_desc = (
                    f"Service provider charges "
                    f"${self._format_price(input_price)} / "
                    f"${self._format_price(output_price)} "
                    f"per 1M input/output tokens"
                )
                pricing = {
                    "type": "one_million_tokens",
                    "input": "0",
                    "output": "0",
                    "description": price_desc,
                }
                # Include cached_input if available
                if "cache_read_input_token_cost" in model_data:
                    cached_price = round(float(
                        model_data["cache_read_input_token_cost"]) * 1_000_000, 4)
                    pricing["cached_input"] = "0"
                    price_desc = (
                        f"Service provider charges "
                        f"${self._format_price(input_price)} / "
                        f"${self._format_price(output_price)} / "
                        f"${self._format_price(cached_price)} "
                        f"per 1M input/output/cached tokens"
                    )
                    pricing["description"] = price_desc

        return {
            # Directory name uses -byok suffix (used by populate_from_iterator)
            "name": f"{model_id}-byok",
            # Offering name is the model_id (without -byok suffix)
            "offering_name": model_id,
            # Offering fields
            "display_name": display_name,
            "description": f"{display_name} language model",
            "service_type": service_type,
            "status": "ready",
            "details": details,
            "payout_price": pricing,
            # Listing fields
            "list_price": pricing,
            "supports_function_calling": supports_function_calling,
            # Provider config (for templates)
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "api_base_url": API_BASE_URL,
            "env_api_key_name": ENV_API_KEY_NAME,
        }

    # Models LiteLLM marks as tool-capable but SambaNova's chat-completion
    # endpoint rejects with 400 when ``tools`` is sent.  Empirically
    # discovered — drop entries when SambaNova adds upstream support.
    # ``DeepSeek-V3.1-cb`` (continuous-batch variant of V3.1) is gated
    # automatically by the ``sambanova/<model>``-first lookup since
    # LiteLLM has no entry for it; kept here as documentation in case a
    # future LiteLLM release adds an optimistic row.
    _FC_DENYLIST = frozenset({
        "DeepSeek-V3.1-cb",
    })

    def _determine_service_type(self, model_id: str) -> str:
        model_lower = model_id.lower()
        if any(kw in model_lower for kw in ["embed", "embedding"]):
            return "embedding"
        if any(kw in model_lower for kw in ["rerank"]):
            return "rerank"
        if any(kw in model_lower for kw in ["vision"]):
            return "vision_language_model"
        return "llm"

    def _format_price(self, price: float) -> str:
        """Format price without trailing .0 for whole numbers."""
        if price == int(price):
            return str(int(price))
        return str(price)


def main():
    api_key = os.environ.get(ENV_API_KEY_NAME)
    if not api_key:
        print(f"Error: {ENV_API_KEY_NAME} not set")
        sys.exit(1)

    source = ModelSource(api_key)
    populate_from_iterator(
        iterator=source.iter_models(),
        templates_dir=SCRIPT_DIR.parent / "templates",
        output_dir=SCRIPT_DIR.parent / "services",
    )


if __name__ == "__main__":
    main()
