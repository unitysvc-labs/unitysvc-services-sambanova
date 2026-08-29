#!/usr/bin/env python3
"""
Template-based update_services.py for SambaNova.

Yields model dictionaries that are rendered using Jinja2 templates.

Usage: python scripts/update_services.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Iterator

import httpx

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.params_render import write_params_from_iterator

# Provider Configuration
PROVIDER_NAME = "sambanova"
PROVIDER_DISPLAY_NAME = "SambaNova"
API_BASE_URL = "https://api.sambanova.ai/v1"
ENV_API_KEY_NAME = "SAMBANOVA_API_KEY"

SCRIPT_DIR = Path(__file__).parent


#: The committed param files this script rewrites. Read for the price guard in
#: :func:`_committed_pricing_note`.
SPECS_DIR = SCRIPT_DIR.parent / "specs"


def _committed_pricing_note(service_name: str) -> str | None:
    """The ``pricing_note`` already committed for *service_name*, if any.

    Read from the generated param file **directly**, never through
    ``load_param_data``: that merges the ``<name>.override.json`` companion, and
    absorbing an override's value into the generated file would make the
    override look redundant and invite its deletion.

    Returns None when the service is new (no param file yet) or when its note is
    null — in both cases this run has no committed rate to silently re-ship.
    """
    path = SPECS_DIR / f"{service_name}.json"
    try:
        raw = path.read_text()
    except OSError:
        return None
    # A committed param file that will not parse is a real problem; let it raise
    # rather than quietly disarming the guard.
    params = (json.loads(raw) or {}).get("parameters") or {}
    note = params.get("pricing_note")
    return note if isinstance(note, str) else None


class ModelSource:
    """Fetches models and yields template dictionaries."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_fetcher = ModelDataFetcher()
        self.litellm_data = None

    def iter_models(self) -> Iterator[dict]:
        """Yield model dictionaries for template rendering."""
        # Fetch LiteLLM data once. The fetcher returns {} when the download
        # fails, which downstream is indistinguishable from "the registry has no
        # row for any of these models": every rate lookup comes back empty, and
        # since a null no longer overwrites a committed value (sellers 0.3.1)
        # the run would report success while quietly re-publishing yesterday's
        # rates for the whole catalog. One failure, so catch it once, up front.
        self.litellm_data = self.data_fetcher.fetch_litellm_model_data()
        if not self.litellm_data:
            print("Error: LiteLLM model registry came back empty — no rate can "
                  "be derived; refusing to populate")
            sys.exit(1)

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
            # Hard failure, not an empty iterator. Absence now RETIRES services
            # (``deprecate_missing``), so returning here would yield nothing and
            # ask the writer to deprecate the entire catalog on a transport
            # error. Exiting non-zero fails the workflow step and skips the
            # PR-creation step, which is the honest outcome for a failed fetch.
            print(f"Error listing models: {e}")
            sys.exit(1)

        if not models:
            # An empty enumeration means the upstream call did not really
            # succeed; it is never "SambaNova retired every model".
            print(f"Error: {PROVIDER_DISPLAY_NAME} returned no models")
            sys.exit(1)

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
        service_name = f"{PROVIDER_NAME}/{model_id}"
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
        # LiteLLM is sometimes optimistic; corrections belong in per-model
        # <name>.override.json companions (merged at render time by every
        # specs command), never in this script. Note ``DeepSeek-V3.1-cb``
        # (continuous-batch variant) is gated automatically by the
        # ``sambanova/<model>``-first lookup since LiteLLM has no entry for
        # it — add an override file if a future LiteLLM release grows an
        # optimistic row.
        supports_function_calling = bool(
            (sambanova_specific or {}).get("supports_function_calling")
        )

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

        # BYOK: the customer's own key pays the provider directly, so the service
        # is free through the gateway. This plain description is what payout_price
        # keeps (seller-facing). The customer-facing listing cell is composed in
        # listing.json.j2 from pricing_note, into the
        # "<amount> ~ <PILL> | <note>" grammar; do not build it here, since this
        # dict feeds payout_price too.
        pricing = {"type": "constant", "price": "0", "description": "Free (BYOK)"}
        pricing_note = None
        if model_data and "input_cost_per_token" in model_data and "output_cost_per_token" in model_data:
            input_price = round(float(model_data["input_cost_per_token"]) * 1_000_000, 4)
            output_price = round(float(model_data["output_cost_per_token"]) * 1_000_000, 4)
            if "cache_read_input_token_cost" in model_data:
                cached_price = round(float(model_data["cache_read_input_token_cost"]) * 1_000_000, 4)
                pricing_note = (
                    f"${self._format_price(input_price)} / "
                    f"${self._format_price(output_price)} / "
                    f"${self._format_price(cached_price)} "
                    f"per 1M input/output/cached tokens"
                )
            else:
                pricing_note = (
                    f"${self._format_price(input_price)} / "
                    f"${self._format_price(output_price)} "
                    f"per 1M input/output tokens"
                )

        # A rate we could not derive is not the same thing as a free model.
        # `pricing_note` is nullable and unvalidated, and a null no longer
        # overwrites a committed value (sellers 0.3.1), so a degraded None here
        # would silently republish yesterday's rate as though it had been
        # re-derived today. Fail when — and only when — there is a committed
        # rate to re-ship: a model LiteLLM has never covered has nothing to
        # protect, and refusing to run for it would break the catalog
        # permanently rather than catch a regression.
        if pricing_note is None:
            committed = _committed_pricing_note(service_name)
            if committed is not None:
                print(
                    f"Error: no litellm input/output rate for {model_id}, but "
                    f"{service_name} already publishes one ({committed!r}). A "
                    "failed lookup would silently keep the committed rate — "
                    "refusing to republish an unverified price."
                )
                sys.exit(1)

        return {
            # The service's name, which is also its path under specs/ ==
            # listing.name (flat layout, #1263). Required by unitysvc-sellers
            # 0.3.1; it is what deprecation matches committed services by, so it
            # must be stated, never inferred.
            "service_name": service_name,
            # Offering name is the bare upstream model_id
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
            # Reference rates for the BYOK pricing paragraph (template-rendered)
            "pricing_note": pricing_note,
            "supports_function_calling": supports_function_calling,
            # Provider config (for templates)
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "api_base_url": API_BASE_URL,
            "env_api_key_name": ENV_API_KEY_NAME,
        }

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
    # ``deprecate_missing`` defaults to True and is left on: this script
    # enumerates the whole upstream catalog on every run and has no --limit,
    # so anything committed that the run does not yield really has stopped
    # being served. Every whole-run failure path above (empty registry,
    # fetch error, empty enumeration, an unverifiable price on a service that
    # already publishes one) exits non-zero instead of reaching this call with
    # a short list.
    stats = write_params_from_iterator(
        iterator=source.iter_models(),
        output_dir=SCRIPT_DIR.parent / "specs",
    )
    print(f"\nDone: {stats}")
    print(f"New: {stats['new']}, deprecated: {stats['deprecated']}")


if __name__ == "__main__":
    main()
