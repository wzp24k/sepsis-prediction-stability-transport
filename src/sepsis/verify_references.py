# -*- coding: utf-8 -*-
"""Verify every manuscript DOI against the registration agency and DOI resolver."""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
REFERENCE_PATH = ROOT / "manuscript" / "references.json"
OUTPUT_PATH = ROOT / "outputs" / "reference_verification.json"
MARKDOWN_PATH = ROOT / "work" / "reference_verification.md"
USER_AGENT = (
    "sepsis-prediction-stability-reference-audit/1.0 "
    "(mailto:311000687@seu.edu.cn)"
)


def normalize_title(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def title_matches(recorded: str, returned: str) -> bool:
    left = normalize_title(recorded)
    right = normalize_title(returned)
    return bool(left and right and left == right)


def manuscript_doi_order(reference_library: list[dict]) -> list[str]:
    """Return DOI order from the assembled manuscript reference list."""
    text = (ROOT / "outputs" / "manuscript_sepsis_stability.md").read_text(
        encoding="utf-8"
    )
    if "## 7. References" not in text:
        raise ValueError("Assembled manuscript is missing the reference section")
    reference_block = text.split("## 7. References", 1)[1]
    dois = [
        value.rstrip(".").lower()
        for value in re.findall(r"doi:(10\.\S+)", reference_block)
    ]
    expected = {entry["doi"].strip().lower() for entry in reference_library}
    if len(dois) != len(reference_library) or set(dois) != expected:
        raise ValueError(
            "Assembled manuscript DOI order does not match references.json"
        )
    return dois


def request_json(
    session: requests.Session, url: str, *, attempts: int = 3
) -> tuple[int, dict | None, str | None]:
    last_error = None
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=(10, 30))
            if response.status_code == 404:
                return 404, None, None
            response.raise_for_status()
            return response.status_code, response.json(), None
        except (requests.RequestException, ValueError) as error:
            last_error = str(error)
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
    return 0, None, last_error


def metadata_title(provider: str, payload: dict) -> str | None:
    if provider == "DataCite":
        titles = payload.get("data", {}).get("attributes", {}).get("titles", [])
        return titles[0].get("title") if titles else None
    titles = payload.get("message", {}).get("title", [])
    return titles[0] if titles else None


def verify_entry(entry: dict) -> dict:
    doi = entry["doi"].strip()
    encoded = quote(doi, safe="")
    provider = "DataCite" if doi.lower().startswith("10.48550/") else "Crossref"
    provider_url = (
        f"https://api.datacite.org/dois/{encoded}"
        if provider == "DataCite"
        else f"https://api.crossref.org/works/{encoded}"
    )
    handle_url = f"https://doi.org/api/handles/{encoded}"

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        handle_status, handle_payload, handle_error = request_json(
            session, handle_url
        )
        provider_status, provider_payload, provider_error = request_json(
            session, provider_url
        )

    handle_registered = bool(
        handle_status == 200
        and handle_payload
        and handle_payload.get("responseCode") == 1
    )
    provider_registered = bool(provider_status == 200 and provider_payload)
    returned_title = (
        metadata_title(provider, provider_payload) if provider_payload else None
    )
    metadata_match = bool(
        returned_title and title_matches(entry["title"], returned_title)
    )
    verified = bool(handle_registered and provider_registered and metadata_match)
    return {
        "key": entry["key"],
        "doi": doi,
        "provider": provider,
        "doi_handle_registered": handle_registered,
        "doi_handle_status": handle_status,
        "provider_record_found": provider_registered,
        "provider_http_status": provider_status,
        "returned_title": returned_title,
        "title_consistent": metadata_match,
        "verified": verified,
        "handle_error": handle_error,
        "provider_error": provider_error,
    }


def main() -> None:
    references = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    if not 40 <= len(references) <= 50:
        raise ValueError(f"Expected 40-50 references, found {len(references)}")
    dois = [entry["doi"].strip().lower() for entry in references]
    if len(set(dois)) != len(dois):
        raise ValueError("Duplicate DOI in references.json")

    entries = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(verify_entry, entry): entry for entry in references
        }
        for future in as_completed(futures):
            entries.append(future.result())
    final_doi_order = manuscript_doi_order(references)
    by_doi = {entry["doi"].strip().lower(): entry for entry in entries}
    entries = [by_doi[doi] for doi in final_doi_order]

    failures = [entry for entry in entries if not entry["verified"]]
    title_mismatches = [
        entry for entry in entries if not entry["title_consistent"]
    ]
    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_file": str(REFERENCE_PATH),
        "order_source": "outputs/manuscript_sepsis_stability.md",
        "count": len(entries),
        "all_verified": not failures,
        "failed_count": len(failures),
        "title_mismatch_count": len(title_mismatches),
        "providers": {
            provider: sum(entry["provider"] == provider for entry in entries)
            for provider in ("Crossref", "DataCite")
        },
        "entries": entries,
    }
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Reference verification",
        "",
        f"Verified: {result['verified_at_utc']}",
        "",
        (
            f"All {result['count']} references were ordered by the final manuscript "
            "and checked through the DOI Handle system and the relevant Crossref or "
            "DataCite record. Titles were compared after normalization. "
            f"Failures: {result['failed_count']}. "
            f"Registered-title mismatches: {result['title_mismatch_count']}. "
            "This audit establishes DOI resolution and title consistency. It does "
            "not independently verify every author, volume, issue, page, or claim."
        ),
        "",
        "| # | Key | DOI | Provider | Handle | Agency record | Title match |",
        "|---:|---|---|---|---|---|---|",
    ]
    for number, entry in enumerate(entries, 1):
        lines.append(
            "| {number} | `{key}` | {doi} | {provider} | {handle} | {record} | {title} |".format(
                number=number,
                key=entry["key"],
                doi=entry["doi"],
                provider=entry["provider"],
                handle="pass" if entry["doi_handle_registered"] else "fail",
                record="pass" if entry["provider_record_found"] else "fail",
                title="pass" if entry["title_consistent"] else "fail",
            )
        )
    MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(f"DOI verification failed for {len(failures)} references")


if __name__ == "__main__":
    main()
