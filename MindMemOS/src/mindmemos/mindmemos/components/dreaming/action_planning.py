"""Parser for dreaming consolidation action planning output."""

from __future__ import annotations

import json

from ...typing import ConsolidationAction


def action_planning_parser(content: str) -> ConsolidationAction:
    """Parse action planner JSON output."""

    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)
    normalized_links = []
    for link in data.get("links", []) or []:
        if not isinstance(link, dict):
            normalized_links.append(link)
            continue
        if "source_kind" not in link:
            source_id = link.get("source_id") or link.get("from") or link.get("from_memory_id")
            target_id = link.get("target_id") or link.get("to") or link.get("to_memory_id")
            if source_id and target_id:
                link = {
                    "source_kind": "Memory",
                    "source_id": source_id,
                    "target_kind": "Memory",
                    "target_id": target_id,
                    "relation_type": link.get("relation_type")
                    or link.get("link_type")
                    or link.get("type")
                    or "related",
                    "property_name": link.get("property_name"),
                    "reason": link.get("reason", ""),
                    "metadata": link.get("metadata", {}),
                }
        if str(link.get("source_kind", "")).lower() == "memory":
            link["source_kind"] = "Memory"
        if str(link.get("target_kind", "")).lower() == "memory":
            link["target_kind"] = "Memory"
        if str(link.get("source_kind", "")).lower() == "entity":
            link["source_kind"] = "Entity"
        if str(link.get("target_kind", "")).lower() == "entity":
            link["target_kind"] = "Entity"
        if "relation_type" not in link and link.get("link_type"):
            link["relation_type"] = link["link_type"]
        normalized_links.append(link)
    data["links"] = normalized_links
    return ConsolidationAction.model_validate(data)
