from __future__ import annotations

from mindmemos.components.dreaming.action_planning import action_planning_parser


def test_action_planning_parser_accepts_link_type_as_relation_type():
    content = """
    {
      "links": [
        {
          "source_kind": "Memory",
          "source_id": "current",
          "target_kind": "Memory",
          "target_id": "stale",
          "link_type": "supersedes"
        }
      ]
    }
    """

    action = action_planning_parser(content)

    assert len(action.links) == 1
    assert action.links[0].relation_type == "supersedes"
