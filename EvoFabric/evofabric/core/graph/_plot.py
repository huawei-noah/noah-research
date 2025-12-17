# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import tempfile
import webbrowser
from collections import defaultdict
from string import Template
from typing import Dict, List, Optional

from ._edge import EdgeSpecBase
from ._node import GraphNodeSpec
from ..typing import DEFAULT_EDGE_GROUP, SpecialNode


INDENT = "    "
SOLID_LINE = "-->"
DOTTED_LINE = "-.->"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>EvoFabric - Mermaid Diagram</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f8f9fa;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: left;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 300;
            letter-spacing: 2px;
            margin: 0;
        }

        .container {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .mermaid-wrapper {
            width: 100%;
            max-width: 80%;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            padding: 30px;
            transition: all 0.3s ease;
        }

        .mermaid {
            width: 100%;
            min-height: 300px;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .mermaid svg {
            max-width: 100%;
            height: auto !important;
            cursor: grab;
            transition: transform 0.2s ease;
        }

        .mermaid svg:active {
            cursor: grabbing;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8rem;
            }

            .mermaid-wrapper {
                max-width: 95%;
                padding: 20px;
            }
        }

        .loading {
            text-align: center;
            color: #666;
            font-size: 1.2rem;
        }

        .loading::after {
            content: '';
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>EvoFabric</h1>
    </header>

    <div class="container">
        <div class="mermaid-wrapper">
            <div class="mermaid" id="mermaid-diagram">
                $mermaid_code
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({ 
            startOnLoad: true,
            theme: 'default',
            themeVariables: {
                primaryColor: '#667eea',
                primaryTextColor: '#ffffff',
                primaryBorderColor: '#764ba2',
                lineColor: '#666666',
                secondaryColor: '#f8f9fa',
                tertiaryColor: '#ffffff'
            },
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            },
            sequence: {
                useMaxWidth: true,
                wrap: true
            },
            gantt: {
                useMaxWidth: true
            }
        });

        function adjustMermaidSize() {
            const mermaidElement = document.querySelector('.mermaid svg');
            if (mermaidElement) {
                const containerWidth = document.querySelector('.mermaid-wrapper').offsetWidth;
                const svgWidth = mermaidElement.getAttribute('width');
                const svgHeight = mermaidElement.getAttribute('height');

                if (svgWidth && containerWidth < svgWidth) {
                    const scale = (containerWidth * 0.9) / svgWidth;
                    mermaidElement.style.transform = `scale(${scale})`;
                    mermaidElement.style.transformOrigin = 'center center';
                }
            }
        }

        window.addEventListener('resize', adjustMermaidSize);

        window.addEventListener('load', () => {
            setTimeout(adjustMermaidSize, 100);
        });

        let scale = 1;
        let isDragging = false;
        let startX, startY, translateX = 0, translateY = 0;

        document.addEventListener('DOMContentLoaded', function() {
            const mermaidElement = document.querySelector('.mermaid');

            mermaidElement.addEventListener('wheel', function(e) {
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                scale *= delta;
                scale = Math.max(0.5, Math.min(scale, 3));
                updateTransform();
            });

            mermaidElement.addEventListener('mousedown', function(e) {
                if (e.target.closest('svg')) {
                    isDragging = true;
                    startX = e.clientX - translateX;
                    startY = e.clientY - translateY;
                }
            });

            document.addEventListener('mousemove', function(e) {
                if (isDragging) {
                    translateX = e.clientX - startX;
                    translateY = e.clientY - startY;
                    updateTransform();
                }
            });

            document.addEventListener('mouseup', function() {
                isDragging = false;
            });

            function updateTransform() {
                const svg = document.querySelector('.mermaid svg');
                if (svg) {
                    svg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
                }
            }
        });
    </script>
</body>
</html>
"""


def _translate_node_name(name: str) -> str:
    if SpecialNode.is_special_node(name):
        name = f"__{name}__"
    return name


def _edge_code(source, edge: EdgeSpecBase, cond_router_cnt: dict):
    code_lines = []
    conn_info = f"|\"{edge.group}\"|" if edge.group != DEFAULT_EDGE_GROUP else ""

    if edge.edge_type == "edge":
        line_style = SOLID_LINE
    else:
        line_style = DOTTED_LINE

    for target in edge.get_possible_targets():
        target = _translate_node_name(target)
        code_lines.append(f"{INDENT}{source} {line_style} {conn_info} {target}")
    return code_lines


def graph_to_mermaid(
        nodes: Dict[str, GraphNodeSpec],
        edges: Dict[str, List[EdgeSpecBase]]
):
    code_lines = ["flowchart TD"]

    for name, node in nodes.items():
        name = _translate_node_name(name)
        code_lines.append(f"{INDENT}{name}({name})")

    cond_router_cnt = defaultdict(int)

    for source, edges in edges.items():
        source = _translate_node_name(source)
        for edge in edges:
            code_lines.extend(_edge_code(source, edge, cond_router_cnt))

    return "\n".join(code_lines)


def render_mermaid_as_html(
        mermaid_code: str,
        save_path: Optional[str] = None,
        auto_open: bool = True
):
    """
    Render mermaid by HTML

    Args:
        mermaid_code (str): mermaid code
        save_path (Optional[str], optional): html file save path, if none, no saving is done. Defaults to None.
        auto_open (bool, optional): whether to automatically open the html file. Defaults to True.

    """
    html_content = Template(HTML_TEMPLATE).safe_substitute(mermaid_code=mermaid_code)

    if save_path:
        output_path = os.path.abspath(save_path)
    else:
        with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=".html",
                delete=False,
                encoding='utf-8'
        ) as f:
            output_path = f.name

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except IOError as e:
        return ""

    if auto_open:
        file_url = f"file://{output_path}"
        webbrowser.open(file_url)

    return output_path
