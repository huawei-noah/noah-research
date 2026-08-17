# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

from __future__ import annotations

import pytest

from tests import gen_description_lite as generator
from tests import regenerate_all_description_lite as regenerator


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            """### Evaluation

#### Evaluation

Submissions are evaluated using categorization accuracy.

#### Submission Format
""",
            "categorization accuracy",
        ),
        (
            """### Evaluation

#### Root Mean Squared Error RMSE

Submissions are scored on root mean squared error.

#### Submission File
""",
            "root mean squared error",
        ),
        (
            """## **Evaluation**

Submissions are evaluated on area under the ROC curve.

### **Submission File**
""",
            "area under the ROC curve",
        ),
        (
            """### Supervised ML Evaluation

This competition is evaluated on the mean Dice coefficient.

#### Submission File
""",
            "mean Dice coefficient",
        ),
    ],
)
def test_extract_evaluation_section_variants(source: str, expected: str) -> None:
    section = generator.extract_header_depth2_to_5(source, "Evaluation")

    assert section is not None
    metric, _ = generator.split_submission_from_evaluation(section)
    assert expected in metric


def test_regenerated_document_keeps_nested_evaluation_metric() -> None:
    source = """# Overview

### Description

Classify each image into one of five categories.

### Evaluation

#### Evaluation

Submissions are evaluated using categorization accuracy.

#### Submission Format

```csv
image_id,label
example.jpg,0
```
"""

    document = regenerator.build_document("example-task", source)

    metric_section = document.split("## Target metric (evaluation)", 1)[1].split(
        "## Brief background", 1
    )[0]
    assert "categorization accuracy" in metric_section
