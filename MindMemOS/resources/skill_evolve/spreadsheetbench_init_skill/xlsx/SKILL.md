---
name: xlsx
description: Use this skill whenever the user wants to do anything with Excel spreadsheet files (.xlsx, .xls, .csv). This includes reading data, writing formulas, manipulating cells, formatting, filtering, creating charts, pivot tables, and any spreadsheet automation tasks.
---

# Excel Spreadsheet Processing

Use `openpyxl` to read and write .xlsx files.

## Quick Start

```python
from openpyxl import load_workbook

wb = load_workbook("input.xlsx")
ws = wb["Sheet1"]

value = ws["A1"].value
ws["B2"] = 42
ws["C2"] = "=SUM(A2:B2)"

wb.save("output.xlsx")
```

Use this for direct cell edits, formula updates, and simple workbook changes.

## Reading Data with pandas

```python
import pandas as pd

df = pd.read_excel('file.xlsx')                          # First sheet
all_sheets = pd.read_excel('file.xlsx', sheet_name=None) # All sheets as dict
```

## Common Pitfalls

- **Cell indices are 1-based**: `ws.cell(row=1, column=1)` is A1.
- **`data_only=True` destroys formulas on save**: Use a separate workbook object for reading calculated values.
- **`ws.max_row` overcounts**: May include formatted-but-empty rows. Scan the column to find the last non-empty cell when you need the true data range.
