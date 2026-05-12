import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate duration and a custom column from a CSV file")
    parser.add_argument("input_file", type=Path, help="Path to input CSV file")
    parser.add_argument("column", help="Column name to aggregate")
    args = parser.parse_args()

    try:
        duration_data: dict[tuple[str, str], list[float]] = defaultdict(list)
        col_data: dict[tuple[str, str], list[float]] = defaultdict(list)

        with args.input_file.open(newline="") as f:
            reader = csv.DictReader(f)

            required = ["Name", "Input Shapes", args.column]
            if not all(col in reader.fieldnames for col in required):
                missing = [col for col in required if col not in reader.fieldnames]
                parser.error(f"Missing columns: {', '.join(missing)}")

            for row in reader:
                try:
                    key = (row["Name"], row["Input Shapes"])
                    duration_data[key].append(float(row["Duration(us)"]))
                    col_data[key].append(float(row[args.column]))
                except ValueError:
                    print(f"Warning: Invalid value at row {reader.line_num}")

        print("\nDuration totals:")
        print("-" * 40)
        print("Name,Shape,Mean_Duration")
        for (name, shape), durations in duration_data.items():
            mean = sum(durations) / len(durations)
            print(f"{name},{shape},{mean:.2f}")
        print("-" * 40)

        print(f"Name,Shape:Total, Mean {args.column}")
        for (name, shape), col_times in col_data.items():
            total = sum(col_times)
            mean = total / len(col_times)
            print(f"{name},{shape}:total={total:.2f}, mean={mean:.2f}")
        print("-" * 40)

    except FileNotFoundError:
        parser.error(f"File not found: {args.input_file}")


if __name__ == "__main__":
    main()