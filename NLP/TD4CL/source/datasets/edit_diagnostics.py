"""
Based on: https://github.com/allenai/cartography/blob/main/cartography/classification/diagnostics_evaluation.py
"""
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import logging
import numpy as np
import os
import tqdm

from collections import defaultdict
from sklearn.metrics import matthews_corrcoef

# from cartography.data_utils_glue import read_glue_tsv, convert_string_to_unique_number

logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Use the same fields as GLUE SNLI test + additional field for Diagnostic NLI.
FIELDS = ["index", "captionID", "pairID", "sentence1_binary_parse", "sentence2_binary_parse",
          "sentence1_parse", "sentence2_parse", "sentence1", "sentence2", "category", "gold_label",
          ]

LOGIC = ["Negation", "Double negation", "Intervals/Numbers", "Conjunction", "Disjunction",
          "Conditionals", "Universal", "Existential", "Temporal", "Upward monotone",
          "Downward monotone", "Non-monotone",
          ]
LEXSEM = ["Lexical entailment", "Morphological negation", "Factivity", "Symmetry/Collectivity",
            "Redundancy", "Named entities", "Quantifiers",
            ]
PAS = ["Core args", "Prepositional phrases", "Ellipsis/Implicits", "Anaphora/Coreference",
        "Active/Passive", "Nominalization", "Genitives/Partitives", "Datives", "Relative clauses",
        "Coordination scope", "Intersectivity", "Restrictivity",
        ]
KNOWLEDGE = ["Common sense", "World knowledge",
              ]


# Based on paper: https://openreview.net/pdf?id=rJ4km2R5t7
category_names = {"logic": 364,
                  "predicate_argument_structure": 424,
                  "lexical_semantics": 368,
                  "knowledge": 284}
coarse_to_fine = {"logic": LOGIC,
                  "predicate_argument_structure": PAS,
                  "lexical_semantics": LEXSEM,
                  "knowledge": KNOWLEDGE}

fine_to_coarse = {}
for coarse_cat, category in coarse_to_fine.items():
  for fine_cat in category:
    assert fine_cat not in fine_to_coarse
    fine_to_coarse[fine_cat] = coarse_cat


def label_balance(label_list):
  distribution = defaultdict(int)
  for label in label_list:
    distribution[label] += 1
  for label in distribution:
    distribution[label] /= len(label_list)
  return np.std(list(distribution.values()))


def read_glue_tsv(file_path: str,
                  guid_index: int,
                  label_index: int = -1,
                  guid_as_int: bool = False):
  """
  Reads TSV files for GLUE-style text classification tasks.
  Returns:
    - a mapping between the example ID and the entire line as a string.
    - the header of the TSV file.
  """
  tsv_dict = {}

  i = -1
  with open(file_path, 'r') as tsv_file:
    for line in tqdm.tqdm([line for line in tsv_file]):
      i += 1
      if i == 0:
        header = line.strip()
        field_names = line.strip().split("\t")
        continue

      fields = line.strip().split("\t")
      label = fields[label_index]
      if len(fields) > len(field_names):
        # SNLI / MNLI fields sometimes contain multiple annotator labels.
        # Ignore all except the gold label.
        reformatted_fields = fields[:len(field_names)-1] + [label]
        assert len(reformatted_fields) == len(field_names)
        reformatted_line = "\t".join(reformatted_fields)
      else:
        reformatted_line = line.strip()

      if label == "-" or label == "":
        logger.info(f"Skippping line: {line}")
        continue

      if guid_index is None:
        guid = i
      else:
        guid = fields[guid_index] # PairID.
      if guid in tsv_dict:
        logger.info(f"Found clash in IDs ... skipping example {guid}.")
        continue
      tsv_dict[guid] = reformatted_line.strip()

  logger.info(f"Read {len(tsv_dict)} valid examples, with unique IDS, out of {i} from {file_path}")
  if guid_as_int:
    tsv_numeric = {int(convert_string_to_unique_number(k)): v for k, v in tsv_dict.items()}
    return tsv_numeric, header
  return tsv_dict, header

def determine_categories_by_fields(fields):
  example_categories = []
  for field in fields[:-4]:
    if field == '':
      continue
    elif ";" in field:
      example_categories.append(fine_to_coarse[field.split(";")[0]])  # Usually same coarse category.
    else:
      example_categories.append(fine_to_coarse[field])

  return example_categories


def diag_test_modifier(original_diag_tsv, output_tsv):
  """Modify the TSV file provided for Diagnostic NLI tests to follow the same
     format as the other test files for GLUE NLI."""
  diag_original, diag_headers = read_glue_tsv(original_diag_tsv, guid_index=None)
  coarse_category_counter = {name: 0 for name in category_names}

  with open(output_tsv, "w") as outfile:
    outfile.write("\t".join(FIELDS) + "\n")
    lines_with_missing_fields = 0
    multiple_categories = 0
    written = 0

    for i, (key, line) in enumerate(diag_original.items()):
      in_fields = line.strip().split("\t")

      if len(in_fields) < 8:
        # logger.info(f"Line with missing fields: {len(in_fields)} out of 8.\n  {in_fields}")
        lines_with_missing_fields += 1

      example_categories = determine_categories_by_fields(fields=in_fields)
      for ec in example_categories:
        coarse_category_counter[ec] += 1
      if len(example_categories) > 1:
        # logger.info(f"{len(category)} Categories : {category} \n {in_fields[:-4]}")
        multiple_categories += 1
      elif not len(example_categories):
        logger.info(f"No category found:\n {line}")
        # HACK: from my understanding, this is an example of factivity.
        example_categories = ["lexical_semantics"]

      guid = str(i)
      out_record = {"index": guid,
                    "captionID": guid,
                    "pairID": guid,
                    "sentence1_binary_parse": "",
                    "sentence2_binary_parse": "",
                    "sentence1_parse": "",
                    "sentence2_parse": "",
                    "gold_label": in_fields[-1],
                    "sentence2": in_fields[-2],
                    "sentence1": in_fields[-3],
                    "category": ";".join(example_categories)}
      out_fields = [out_record[field] if field in out_record else "" for field in FIELDS]
      outfile.write("\t".join(out_fields) + "\n")
      written += 1

  for c, count in coarse_category_counter.items():
    logger.info(f"Items in {c}: {count}")
    assert category_names[c] == count

  logger.info(f"Total records:               {len(diag_original)}")
  logger.info(f"Records with missing fields: {lines_with_missing_fields}.")
  logger.info(f"Records with 2+ categories:  {multiple_categories}.")
  logger.info(f"Total records written:       {written} to {output_tsv}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--diagnostics_input", "-i", type=str)
  parser.add_argument("--output", "-o", type=str)

  args = parser.parse_args()
  logger.info(args)

  diag_test_modifier(args.diagnostics_input, args.output)
