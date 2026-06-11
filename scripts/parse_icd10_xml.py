import xml.etree.ElementTree as ET
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

"""
ICD-10-CM XML parser.

Download the official tabular file from:
  https://www.cms.gov/medicare/coding-billing/icd-10-codes
  → April 1, 2026 Code Tables, Tabular and Index (ZIP)
  → extract icd10cm_tabular_2026.xml → place at data/raw/icd10cm_tabular_2026.xml
"""


def parse_icd10_xml(xml_path: str, out_path: str) -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    codes: dict[str, str] = {}
    for diag in root.iter("diag"):
        code_el = diag.find("name")
        desc_el = diag.find("desc")
        if code_el is None or desc_el is None:
            continue

        code = code_el.text.strip()
        desc = desc_el.text.strip()

        # Billable = has a decimal  OR  is an extended code ending in a letter
        is_billable = ("." in code) or (len(code) > 3 and code[-1].isalpha())
        if is_billable:
            codes[code] = desc

    with open(out_path, "w") as f:
        json.dump(codes, f, indent=2)

    print(f"Parsed {len(codes)} billable ICD-10-CM codes → {out_path}")
    return len(codes)


if __name__ == "__main__":
    raw_data_dir = PROJECT_ROOT / "data" / "raw"
    xml_input = raw_data_dir / "icd10cm_tabular_2026.xml"
    json_output = raw_data_dir / "icd10cm_codes_2026.json"

    if not xml_input.exists():
        raise FileNotFoundError(
            f"XML file not found: {xml_input}\n"
            "Download it from https://www.cms.gov/medicare/coding-billing/icd-10-codes"
        )

    parse_icd10_xml(str(xml_input), str(json_output))