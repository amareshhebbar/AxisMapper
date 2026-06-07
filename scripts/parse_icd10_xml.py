import xml.etree.ElementTree as ET
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

"""
I downloaded the ICD10 code from the https://www.cms.gov/medicare/coding-billing/icd-10-codes
here i download what i needed...>>April 1, 2026 Code Tables, Tabular and Index (ZIP)<< 

i did the google agent to ask which one is the best and why we need that...it suggest me this
and when i ran the coverstion then i got 44k+ code

and last but not least that this is the first file which i ran cause dataset is important than method of ai finetune we do

and also github doesn;t allow me add the dataset but when you download from there...>>icd10cm_tabular_2026.xml<<
add this file inside the data/raw(if you wanna run my code)
"""

def parse_icd10_xml(xml_path: str, out_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    codes = {}
    for diag in root.iter("diag"):
        code_el = diag.find("name")
        desc_el = diag.find("desc")
        if code_el is not None and desc_el is not None:
            code = code_el.text.strip()
            desc = desc_el.text.strip()
            
            if "." in code or (len(code) > 3 ) and code[-1].isalpha():
                codes[code] = desc
    with open(out_path, "w") as f:
        json.dump(codes, f, indent=2)
    
    print(f"Parsed {len(codes)} billade ICD10-CM Code")
    
if __name__ == "__main__":
    raw_data_dir = PROJECT_ROOT / "data" / "raw"
    xml_input = raw_data_dir / "icd10cm_tabular_2026.xml"
    json_output = raw_data_dir / "icd10cm_codes_2026.json"
    
    parse_icd10_xml(xml_input, json_output)