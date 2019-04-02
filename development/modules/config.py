from pathlib import Path

# Edit PYTHONPATH
PACKAGE_DIR = Path(__file__).parent / "development" / "modules"

# Directory for specification of baselines
SPEC_DIR = PACKAGE_DIR / "respy" / "tests" / "resources"
SPECS = ["kw_data_one", "kw_data_two", "kw_data_three"]
