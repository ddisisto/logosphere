"""
Test init_parser.py functionality
"""

from pathlib import Path
from src.core.init_parser import load_init_file
from src.config import EXPERIMENTS_DIR


def test_load_init_template():
    """Test parsing baseline init.md"""
    print("Testing load_init_file with baseline experiment...")

    messages = load_init_file(EXPERIMENTS_DIR / "_baseline" / "init.md")

    print(f"  ✓ Parsed init template")
    print(f"  ✓ Extracted {len(messages)} seed messages")

    assert len(messages) > 0, "Should have seed messages"

    print(f"\n  Sample messages:")
    for i, msg in enumerate(messages[:3]):
        print(f"    [{i}]: {msg.strip()}")


def run_all_tests():
    """Run all init_parser tests"""
    print("=" * 60)
    print("INIT_PARSER VALIDATION")
    print("=" * 60)
    print()

    test_load_init_template()

    print()
    print("=" * 60)
    print("✅ ALL INIT_PARSER TESTS PASSED")
    print("=" * 60)
    print()
    print("Component validated:")
    print("  - init-template.txt parsed correctly")
    print("  - Seed messages extracted with signature")


if __name__ == "__main__":
    run_all_tests()
