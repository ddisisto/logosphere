"""
Test init_parser.py functionality
"""

from pathlib import Path
from init_parser import load_init_file
from config import INIT_TEMPLATE, INIT_SIGNATURE


def test_load_init_template():
    """Test parsing init-template.txt"""
    print("Testing load_init_file with init-template.txt...")

    messages, signature = load_init_file(INIT_TEMPLATE)

    print(f"  ✓ Parsed init template")
    print(f"  ✓ Extracted {len(messages)} seed messages")
    print(f"  ✓ Signature: '{signature}'")

    assert signature == INIT_SIGNATURE, f"Expected '{INIT_SIGNATURE}', got '{signature}'"
    assert len(messages) > 0, "Should have seed messages"

    # Check that signature is appended to all messages
    for msg in messages:
        assert msg.endswith(f"\n\n{INIT_SIGNATURE}"), "Signature should be appended to messages"

    print(f"\n  Sample messages:")
    for i, msg in enumerate(messages[:3]):
        # Show message without signature for readability
        msg_content = msg.replace(f"\n\n{INIT_SIGNATURE}", "")
        print(f"    [{i}]: {msg_content.strip()}")

    print(f"\n  ✓ All {len(messages)} messages have signature appended")


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
