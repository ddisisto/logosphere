"""
Verification tests for mind.py

Tests parsing logic and message formatting.
API invocation test requires actual API call (marked clearly).
"""

from mind import format_input, parse_output, invoke_mind
from config import SYSTEM_PROMPT


def test_format_input():
    """Test input formatting."""
    print("Testing format_input...")

    # Basic case
    result = format_input("System prompt", ["msg1", "msg2"])
    expected = "System prompt\n---\nmsg1\n---\nmsg2"
    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
    print("  ✓ Basic formatting correct")

    # Empty messages
    result = format_input("System prompt", [])
    assert result == "System prompt"
    print("  ✓ Empty messages handled")

    # Single message
    result = format_input("Prompt", ["single"])
    assert result == "Prompt\n---\nsingle"
    print("  ✓ Single message correct")


def test_parse_output_basic():
    """Test basic output parsing."""
    print("\nTesting parse_output - basic case...")

    raw = """I should think about this first.
---
This is message one.
---
This is message two."""

    thinking, messages = parse_output(raw)

    assert thinking == "I should think about this first."
    assert len(messages) == 2
    assert messages[0] == "This is message one."
    assert messages[1].strip() == "This is message two."  # Last block may have trailing newline
    print("  ✓ Basic parsing correct")


def test_parse_output_no_messages():
    """Test output with only thinking (no delimiter)."""
    print("\nTesting parse_output - thinking only...")

    raw = """Just thinking, no messages"""

    thinking, messages = parse_output(raw)

    assert thinking.strip() == "Just thinking, no messages"
    assert len(messages) == 0
    print("  ✓ Thinking-only (silence) parsed correctly")


def test_parse_output_no_thinking():
    """Test output that jumps straight to messages."""
    print("\nTesting parse_output - no thinking block...")

    raw = """---
Direct message.
---
Another message."""

    thinking, messages = parse_output(raw)

    assert thinking == ""
    assert len(messages) == 2
    assert messages[0] == "Direct message."
    assert messages[1].strip() == "Another message."
    print("  ✓ Output without thinking block handled")


def test_parse_output_blank_messages():
    """Test that blank blocks are valid messages."""
    print("\nTesting parse_output - blank messages...")

    raw = """Thinking.
---
Message one.
---

---
Message two."""

    thinking, messages = parse_output(raw)

    assert thinking == "Thinking."
    assert len(messages) == 3
    assert messages[0] == "Message one."
    assert messages[1].strip() == ""  # Blank message is valid
    assert messages[2].strip() == "Message two."
    print("  ✓ Blank messages transmitted as valid")


def test_parse_output_truncated():
    """Test truncated output (partial is valid)."""
    print("\nTesting parse_output - truncated output...")

    raw = """Thinking
---
Complete message
---
Partial mess"""  # No trailing delimiter

    thinking, messages = parse_output(raw)

    assert thinking == "Thinking"
    assert len(messages) == 2
    assert messages[0] == "Complete message"
    assert messages[1].strip() == "Partial mess"  # Partial message is transmitted
    print("  ✓ Truncated/partial output transmitted")


def test_invoke_mind_demo():
    """
    Demonstration of invoke_mind function.

    WARNING: This makes an actual API call and costs tokens.
    Uncomment to run actual test.
    """
    print("\nTesting invoke_mind (DEMO - not executing)...")
    print("  To test API invocation, uncomment the code in test_invoke_mind_demo()")
    print("  This will make a real API call with minimal token usage.")

    # UNCOMMENT TO TEST:
    # result = invoke_mind(SYSTEM_PROMPT, ["test message"], token_limit=500)
    # print(f"\n  API Response:")
    # print(f"    Thinking: {result['thinking'][:50]}...")
    # print(f"    Transmitted: {len(result['transmitted'])} messages")
    # print(f"    Tokens used: {result['tokens_used']}")
    # assert 'thinking' in result
    # assert 'transmitted' in result


def run_all_tests():
    """Run all mind.py tests."""
    print("=" * 60)
    print("MIND.PY VALIDATION")
    print("=" * 60)

    test_format_input()
    test_parse_output_basic()
    test_parse_output_no_messages()
    test_parse_output_no_thinking()
    test_parse_output_blank_messages()
    test_parse_output_truncated()
    test_invoke_mind_demo()

    print()
    print("=" * 60)
    print("✅ ALL MIND TESTS PASSED")
    print("=" * 60)
    print()
    print("Component validated:")
    print("  - Input formatting correct")
    print("  - Output parsing handles all cases")
    print("  - No termination requirement")
    print("  - Blank and partial messages valid")
    print("  - invoke_mind ready (API test available)")


if __name__ == "__main__":
    run_all_tests()
