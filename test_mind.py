"""
Verification tests for mind.py

Tests parsing logic and message formatting.
API invocation test requires actual API call (marked clearly).
"""

from mind import format_input, parse_output, invoke_mind
from config import SYSTEM_PROMPT, SEED_MESSAGES


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
    """Test basic output parsing with proper termination."""
    print("\nTesting parse_output - basic termination...")

    raw = """I should think about this first.
---
This is message one.
---
This is message two.
---

---"""

    thinking, messages, completed, signature = parse_output(raw)

    assert thinking == "I should think about this first."
    assert len(messages) == 2
    assert messages[0] == "This is message one."
    assert messages[1] == "This is message two."
    assert completed == True
    assert signature == ""
    print("  ✓ Proper termination parsed correctly")


def test_parse_output_no_termination():
    """Test output without termination - should drop all messages."""
    print("\nTesting parse_output - no termination...")

    raw = """Thinking phase.
---
Message one.
---
Message two.
---
Message three."""

    thinking, messages, completed, signature = parse_output(raw)

    assert thinking == "Thinking phase."
    assert len(messages) == 0  # Not completed, all dropped
    assert completed == False
    assert signature == ""
    print("  ✓ Incomplete output drops all messages")


def test_parse_output_with_signature():
    """Test output with context signature."""
    print("\nTesting parse_output - with signature...")

    raw = """Private thinking here.
---
First message.
---
Second message.
---

---
context-sig"""

    thinking, messages, completed, signature = parse_output(raw)

    assert thinking == "Private thinking here."
    assert len(messages) == 2
    assert completed == True
    assert signature == "context-sig"
    # Signature should be appended to messages
    assert messages[0] == "First message.\n\ncontext-sig"
    assert messages[1] == "Second message.\n\ncontext-sig"
    print("  ✓ Signature extracted and appended to messages")


def test_parse_output_signature_truncation():
    """Test signature truncation to max length."""
    print("\nTesting parse_output - signature truncation...")

    long_sig = "x" * 100
    raw = f"""Thinking.
---
Message.
---

---
{long_sig}"""

    thinking, messages, completed, signature = parse_output(raw, signature_max_len=32)

    assert len(signature) == 32
    assert signature == "x" * 32
    assert messages[0] == f"Message.\n\n{'x' * 32}"
    print("  ✓ Signature truncated to max length")


def test_parse_output_silence():
    """Test valid silence - terminated immediately with no messages."""
    print("\nTesting parse_output - valid silence...")

    raw = """I choose silence.
---

---"""

    thinking, messages, completed, signature = parse_output(raw)

    assert thinking == "I choose silence."
    assert len(messages) == 0
    assert completed == True
    assert signature == ""
    print("  ✓ Valid silence (immediate termination) recognized")


def test_parse_output_no_thinking():
    """Test output that jumps straight to messages."""
    print("\nTesting parse_output - no thinking block...")

    raw = """---
Direct message.
---

---"""

    thinking, messages, completed, signature = parse_output(raw)

    assert thinking == ""
    assert len(messages) == 1
    assert messages[0] == "Direct message."
    assert completed == True
    print("  ✓ Output without thinking block handled")


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
    # result = invoke_mind(SYSTEM_PROMPT, SEED_MESSAGES[:2], token_limit=500)
    # print(f"\n  API Response:")
    # print(f"    Thinking: {result['thinking'][:50]}...")
    # print(f"    Transmitted: {len(result['transmitted'])} messages")
    # print(f"    Completed: {result['completed']}")
    # print(f"    Signature: {result['signature']}")
    # print(f"    Tokens used: {result['tokens_used']}")
    # assert 'thinking' in result
    # assert 'transmitted' in result
    # assert isinstance(result['completed'], bool)


def run_all_tests():
    """Run all mind.py tests."""
    print("=" * 60)
    print("MIND.PY VALIDATION")
    print("=" * 60)

    test_format_input()
    test_parse_output_basic()
    test_parse_output_no_termination()
    test_parse_output_with_signature()
    test_parse_output_signature_truncation()
    test_parse_output_silence()
    test_parse_output_no_thinking()
    test_invoke_mind_demo()

    print()
    print("=" * 60)
    print("✅ ALL MIND TESTS PASSED")
    print("=" * 60)
    print()
    print("Component validated:")
    print("  - Input formatting correct")
    print("  - Output parsing handles all cases")
    print("  - Signature extraction and appending works")
    print("  - Termination logic correct")
    print("  - invoke_mind ready (API test available)")


if __name__ == "__main__":
    run_all_tests()
