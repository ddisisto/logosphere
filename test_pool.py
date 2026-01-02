"""Quick verification test for pool.py"""

from pool import Pool
from config import SEED_MESSAGES, M_ACTIVE_POOL, K_SAMPLES


def test_pool():
    print("Testing Pool class...")

    # Create pool
    pool = Pool(max_active=M_ACTIVE_POOL)
    print(f"✓ Pool created (max_active={M_ACTIVE_POOL})")

    # Add seed messages
    for msg in SEED_MESSAGES:
        pool.add_message(msg)
    print(f"✓ Added {len(SEED_MESSAGES)} seed messages")

    # Check sizes
    assert pool.size() == len(SEED_MESSAGES)
    assert pool.active_size() == len(SEED_MESSAGES)
    print(f"✓ Pool size: {pool.size()}, Active size: {pool.active_size()}")

    # Sample
    sample = pool.sample(K_SAMPLES)
    assert len(sample) == K_SAMPLES
    print(f"✓ Sampled {len(sample)} messages")
    print(f"  Sample: {sample}")

    # Add more messages to exceed active pool
    for i in range(20):
        pool.add_message(f"Message {i}")

    print(f"✓ Added 20 more messages")
    print(f"  Total size: {pool.size()}")
    print(f"  Active size: {pool.active_size()}")
    assert pool.active_size() == M_ACTIVE_POOL
    print(f"✓ Active pool correctly limited to {M_ACTIVE_POOL}")

    # Verify sampling only from active pool
    sample = pool.sample(K_SAMPLES)
    all_msgs = pool.get_all()
    active_msgs = pool.get_active()

    for msg in sample:
        assert msg in active_msgs, "Sample should only come from active pool"
    print(f"✓ Sampling correctly uses only active pool (tail {M_ACTIVE_POOL})")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_pool()
