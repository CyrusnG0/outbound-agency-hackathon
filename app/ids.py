"""
ID generation for the outbound-agency pipeline.

Every record in this system gets a short, prefixed unique identifier (e.g. "tgt_3f9a2b1c").
The prefix makes ids self-describing in logs — you can tell a "tgt_..." (target) from an
"acc_..." (account) or "msg_..." (message) at a glance when debugging "steps" or "write_log"
rows. This matters because those tables are the audit trail: every row references at least
one entity id, and without prefixes you'd have to join just to know what kind of thing was
being acted on.

We use uuid4 hex truncated to 12 chars rather than a full 36-char UUID. 12 hex chars gives
~2.8e14 possible values per prefix — more than enough for a single-operator harness — while
keeping log lines and JSON payloads readable.
"""

# We use uuid4 (not random, secrets, or a counter) because:
# - uuid4 is a cryptographically-random 128-bit value — collision risk is negligible even
#   at this harness's scale, and there's no coordination needed across processes or restarts.
# - `random` is seeded and predictable once the seed is known; `secrets` would also work
#   but uuid4 carries the same CSPRNG guarantees from the stdlib with a cleaner API for ids.
# - A monotonic counter (AUTOINCREMENT, sequence) leaks order and count, ties us to a
#   single-writer model, and would require a coordination layer. uuid4 needs none of that.
import uuid


def new_id(prefix: str) -> str:
    """Generate a short, prefixed unique id, e.g. new_id('tgt') -> 'tgt_3f9a2b1c'."""
    # Build the id in two parts: the prefix (e.g. "tgt") + underscore, then 12 hex chars
    # from a uuid4 digest. 12 hex chars = 48 bits = ~2.8e14 values per prefix — far beyond
    # what a single-operator harness will ever produce, while keeping log lines compact.
    # Truncating to [:12] is a deliberate trade-off:
    #   [:4]  →  16 bits, 65k values — collision-likely under moderate volume (birthday
    #            problem: ~50% chance of collision after only ~300 ids per prefix).
    #   [:6]  →  24 bits, 16M values — safe for toy usage but tight for years of daily ops.
    #   [:12] →  48 bits, 281 trillion values — collision probability is astronomically low
    #            even if the harness runs for decades at millions of ids per day.
    #   [:32] → full 128 bits — safest, but makes log lines and JSON payloads 20 chars
    #           longer per id with no practical benefit at this scale.
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
