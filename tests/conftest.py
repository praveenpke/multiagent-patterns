import os

# Force deterministic scripted fake models even if a developer machine has a key.
os.environ["LGP_FORCE_FAKE"] = "1"
