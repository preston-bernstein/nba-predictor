"""Dead-code findings that existed when the vulture gate was adopted (2026-11-27).

Remove an entry when its code is deleted, or keep it with a comment saying which framework
or reflection uses the name.
"""

# ruff: noqa: B018, F821 -- a vulture whitelist is bare names, one per line, by design
_.row_factory  # unused attribute (src/events/ledger.py:100)
_.count_events  # unused method (src/events/ledger.py:121)
_value_error_handler  # unused function (src/service/errors.py:31)
_key_error_handler  # unused function (src/service/errors.py:37)
health  # unused function (src/service/routes.py:37)
ledger_summary  # unused function (src/service/routes.py:49)
ok  # unused variable (src/service/schemas.py:17)
model_config  # unused variable (src/service/schemas.py:35)
delta_off  # unused variable (src/service/schemas.py:36)
delta_def  # unused variable (src/service/schemas.py:37)
delta_rest  # unused variable (src/service/schemas.py:38)
delta_elo  # unused variable (src/service/schemas.py:39)
as_of  # unused variable (src/service/schemas.py:45)
accuracy  # unused variable (src/service/schemas.py:54)
