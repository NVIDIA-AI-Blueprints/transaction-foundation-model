"""Enable ``python -m loom <verb> …`` as the engine entry point.

The standalone agentic CLI (the Node/Pi front door, the user-facing ``loom``)
drives this Python engine by invoking ``python -m loom <verb> --json`` with a
resolved interpreter — so it does not depend on a ``loom`` console script being
on ``PATH`` and there is no name collision with the agent command. This module
delegates to the exact same :func:`loom.cli.main` the ``loom`` console script
(``pyproject.toml`` ``[project.scripts]``) uses.
"""

from __future__ import annotations

import sys

from loom.cli import main

if __name__ == "__main__":
    sys.exit(main())
