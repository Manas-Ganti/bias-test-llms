"""vf-eval with aiohttp's server errors sent to stderr.

verifiers routes only its own logger to eval.log, so an unhandled exception inside the
interception server reaches the harness as a bare "500 Server got itself in trouble" and its
traceback is dropped ("No handlers could be found for logger aiohttp.server"). Usage is the
same as vf-eval:  python scripts/vf_eval_logged.py @ configs/arc.toml ...
"""

import logging
import sys

from verifiers.v1.cli.eval.main import main

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
for name in ("aiohttp.server", "aiohttp.web", "aiohttp.access"):
    log = logging.getLogger(name)
    log.addHandler(handler)
    log.setLevel(logging.ERROR)

if __name__ == "__main__":
    sys.exit(main())
