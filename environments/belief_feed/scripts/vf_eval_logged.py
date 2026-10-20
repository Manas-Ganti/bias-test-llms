"""vf-eval with three fixes for running against a local vLLM 0.8.5 server.

1. aiohttp's server errors go to stderr. verifiers routes only its own logger to eval.log,
   so an unhandled exception inside the interception server reaches the harness as a bare
   "500 Server got itself in trouble" and its traceback is dropped.
2. A tool call with no "arguments" key is read as "{}". vLLM 0.8.5's streaming hermes parser
   sends no arguments delta for a zero-argument call (next_page()), and verifiers 0.4.0's
   parse_message indexes native["arguments"] directly, so the next request raised KeyError.
   Newer vLLM (the ARC API) sends "{}" and never hits this.

Usage is the same as vf-eval:  python scripts/vf_eval_logged.py @ configs/arc.toml ...
"""

import logging
import os
import sys

import verifiers.v1.agent as vf_agent
import verifiers.v1.dialects as vf_dialects
import verifiers.v1.dialects.chat as vf_chat
from verifiers.v1.cli.eval.main import main

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
for name in ("aiohttp.server", "aiohttp.web", "aiohttp.access"):
    log = logging.getLogger(name)
    log.addHandler(handler)
    log.setLevel(logging.ERROR)

_parse_message = vf_chat.parse_message


def parse_message(raw: dict):
    if raw.get("role") == "assistant":
        for call in raw.get("tool_calls") or []:
            native = call.get("function")
            if isinstance(native, dict) and native.get("arguments") is None:
                native["arguments"] = "{}"
    return _parse_message(raw)


vf_chat.parse_message = parse_message
vf_dialects.parse_message = parse_message
vf_agent.parse_message = parse_message

# 3. BF_UNSTREAMED_UPSTREAM=1: ask the provider for whole completions and let the interception
#    server frame the SSE the harness reads, the path verifiers already uses for training.
#    vLLM 0.8.5's *streaming* tool parser also drops the finish_reason chunk when a reply is
#    cut off ("Model stream ended before a completion finished"); non-streaming replies always
#    carry it.
if os.environ.get("BF_UNSTREAMED_UPSTREAM") == "1":
    import verifiers.v1.interception.server as vf_server
    from verifiers.v1.clients.eval import EvalClient
    from verifiers.v1.configs.client import EvalClientConfig, TrainClientConfig

    # handle_request relays a stream only when the client is not a TrainClientConfig.
    vf_server.TrainClientConfig = (TrainClientConfig, EvalClientConfig)
    _get_response = EvalClient.get_response

    async def get_response(self, dialect, body, *args, **kwargs):
        if body.get("stream"):
            body = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
        return await _get_response(self, dialect, body, *args, **kwargs)

    EvalClient.get_response = get_response

if __name__ == "__main__":
    sys.exit(main())
