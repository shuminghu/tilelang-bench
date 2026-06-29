"""LitellmModel variant hardened for OpenAI-compatible gateways (e.g. OpenCode Zen).

Some providers (DeepSeek, GLM, ...) reject messages that carry keys litellm copies
back from a previous response -- notably `provider_specific_fields` and assorted
`null` fields. The stock LitellmModel only strips `extra`, so a multi-turn run dies
on the 2nd request with: "Extra inputs are not permitted, field:
'messages[*].provider_specific_fields'". We scrub those before every request.
"""
from minisweagent.models.litellm_model import LitellmModel

_DROP_KEYS = {"provider_specific_fields", "reasoning_content", "annotations",
              "audio", "refusal", "function_call"}


def _clean(msg: dict) -> dict:
    out = {}
    for k, v in msg.items():
        if k in _DROP_KEYS:
            continue
        # Drop null-valued optional fields; keep content even if empty string.
        if v is None and k != "content":
            continue
        out[k] = v
    return out


class CleanLitellmModel(LitellmModel):
    def _prepare_messages_for_api(self, messages):
        prepared = super()._prepare_messages_for_api(messages)
        return [_clean(m) for m in prepared]
