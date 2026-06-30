"""LitellmModel variant hardened for OpenAI-compatible gateways (e.g. OpenCode Zen).

Some providers (DeepSeek, GLM, ...) reject messages that carry keys litellm copies
back from a previous response -- notably `provider_specific_fields` and assorted
`null` fields. The stock LitellmModel only strips `extra`, so a multi-turn run dies
on the 2nd request with: "Extra inputs are not permitted, field:
'messages[*].provider_specific_fields'". We scrub those before every request.
"""
import copy

from minisweagent.models.litellm_model import LitellmModel

_DROP_KEYS = {"provider_specific_fields", "reasoning_content", "annotations",
              "audio", "refusal", "function_call"}

# model_kwargs entries that carry credentials and must never be written to logs.
_SECRET_KWARGS = {"api_key", "api_base_key", "aws_secret_access_key",
                  "aws_session_token", "azure_ad_token", "vertex_credentials"}
_REDACTED = "***REDACTED***"


def _redact_model_kwargs(model_cfg: dict) -> dict:
    """Return a copy of a serialized model config with credential kwargs masked."""
    kwargs = model_cfg.get("model_kwargs")
    if isinstance(kwargs, dict):
        for k in list(kwargs):
            if k in _SECRET_KWARGS or "key" in k.lower() or "token" in k.lower():
                kwargs[k] = _REDACTED
    return model_cfg


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

    def serialize(self) -> dict:
        # The base implementation dumps the full config (including model_kwargs
        # with the raw api_key) into the persisted trajectory. Mask credentials
        # so secrets never land in runs/*.trajectory.json.
        data = copy.deepcopy(super().serialize())
        try:
            _redact_model_kwargs(data["info"]["config"]["model"])
        except (KeyError, TypeError):
            pass
        return data

    def get_template_vars(self, **kwargs):
        # Templates can render config values; keep credentials out of any text.
        return _redact_model_kwargs(copy.deepcopy(super().get_template_vars(**kwargs)))
