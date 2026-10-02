"""Pinned policy and safety-model pairs supported by the standalone trainer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    policy: str
    policy_revision: str
    policy_parameters: int
    safety: str
    safety_revision: str
    safety_parameters: int


LARGE = ModelProfile(
    policy="Qwen/Qwen2.5-3B-Instruct",
    policy_revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
    policy_parameters=3_085_938_688,
    safety="Qwen/Qwen3Guard-Gen-4B",
    safety_revision="6ec42827da0c1ff11e7a49dc269d2e810d27e108",
    safety_parameters=4_411_424_256,
)

SMALL = ModelProfile(
    policy="Qwen/Qwen2.5-0.5B-Instruct",
    policy_revision="7ae557604adf67be50417f59c2c2f167def9a775",
    policy_parameters=494_032_768,
    safety="Qwen/Qwen3Guard-Gen-0.6B",
    safety_revision="3706d237aa5d05ee7f6c8274c208a37211e9fd65",
    safety_parameters=596_049_920,
)

PROFILES = {profile.policy: profile for profile in (LARGE, SMALL)}


def profile_for_policy(model_path: str) -> ModelProfile:
    try:
        return PROFILES[model_path]
    except KeyError as error:
        raise ValueError(f"Unsupported policy model: {model_path}") from error
