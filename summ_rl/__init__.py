"""RL core for summarization. Heavy imports (torch, transformers) are loaded lazily so that
`summ_rl.text` and `summ_rl.rewards` stay usable from the standard-library-only data scripts."""

__all__ = ['RewardConfig', 'reward_breakdown', 'GroupRLConfig', 'GroupRLTrainer']


def __getattr__(name):
    if name in ('RewardConfig', 'reward_breakdown'):
        from . import rewards
        return getattr(rewards, name)
    if name in ('GroupRLConfig', 'GroupRLTrainer'):
        from . import grpo
        return getattr(grpo, name)
    raise AttributeError(name)
