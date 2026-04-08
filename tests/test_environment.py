from __future__ import annotations

from src.environment import PipelineDebugEnv
from src.models import PipelineAction


def test_reset_returns_valid_observation() -> None:
    for task_id in ["easy", "medium", "hard"]:
        env = PipelineDebugEnv(task_id=task_id)
        obs = env.reset()
        assert obs.task_id == task_id
        assert isinstance(obs.current_schema, dict)
        assert isinstance(obs.expected_schema, dict)
        assert isinstance(obs.preview_rows, list)


def test_step_cast_column_success() -> None:
    env = PipelineDebugEnv(task_id="easy")
    env.reset()
    obs, reward, _done, info = env.step(
        PipelineAction(command="cast_column", params={"column": "revenue", "dtype": "float64"})
    )
    assert obs.last_action_result == "success"
    assert reward > 0
    assert info["action_result"] == "success"


def test_step_invalid_command_returns_invalid() -> None:
    env = PipelineDebugEnv(task_id="easy")
    env.reset()
    obs, _reward, _done, info = env.step(PipelineAction(command="bad_command", params={}))
    assert obs.last_action_result == "invalid"
    assert info["action_result"] == "invalid"


def test_reward_determinism() -> None:
    actions = [
        PipelineAction(command="cast_column", params={"column": "revenue", "dtype": "float64"}),
        PipelineAction(command="fix_date_format", params={"column": "date", "format": "%d-%m-%Y"}),
    ]

    env1 = PipelineDebugEnv(task_id="easy")
    env2 = PipelineDebugEnv(task_id="easy")
    env1.reset()
    env2.reset()

    rewards1 = [env1.step(action)[1] for action in actions]
    rewards2 = [env2.step(action)[1] for action in actions]
    assert rewards1 == rewards2


def test_episode_terminates() -> None:
    env = PipelineDebugEnv(task_id="easy")
    env.reset()
    env.step(PipelineAction(command="cast_column", params={"column": "revenue", "dtype": "float64"}))
    env.step(PipelineAction(command="fix_date_format", params={"column": "date", "format": "%d-%m-%Y"}))
    _obs, _reward, done, _info = env.step(
        PipelineAction(command="fill_nulls", params={"column": "units_sold", "value": 0})
    )
    assert done is True


def test_no_change_penalty() -> None:
    env = PipelineDebugEnv(task_id="easy")
    env.reset()
    _obs, reward, _done, _info = env.step(
        PipelineAction(command="fill_nulls", params={"column": "units_sold", "value": -1})
    )
    _obs, reward2, _done, info = env.step(
        PipelineAction(command="fill_nulls", params={"column": "units_sold", "value": -1})
    )
    assert info["action_result"] in ("no_change", "success")
    assert reward2 <= reward
