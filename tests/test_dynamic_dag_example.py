from importlib import import_module
from types import SimpleNamespace

import examples.dynamic_dag_example as ex


def test_dynamic_dag_example_runs(capsys, monkeypatch):
    accel_mod = import_module("tygent.accelerate")

    async def patched_optimize_async_function(func, args, kwargs):
        return await func(*args, **kwargs)

    monkeypatch.setattr(
        accel_mod, "_optimize_async_function", patched_optimize_async_function
    )

    class DummyClient:
        class DummyCompletions:
            async def create(self, model=None, messages=None):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="LLM output"))
                    ]
                )

        def __init__(self):
            self.chat = SimpleNamespace(completions=DummyClient.DummyCompletions())

    monkeypatch.setattr(ex, "_get_openai_client", lambda: DummyClient())

    import asyncio

    asyncio.run(ex.main())

    captured = capsys.readouterr().out
    print(captured)
    assert "Dynamic DAG Modification Example" in captured
    assert "Performance improvement" in captured
