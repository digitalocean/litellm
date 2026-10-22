import subprocess
import sys
from typing import Final


def test_pass_through_streaming_handler_imports_without_redis_package() -> None:
    code: Final = r"""
import builtins
import importlib

real_import = builtins.__import__

def block_redis(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] == "redis":
        raise ModuleNotFoundError("No module named 'redis'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = block_redis
mod = importlib.import_module("litellm.proxy.pass_through_endpoints.streaming_handler")
assert hasattr(mod, "PassThroughStreamingHandler")
buf = importlib.import_module("litellm.proxy.db.db_transaction_queue.redis_update_buffer")
assert issubclass(buf.RedisError, Exception)
print("ok")
"""
    result: Final = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "No module named" in (result.stderr + result.stdout):
        missing: Final = next(
            (
                line
                for line in (result.stderr + result.stdout).splitlines()
                if "No module named" in line and "redis" not in line
            ),
            None,
        )
        if missing is not None:
            import pytest

            pytest.skip(f"dev env missing dependency for this smoke: {missing}")
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
