"""
Unit tests for the retry decorator.
"""

import pytest

from src.core.retry import async_retry


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        call_count = 0

        @async_retry(max_retries=3, base_delay=0.01)
        async def succeeds():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await succeeds()
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        call_count = 0

        @async_retry(max_retries=3, base_delay=0.01)
        async def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        result = await fails_twice()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        call_count = 0

        @async_retry(max_retries=2, base_delay=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            await always_fails()

        assert call_count == 3  # 1 initial + 2 retries
