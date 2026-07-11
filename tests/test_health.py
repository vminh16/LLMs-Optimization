import unittest
from unittest.mock import patch

import httpx

from inference_opt.clients import health


class HealthTest(unittest.TestCase):
    def test_startup_grace_is_not_counted_against_poll_timeout(self):
        events = []

        def sleep(seconds):
            events.append(("sleep", seconds))

        def monotonic():
            events.append(("monotonic", None))
            return 0.0 if len([event for event in events if event[0] == "monotonic"]) == 1 else 301.0

        with (
            patch.object(health, "fetch_models", side_effect=httpx.ConnectError("offline")),
            patch.object(health.time, "sleep", side_effect=sleep),
            patch.object(health.time, "monotonic", side_effect=monotonic),
        ):
            with self.assertRaises(TimeoutError):
                health.wait_for_model(
                    "http://localhost:8000/v1",
                    "Qwen3.5-2B",
                    request_timeout_s=5.0,
                    startup_grace_s=60.0,
                    poll_interval_s=5.0,
                    total_timeout_s=300.0,
                    stable_successes=2,
                )

        self.assertEqual(events[0], ("sleep", 60.0))

    def test_wait_for_model_requires_consecutive_successes(self):
        payload = {"data": [{"id": "Qwen3.5-2B"}]}
        with (
            patch.object(
                health,
                "fetch_models",
                side_effect=[httpx.ConnectError("offline"), payload, payload],
            ) as fetch,
            patch.object(health.time, "sleep") as sleep,
            patch("builtins.print") as output,
        ):
            result = health.wait_for_model(
                "http://localhost:8000/v1",
                "Qwen3.5-2B",
                request_timeout_s=5.0,
                startup_grace_s=60.0,
                poll_interval_s=5.0,
                total_timeout_s=300.0,
                stable_successes=2,
            )

        self.assertEqual(result, payload)
        self.assertEqual(fetch.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60.0, 5.0, 5.0])
        output.assert_not_called()

    def test_wait_for_model_reports_last_failure_on_timeout(self):
        with (
            patch.object(
                health,
                "fetch_models",
                side_effect=httpx.ConnectError("offline"),
            ),
            patch.object(health.time, "sleep"),
            patch.object(health.time, "monotonic", side_effect=[0.0, 301.0]),
        ):
            with self.assertRaisesRegex(TimeoutError, "offline"):
                health.wait_for_model(
                    "http://localhost:8000/v1",
                    "Qwen3.5-2B",
                    request_timeout_s=5.0,
                    startup_grace_s=60.0,
                    poll_interval_s=5.0,
                    total_timeout_s=300.0,
                    stable_successes=2,
                )


if __name__ == "__main__":
    unittest.main()
