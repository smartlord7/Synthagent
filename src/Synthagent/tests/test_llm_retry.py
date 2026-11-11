import io
import os
import sys
import json
import unittest
import urllib.error
from email.message import Message
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.llm_client import LocalOpenAIProvider, OpenAIProvider
import run_batch_pipeline as rbp


def _http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError("http://x", code, "err", headers, io.BytesIO(b"{}"))


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


OK = {"choices": [{"message": {"content": "hi"}}]}


class TestRetryDelay(unittest.TestCase):
    def test_429_and_5xx_are_retried(self):
        for code in (429, 500, 502, 503, 504):
            self.assertIsNotNone(LocalOpenAIProvider._retry_delay(_http_error(code), 0), code)

    def test_other_http_errors_are_not_retried(self):
        # 400/422 in particular are handled by generate_bulk_data (context shrinking).
        for code in (400, 401, 404, 422):
            self.assertIsNone(LocalOpenAIProvider._retry_delay(_http_error(code), 0), code)

    def test_dns_failure_and_connection_reset_are_retried(self):
        self.assertIsNotNone(LocalOpenAIProvider._retry_delay(urllib.error.URLError("getaddrinfo failed"), 0))
        self.assertIsNotNone(LocalOpenAIProvider._retry_delay(ConnectionResetError("forcibly closed"), 0))

    def test_timeouts_are_not_retried(self):
        self.assertIsNone(LocalOpenAIProvider._retry_delay(TimeoutError("timed out"), 0))
        self.assertIsNone(LocalOpenAIProvider._retry_delay(urllib.error.URLError(TimeoutError("timed out")), 0))

    def test_backoff_grows_and_is_capped(self):
        delays = [LocalOpenAIProvider._retry_delay(_http_error(429), a) for a in range(10)]
        self.assertEqual(delays[:3], [2, 4, 8])
        self.assertEqual(max(delays), 120)

    def test_retry_after_header_is_honoured(self):
        self.assertEqual(LocalOpenAIProvider._retry_delay(_http_error(429, retry_after=45), 0), 45)


class TestCallRetries(unittest.TestCase):
    def setUp(self):
        self.provider = OpenAIProvider(model_name="gpt-4o-mini", api_key="sk-test")

    def _call(self, side_effects):
        with mock.patch("urllib.request.urlopen", side_effect=side_effects) as urlopen, \
             mock.patch("services.llm_client.time.sleep") as sleep:
            result = self.provider._call_local_llm([{"role": "user", "content": "x"}])
        return result, urlopen, sleep

    def test_recovers_after_transient_429s(self):
        result, urlopen, sleep = self._call([_http_error(429), _http_error(429), _FakeResponse(OK)])
        self.assertEqual(result, OK)
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_recovers_after_dns_failure(self):
        result, urlopen, _ = self._call([urllib.error.URLError("getaddrinfo failed"), _FakeResponse(OK)])
        self.assertEqual(result, OK)
        self.assertEqual(urlopen.call_count, 2)

    def test_400_is_raised_immediately_without_retry(self):
        with mock.patch("urllib.request.urlopen", side_effect=_http_error(400)) as urlopen, \
             mock.patch("services.llm_client.time.sleep") as sleep:
            with self.assertRaises(urllib.error.HTTPError):
                self.provider._call_local_llm([{"role": "user", "content": "x"}])
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_gives_up_after_max_retries(self):
        with mock.patch.dict(os.environ, {"TABULLM_LLM_MAX_RETRIES": "3"}), \
             mock.patch("urllib.request.urlopen", side_effect=_http_error(429)) as urlopen, \
             mock.patch("services.llm_client.time.sleep"):
            with self.assertRaises(urllib.error.HTTPError):
                self.provider._call_local_llm([{"role": "user", "content": "x"}])
        self.assertEqual(urlopen.call_count, 4)   # 1 attempt + 3 retries


class TestAllFoldsSucceeded(unittest.TestCase):
    """A unit with missing folds must not be checkpointed as complete."""

    def test_full_fold_set_passes(self):
        self.assertTrue(rbp._all_folds_succeeded({"results": [{}] * 10}, 10, "full", "d.csv"))

    def test_zero_folds_fails(self):
        self.assertFalse(rbp._all_folds_succeeded({"results": []}, 10, "full", "d.csv"))

    def test_missing_results_key_fails(self):
        self.assertFalse(rbp._all_folds_succeeded({}, 10, "full", "d.csv"))

    def test_nine_of_ten_fails(self):
        self.assertFalse(rbp._all_folds_succeeded({"results": [{}] * 9}, 10, "full", "d.csv"))


if __name__ == "__main__":
    unittest.main()
