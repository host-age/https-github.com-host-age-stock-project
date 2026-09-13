"""Tests for tools/kite_login.py's --server-url token push.

The daily login itself needs a real browser + Zerodha 2FA and cannot be
unit-tested; these cover the one piece that can be automated safely --
handing the resulting token to a running server instead of a manual curl.
"""
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, "tools")
import kite_login  # noqa: E402


def test_push_token_without_admin_secret_does_not_post():
    with patch("requests.post") as post:
        kite_login.push_token("https://example.onrender.com", "", "tok123")
    post.assert_not_called()


def test_push_token_posts_secret_and_token_to_set_token_endpoint():
    with patch("requests.post") as post:
        post.return_value = SimpleNamespace(ok=True, status_code=200, text="")
        kite_login.push_token("https://example.onrender.com/", "s3cret", "tok123")
    post.assert_called_once()
    args, kwargs = post.call_args
    assert args[0] == "https://example.onrender.com/set-token"
    assert kwargs["json"] == {"secret": "s3cret", "access_token": "tok123"}


def test_push_token_reports_non_ok_response(capsys):
    with patch("requests.post") as post:
        post.return_value = SimpleNamespace(ok=False, status_code=401,
                                            text="unauthorized")
        kite_login.push_token("https://example.onrender.com", "wrong", "tok123")
    err = capsys.readouterr().err
    assert "401" in err and "unauthorized" in err
