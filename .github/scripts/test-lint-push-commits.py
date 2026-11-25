#!/usr/bin/env python3
"""Exercise push filtering with the real commitlint action image and a fake GitHub API."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/lint-push-commits.sh"
BOT = "a" * 40
HUMAN = "b" * 40


def commit(sha, author, message):
    return {"sha": sha, "author": {"login": author} if author else None, "commit": {"message": message}}


class PushCommitLintTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.event = self.path / "event.json"
        self.response = self.path / "response.json"
        gh = self.path / "gh"
        gh.write_text('#!/bin/sh\n[ "${FAIL_GH:-0}" != 1 ] || exit 1\ncat "$TEST_RESPONSE"\n')
        gh.chmod(0o755)
        self.env = os.environ.copy()
        self.env.pop("BASH_ENV", None)
        self.env.update(
            PATH=f"{self.path}{os.pathsep}{self.env['PATH']}",
            GITHUB_EVENT_PATH=str(self.event),
            GITHUB_REPOSITORY="sdd-org/kiddo",
            GITHUB_WORKSPACE=str(ROOT),
            RUNNER_TEMP=str(self.path),
            TEST_RESPONSE=str(self.response),
        )
        self.event.write_text(json.dumps({"before": "1" * 40, "after": "2" * 40}))

    def run_lint(self, commits, total=None):
        self.response.write_text(json.dumps([{"total_commits": len(commits) if total is None else total, "commits": commits}]))
        return subprocess.run(["bash", str(SCRIPT)], env=self.env, text=True, capture_output=True)

    def test_invalid_bot_message_is_exempt_but_valid_human_message_is_linted(self):
        bot = commit(BOT, "dependabot[bot]", "chore: update\n\n" + "x" * 120)
        human = commit(HUMAN, "sdd", "fix: correct the workflow")
        result = self.run_lint([bot, human])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"Skipping Dependabot commit {BOT}", result.stdout)

    def test_invalid_human_in_mixed_push_fails(self):
        result = self.run_lint([commit(BOT, "dependabot[bot]", "invalid"), commit(HUMAN, "sdd", "invalid")])
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(HUMAN, result.stdout)

    def test_unmapped_author_is_not_exempt(self):
        result = self.run_lint([commit(HUMAN, None, "invalid")])
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(HUMAN, result.stdout)

    def test_invalid_human_on_later_comparison_page_fails(self):
        self.response.write_text(json.dumps([
            {"total_commits": 2, "commits": [commit(BOT, "dependabot[bot]", "invalid")]},
            {"total_commits": 2, "commits": [commit(HUMAN, "sdd", "invalid")]},
        ]))
        result = subprocess.run(["bash", str(SCRIPT)], env=self.env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(HUMAN, result.stdout)

    def test_new_branch_looks_up_each_commit(self):
        self.event.write_text(json.dumps({"before": "0" * 40, "after": "2" * 40, "commits": [{"id": HUMAN}]}))
        self.response.write_text(json.dumps(commit(HUMAN, "sdd", "invalid")))
        result = subprocess.run(["bash", str(SCRIPT)], env=self.env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(HUMAN, result.stdout)

    def test_incomplete_comparison_fails_closed(self):
        self.assertNotEqual(self.run_lint([commit(HUMAN, "sdd", "fix: valid")], total=2).returncode, 0)

    def test_api_failure_fails_closed(self):
        self.env["FAIL_GH"] = "1"
        self.assertNotEqual(self.run_lint([]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
