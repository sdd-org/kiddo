#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_EVENT_PATH:?Missing push event payload}"
: "${GITHUB_REPOSITORY:?Missing repository}"
: "${GITHUB_WORKSPACE:?Missing workspace}"

before=$(jq -er '.before | select(type == "string" and test("^[0-9a-f]{40}$"))' "$GITHUB_EVENT_PATH")
after=$(jq -er '.after | select(type == "string" and test("^[0-9a-f]{40}$"))' "$GITHUB_EVENT_PATH")
tmp_dir=$(mktemp -d "${RUNNER_TEMP:-/tmp}/push-commitlint.XXXXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

if [[ "$before" == 0000000000000000000000000000000000000000 ]]; then
  # A new branch has no base for the compare API. The push payload contains at
  # most 2048 commits; fail closed if it may have been truncated.
  jq -e '.commits | type == "array" and length < 2048 and all(.[]; .id | type == "string" and test("^[0-9a-f]{40}$"))' "$GITHUB_EVENT_PATH" >/dev/null
  jq -r '.commits[].id' "$GITHUB_EVENT_PATH" > "$tmp_dir/ids"
  : > "$tmp_dir/commits.jsonl"
  while IFS= read -r sha; do
    gh api "repos/$GITHUB_REPOSITORY/commits/$sha" >> "$tmp_dir/commits.jsonl"
    printf '\n' >> "$tmp_dir/commits.jsonl"
  done < "$tmp_dir/ids"
  jq -s '[.[] | {sha, author: .author.login, message: .commit.message}]' "$tmp_dir/commits.jsonl" > "$tmp_dir/commits.json"
else
  # Unlike the action's default single page, --paginate also covers large pushes.
  gh api --paginate --slurp "repos/$GITHUB_REPOSITORY/compare/$before...$after?per_page=100" > "$tmp_dir/pages.json"
  jq -e '.[0].total_commits as $total | ($total | type == "number") and ([.[].commits[]] | length == $total)' "$tmp_dir/pages.json" >/dev/null
  jq '[.[] | .commits[] | {sha, author: .author.login, message: .commit.message}]' "$tmp_dir/pages.json" > "$tmp_dir/commits.json"
fi

# Use the same image and commitlint packages as wagoid/commitlint-github-action@v6.
docker run --rm --interactive \
  --volume "$GITHUB_WORKSPACE:/workspace:ro" \
  --workdir /workspace \
  --entrypoint node \
  wagoid/commitlint-github-action:6.2.1 \
  /workspace/.github/scripts/lint-push-commits.mjs < "$tmp_dir/commits.json"
