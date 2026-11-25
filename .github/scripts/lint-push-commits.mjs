import lint from "@commitlint/lint";
import load from "@commitlint/load";
import { format } from "@commitlint/format";

let input = "";
process.stdin.setEncoding("utf8");
for await (const chunk of process.stdin) input += chunk;
const commits = JSON.parse(input);
if (!Array.isArray(commits)) throw new Error("Expected a list of push commits");

const config = await load({}, { file: `${process.cwd()}/.commitlintrc.yaml` });
const options = {
  parserOpts: config.parserPreset?.parserOpts ?? {},
  plugins: config.plugins ?? {},
  ignores: config.ignores ?? [],
  defaultIgnores: config.defaultIgnores ?? true,
};

let invalid = false;
for (const commit of commits) {
  if (
    !/^[0-9a-f]{40}$/.test(commit.sha) ||
    typeof commit.message !== "string"
  ) {
    throw new Error("Invalid commit returned by GitHub");
  }
  if (commit.author === "dependabot[bot]") {
    console.log(`Skipping Dependabot commit ${commit.sha}`);
    continue;
  }

  const result = await lint(commit.message, config.rules, options);
  const output = format({ results: [result] }, { color: false });
  if (output) console.log(`Commit ${commit.sha}:\n${output}`);
  if (!result.valid) invalid = true;
}

if (invalid) process.exitCode = 1;
else console.log("All non-Dependabot commit messages passed.");
