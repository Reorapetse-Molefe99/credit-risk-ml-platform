# Agents guide: using and creating Claude Code agents for this project

Claude Code **subagents** are specialised assistants, each with its own system prompt, tool
permissions, model and separate context window. The main Claude session hands them focused jobs
(e.g. "validate model v3") and gets back a short report, so long data dumps and training logs stay out
of your main conversation.

This project ships with five agents in `.claude/agents/`. They load automatically whenever you run
Claude Code from inside this repository (they are committed, so anyone who clones it gets them).

---

## 1. The agents included

| Agent | Role in the lifecycle | Can edit code? | Key guardrail |
|---|---|---|---|
| `credit-data-analyst` | EDA, data quality, leakage and fairness risks | No | Uses the train split only |
| `feature-engineer` | Adds features to `features.py`, writes tests, A/B tests on a sample | Yes | Row-wise features, no protected attributes, accepts features on val metrics |
| `model-trainer` | Runs training and `--tune`, summarises MLflow runs, registers candidates | No | Selects on `val_*`, `--register` only for full-data runs |
| `model-validator` | Independent validation: discrimination, calibration, PSI, segments, fairness → `reports/` | Reports only | Never promotes |
| `mlops-deployer` | Registry aliases, promote/rollback/export, API, dashboard, Docker, batch scoring, drift checks | No | Asks you before moving `@champion` |

The split mirrors how credit-risk teams separate model development, independent validation and
deployment. The agent that builds a model doesn't approve it.

```
credit-data-analyst ─► feature-engineer ─► model-trainer ─► model-validator ─► mlops-deployer
   (findings)          (features + tests)   (runs, vN)       (report, verdict)   (@champion, serve)
```

---

## 2. Using them

Start the tracking server first
(`uv run mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --port 5000`),
then run `claude` in the project folder,
or open the folder in VS Code with the Claude extension.

**Ask by name**:
```
Use the credit-data-analyst agent to check whether missing EXT_SOURCE values predict default.
Have the model-trainer agent tune xgboost for 40 trials on a 60k sample, then confirm the best on full data and register it.
Ask the model-validator to validate the latest registered version against the champion.
Use the mlops-deployer to promote it if validation passed, then start the REST API and smoke-test it.
```

**Or let Claude choose.** Each agent's `description` says when it applies, so requests like
"is v3 safe to deploy?" get routed to `model-validator` automatically.

**Chain them in one request**:
```
Run the full cycle: analyst looks for 3 new feature ideas around income and employment,
feature-engineer implements the best ones, model-trainer trains and registers a full-data LightGBM,
model-validator validates it. Stop before promoting and show me the validation report.
```

**Run independent work in parallel**:
```
In parallel: have model-trainer tune lightgbm and, separately, tune xgboost (30 trials each, 60k sample).
```

Type `/agents` in Claude Code to list, view, edit or create agents interactively.

---

## 3. Creating your own agent

### Option A: interactive
Run `/agents` → **Create new agent** → choose *Project* (saved in `.claude/agents/`, shared with the
repo) or *Personal* (`~/.claude/agents/`, available in every project). Describe the agent. Claude drafts
the prompt, then you choose its tools and model.

### Option B: write the file
Create `.claude/agents/<name>.md`:

```markdown
---
name: scorecard-builder
description: Builds a traditional WoE / logistic-regression scorecard (points-based) from the credit data and logs it to MLflow. Use when the user asks for a scorecard, WoE/IV binning, or an interpretable regulatory model.
tools: Read, Edit, Write, Grep, Glob, Bash
model: sonnet
---

You build interpretable credit scorecards.

Steps:
1. Bin each candidate variable on the train split (monotonic bins, ≥5% population per bin).
2. Compute WoE and IV; drop variables with IV < 0.02 or suspicious IV > 0.5 (possible leakage).
3. Fit logistic regression on WoE values; check signs are consistent with WoE.
4. Scale to points: PDO = 20, base score 600 at odds 50:1.
5. Log to MLflow with `mlflow.start_run(run_name="scorecard")`: IV table, bins, coefficients,
   points table, and val/test metrics via `credit_risk.evaluate.classification_metrics`.

Report: selected variables with IV, val/test Gini and KS, and the run ID.
```

### Frontmatter fields

| Field | Required | Notes |
|---|---|---|
| `name` | yes | lowercase-with-hyphens; how you refer to the agent |
| `description` | yes | **Most important field.** Claude reads it to decide when to delegate. Say *what* it does and *when* to use it. "Use proactively when..." encourages automatic use. |
| `tools` | no | Comma-separated allowlist (`Read, Grep, Glob, Bash, Edit, Write, WebSearch`, MCP tools...). Leave it out to inherit all tools. Read-only agents are safer. |
| `model` | no | `sonnet`, `opus`, `haiku`, or `inherit` (same as the main session). Use `inherit`/`opus` for judgement-heavy work like validation, `sonnet` for routine runs. |

The text below the frontmatter is the agent's **system prompt**. The agent doesn't see your main
conversation, so the prompt (plus `CLAUDE.md`, which every agent reads) must contain everything it needs.

### Writing a good agent prompt
1. **One job.** "Validate a model" is a good scope. "Do ML" is too broad.
2. **Give the exact commands.** Agents work best when they're told the CLI to run (`uv run python -m credit_risk.train ...`).
3. **State the guardrails explicitly.** Examples: never touch the test split for tuning, never promote, don't edit `src/`.
   Back up the important ones by withholding tools: an agent without `Edit`/`Write` can't change code.
4. **Define the report format.** The main session only sees the agent's final message, so ask for
   run IDs, version numbers, a metrics table and a recommendation.
5. **Tie it to MLflow.** Have every experiment go through MLflow so results are auditable later, even
   if the agent's chat is gone.

### More agent ideas for this project
- `monitoring-analyst`: runs `credit_risk.monitoring` on each new monthly cohort, explains the feature-level PSI
  drivers, and raises retraining alerts. The mlops-deployer covers this at a basic level today.
- `ifrs9-analyst`: turns PDs into 12-month/lifetime ECL with LGD/EAD assumptions and stage allocation.
- `explainability-analyst`: SHAP global and local explanations and adverse-action reason codes for
  declined applicants (logged as MLflow artifacts).
- `documentation-writer`: generates model documentation (purpose, data, methodology, performance,
  limitations) from MLflow runs and validation reports.

---

## 4. Tips
- Agents can't start interactive or long-running processes for you unless asked. Keep
  the MLflow server running in your own terminal.
- To change an agent, edit its `.md` file. The change applies to new Claude Code sessions, or right away via `/agents`.
- Store project-specific facts and rules in `CLAUDE.md` rather than repeating them in every agent.
- If an agent keeps being picked for the wrong tasks, tighten its `description`.
