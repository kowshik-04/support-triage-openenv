---
title: Support Triage OpenEnv
sdk: docker
app_port: 8000
tags:
  - openenv
  - enterprise-support
  - policy-reasoning
  - customer-support
---

# Support Triage OpenEnv

`support-triage-openenv` is a real-world OpenEnv benchmark for customer support triage. It simulates a practical support workflow where an agent must inspect a ticket, look up policy, set structured workflow fields, write a compliant customer reply, and resolve the case while receiving dense reward and deterministic grading.

This project is designed for OpenEnv Round 1 style evaluation:

- real-world task simulation
- typed OpenEnv models
- `reset()`, `step()`, `state()`
- deterministic task graders
- dense reward shaping
- root-level `inference.py`
- Docker + HF Spaces readiness

## What The Environment Simulates

The environment models the kind of work a real support agent does:

- understand the customer issue
- research relevant internal policy
- classify the case correctly
- set the correct urgency
- route it to the right internal team
- add useful operational tags
- write a safe, policy-compliant reply
- resolve only after the workflow is complete

This is not a toy chatbot environment. The tasks mix billing, security, privacy, and SLA pressure so an agent has to behave like an operations worker, not just answer free-form text.

## Tasks

The environment contains exactly 5 deterministic tasks:

1. `duplicate_charge_easy`
   Duplicate charge inside the refund window. The agent should process the case as a valid billing refund.

2. `refund_outside_window_easy_plus`
   Refund request outside the allowed window. The agent should reject the refund politely and correctly.

3. `account_takeover_medium`
   Possible account takeover. The agent must treat it as an urgent security incident.

4. `sla_breach_escalation_medium_plus`
   Enterprise production issue close to SLA breach. The agent must escalate with urgency and ownership.

5. `gdpr_billing_mix_hard`
   GDPR deletion request mixed with a billing dispute. The agent must balance privacy, legal retention, and billing investigation language.

## Action Space

The action model is [`SupportTriageAction`](/Users/kowshikmente/Desktop/metaXhack/models.py).

Supported action types:

- `read_ticket`
- `search_policy`
- `set_classification`
- `set_priority`
- `route_ticket`
- `add_tag`
- `draft_reply`
- `resolve_ticket`

Structured fields:

- `query`
- `classification`
- `priority`
- `route`
- `tag`
- `reply_text`

Allowed structured values used by the benchmark agent:

- `query`: `refund_policy`, `account_security`, `enterprise_sla`, `privacy_deletion`, `billing_disputes`
- `classification`: `billing_refund`, `security_incident`, `privacy_request`
- `priority`: `low`, `normal`, `high`, `urgent`
- `route`: `billing_ops`, `trust_and_safety`, `privacy_legal`
- `tag`: `refund`, `duplicate_charge`, `account_takeover`, `identity_verification`, `privacy`, `billing_dispute`, `retention_review`

## Observation Space

The observation model is [`SupportTriageObservation`](/Users/kowshikmente/Desktop/metaXhack/models.py).

Each step returns:

- task id and difficulty
- ticket snapshot
- current classification, priority, route, and tags
- latest drafted reply
- last policy search results
- completed actions
- guidance string
- deterministic `grader_score`
- per-component `component_scores`
- violation list

The environment also exposes a structured [`SupportTriageState`](/Users/kowshikmente/Desktop/metaXhack/models.py) through `state()`.

## Reward And Grading

Reward is shaped across the full trajectory rather than only at episode end.

The grader is deterministic:

- same action sequence -> same score
- different action sequence -> different score

Scoring is based on workflow state:

- policy research
- classification correctness
- priority correctness
- route correctness
- tag coverage
- reply quality
- proper resolution

Penalties are applied for:

- repeated actions
- invalid or noisy actions
- unsafe or policy-breaking replies
- premature resolution
- wasting the step budget

## OpenEnv API

The environment implements the required OpenEnv interface:

- `reset()` -> initial observation
- `step(action)` -> observation, reward, done
- `state()` -> current structured environment state

HTTP routes exposed by [`server/app.py`](/Users/kowshikmente/Desktop/metaXhack/server/app.py):

- `GET /`
- `GET /health`
- `GET /metadata`
- `GET /schema`
- `POST /reset`
- `POST /step`
- `GET /state`

`/reset` and `/step` return top-level:

- `grader_score`
- `observation`
- `reward`
- `done`

## UI

The root page `/` serves a simple interactive console for:

- selecting one of the 5 tasks
- resetting the episode
- stepping through actions manually
- seeing live score updates
- viewing plain-English explanations
- inspecting component-by-component grading

The UI is useful for demos, debugging trajectories, and human review.

## Project Structure

- [`models.py`](/Users/kowshikmente/Desktop/metaXhack/models.py): OpenEnv typed models
- [`tasks.py`](/Users/kowshikmente/Desktop/metaXhack/tasks.py): ticket definitions and policy base
- [`client.py`](/Users/kowshikmente/Desktop/metaXhack/client.py): typed OpenEnv client
- [`openenv.yaml`](/Users/kowshikmente/Desktop/metaXhack/openenv.yaml): OpenEnv metadata
- [`server/support_triage_environment.py`](/Users/kowshikmente/Desktop/metaXhack/server/support_triage_environment.py): state machine, rewards, and grader
- [`server/app.py`](/Users/kowshikmente/Desktop/metaXhack/server/app.py): FastAPI + UI
- [`server/test_support_triage_env.py`](/Users/kowshikmente/Desktop/metaXhack/server/test_support_triage_env.py): local verification tests
- [`inference.py`](/Users/kowshikmente/Desktop/metaXhack/inference.py): LLM agent runner
- [`Dockerfile`](/Users/kowshikmente/Desktop/metaXhack/Dockerfile): root image for local / HF use
- [`server/Dockerfile`](/Users/kowshikmente/Desktop/metaXhack/server/Dockerfile): runtime image referenced by `openenv.yaml`

## Setup

### Local Python

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open:

- `http://127.0.0.1:8000/`

### Docker

```bash
docker build -t support-triage-openenv .
docker run --rm -p 8000:8000 support-triage-openenv
```

### Hugging Face Spaces

Use the root [`Dockerfile`](/Users/kowshikmente/Desktop/metaXhack/Dockerfile) in a Docker Space and tag the Space with `openenv`.

## Example API Usage

Reset a task:

```bash
curl -X POST http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"duplicate_charge_easy"}'
```

Take a step:

```bash
curl -X POST http://127.0.0.1:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"search_policy","query":"refund_policy"}'
```

Read state:

```bash
curl http://127.0.0.1:8000/state
```

## Inference Agent

The root-level [`inference.py`](/Users/kowshikmente/Desktop/metaXhack/inference.py) uses the OpenAI client for all LLM calls, as required.

It includes:

- plan generation
- step-by-step plan execution
- strict action validation
- task-aware correction
- anti-loop and anti-stall guards
- structured logs in `[START]`, `[STEP]`, and `[END]` format

Environment variables:

- `OPENAI_API_KEY`
- `API_BASE_URL`
- `MODEL_NAME`
- optional: `HF_TOKEN`
- optional: `OPENENV_PREFER_INPROCESS=true`
- optional: `ENV_MESSAGE_TIMEOUT_S=300`

### Reproducible Fallback Baseline

If the model call fails, the agent falls back to a deterministic controller. This provides a reproducible baseline for validation.

Verified locally with:

```bash
OPENAI_API_KEY=dummy \
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o-mini \
OPENENV_PREFER_INPROCESS=true \
.venv/bin/python inference.py
```

Observed reproducible fallback scores:

- `duplicate_charge_easy`: `0.9500`
- `refund_outside_window_easy_plus`: `1.0000`
- `account_takeover_medium`: `0.9500`
- `sla_breach_escalation_medium_plus`: `0.9360`
- `gdpr_billing_mix_hard`: `0.8830`
- average: `0.9438`

### Real LLM Example

OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=your_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
.venv/bin/python inference.py
```

Ollama example:

```bash
export API_BASE_URL=http://127.0.0.1:11434/v1
export MODEL_NAME=deepseek-r1:8b
export OPENAI_API_KEY=ollama
export OPENENV_PREFER_INPROCESS=true
.venv/bin/python inference.py
```

Real-model scores vary by model and run. During local development, Ollama-based runs with `deepseek-r1:8b` produced average scores in the strong but non-deterministic range, while the fallback controller stayed reproducible.

## Validation

Local checks used during development:

```bash
.venv/bin/openenv validate .
.venv/bin/python -m unittest server.test_support_triage_env
docker build -t support-triage-openenv .
```

Current local status:

- `openenv validate .` passed
- unit tests passed
- Docker build passed
- live API and UI were exercised locally
- fallback baseline in [`inference.py`](/Users/kowshikmente/Desktop/metaXhack/inference.py) completed successfully

## Notes

- The grader is deterministic and returns values in `[0.0, 1.0]`
- `reset()` creates a clean episode state
- `state()` exposes the structured internal environment state
- the hard task is intentionally multi-objective rather than single-label
- the benchmark is designed so the environment stays deterministic while the agent behavior can vary
