from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
try:
    from ..models import SupportTriageAction, SupportTriageObservation
    from ..models import SupportTriageState
    from .support_triage_environment import SupportTriageEnvironment
except ImportError:
    from models import SupportTriageAction, SupportTriageObservation
    from models import SupportTriageState
    from server.support_triage_environment import SupportTriageEnvironment

from openenv.core.env_server import create_app


app = create_app(
    SupportTriageEnvironment,
    SupportTriageAction,
    SupportTriageObservation,
    env_name="support-triage-openenv",
)
http_env = SupportTriageEnvironment()


def _remove_generated_route(path: str, method: str) -> None:
    app.router.routes = [
        route
        for route in app.router.routes
        if not (
            getattr(route, "path", None) == path
            and method.upper() in getattr(route, "methods", set())
        )
    ]


for route_path, route_method in [("/reset", "POST"), ("/step", "POST"), ("/state", "GET")]:
    _remove_generated_route(route_path, route_method)


@app.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Support Triage OpenEnv</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --surface: #ffffff;
      --surface-soft: #f9fbfc;
      --text: #15232e;
      --muted: #677887;
      --line: #dbe4ea;
      --accent: #0f6b72;
      --accent-soft: #e1f3f4;
      --good: #18794e;
      --warn: #b7791f;
      --bad: #c2410c;
      --shadow: 0 18px 46px rgba(21, 35, 46, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top right, rgba(15, 107, 114, 0.08), transparent 24%), var(--bg);
      line-height: 1.55;
    }
    .shell {
      max-width: 1340px;
      margin: 0 auto;
      padding: 24px 20px 44px;
    }
    .top {
      display: grid;
      grid-template-columns: 1.5fr 0.95fr;
      gap: 16px;
      margin-bottom: 16px;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
    }
    .overview {
      padding: 24px;
    }
    .eyebrow {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      color: var(--accent);
      margin-bottom: 8px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(30px, 4.2vw, 52px);
      line-height: 1.05;
      letter-spacing: -0.03em;
    }
    .subtitle {
      font-size: 17px;
      color: var(--muted);
      max-width: 54rem;
    }
    .overview-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-top: 20px;
    }
    .mini {
      background: var(--surface-soft);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }
    .mini span {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .mini strong {
      font-size: 24px;
      letter-spacing: -0.02em;
    }
    .scorebox {
      padding: 22px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 18px;
    }
    .score-ring {
      width: 138px;
      height: 138px;
      border-radius: 50%;
      background: conic-gradient(var(--accent) 0deg, #dfe9ee 0deg);
      display: grid;
      place-items: center;
      margin: 0 auto;
      position: relative;
    }
    .score-ring::after {
      content: "";
      position: absolute;
      inset: 12px;
      background: var(--surface);
      border-radius: 50%;
      border: 1px solid var(--line);
    }
    .score-ring span {
      position: relative;
      z-index: 1;
      font-size: 30px;
      font-weight: 800;
    }
    .score-copy { text-align: center; }
    .score-copy small {
      color: var(--muted);
      display: block;
      margin-bottom: 6px;
    }
    .score-copy strong {
      display: block;
      font-size: 24px;
      margin-bottom: 6px;
    }
    .score-copy p {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }
    .layout {
      display: grid;
      grid-template-columns: 0.95fr 1.05fr;
      gap: 16px;
    }
    .panel { padding: 22px; }
    h2 {
      margin: 0 0 14px;
      font-size: 24px;
      letter-spacing: -0.02em;
    }
    .task-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 14px;
    }
    .task-btn, button, select, textarea, input { font: inherit; }
    .task-btn {
      text-align: left;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--surface);
      cursor: pointer;
      transition: border-color 0.15s ease, transform 0.15s ease;
    }
    .task-btn:hover { transform: translateY(-1px); border-color: var(--accent); }
    .task-btn.active { border-color: var(--accent); background: var(--accent-soft); }
    .task-btn strong { display: block; font-size: 16px; margin-bottom: 6px; }
    .task-btn small { display: block; color: var(--muted); font-size: 13px; }
    .case-box {
      padding: 18px;
      background: #fffdf7;
      border: 1px solid #efe2bf;
      border-radius: 18px;
      margin-bottom: 16px;
    }
    .case-box strong {
      display: block;
      font-size: 18px;
      margin-bottom: 10px;
    }
    .case-meta {
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .meta-pill {
      padding: 8px 10px;
      border-radius: 999px;
      background: white;
      border: 1px solid var(--line);
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
    }
    .controls {
      display: grid;
      gap: 12px;
      margin-bottom: 14px;
    }
    .row {
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 12px;
      align-items: center;
    }
    label {
      font-size: 14px;
      color: var(--muted);
      font-weight: 700;
    }
    select, textarea, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 13px 14px;
      background: var(--surface);
      color: var(--text);
      font-size: 15px;
    }
    select:focus, textarea:focus, input:focus {
      outline: 2px solid rgba(15, 107, 114, 0.18);
      border-color: var(--accent);
    }
    textarea { min-height: 132px; resize: vertical; }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    button {
      padding: 12px 18px;
      border-radius: 999px;
      border: 0;
      cursor: pointer;
      background: var(--accent);
      color: white;
      font-weight: 700;
      box-shadow: 0 10px 20px rgba(15, 107, 114, 0.18);
    }
    button.secondary { background: var(--good); }
    button.ghost {
      background: transparent;
      color: var(--text);
      border: 1px solid var(--line);
      box-shadow: none;
    }
    .helper-box {
      margin-top: 14px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--surface-soft);
    }
    .helper-box strong { display: block; margin-bottom: 8px; }
    .helper-list {
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    .right-stack {
      display: grid;
      gap: 12px;
    }
    .result-banner {
      padding: 18px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: var(--surface-soft);
    }
    .result-banner h3 {
      margin: 0 0 8px;
      font-size: 26px;
      letter-spacing: -0.02em;
    }
    .result-banner p {
      margin: 0;
      color: var(--muted);
      font-size: 16px;
    }
    .result-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .detail {
      padding: 16px;
      border-radius: 16px;
      background: var(--surface-soft);
      border: 1px solid var(--line);
    }
    .detail strong { display: block; margin-bottom: 8px; font-size: 15px; }
    .subtle { color: var(--muted); line-height: 1.6; }
    .progress {
      height: 12px;
      background: #dfe9ee;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 8px;
    }
    .progress > div {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #25a0a3);
    }
    .breakdown-item {
      padding: 12px 0;
      border-bottom: 1px solid var(--line);
    }
    .breakdown-item:last-child {
      border-bottom: 0;
      padding-bottom: 0;
    }
    .break-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
    }
    .status {
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .timeline {
      display: grid;
      gap: 10px;
    }
    .timeline-entry {
      padding: 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--surface);
    }
    .timeline-entry strong {
      display: block;
      margin-bottom: 6px;
      font-size: 15px;
    }
    .timeline-entry .why {
      color: var(--text);
      margin-bottom: 4px;
    }
    .timeline-entry .what {
      color: var(--muted);
      font-size: 14px;
    }
    .bar {
      height: 10px;
      border-radius: 999px;
      background: #dfe9ee;
      overflow: hidden;
      margin-top: 8px;
    }
    .bar > div {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #25a0a3);
    }
    .good { color: var(--good); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    @media (max-width: 980px) {
      .top, .layout, .overview-grid, .task-grid, .row, .result-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="top">
      <div class="card overview">
        <div class="eyebrow">OpenEnv Environment Console</div>
        <h1>Customer Support Triage</h1>
        <div class="subtitle">
          This environment simulates a real support case. The goal is to understand the customer problem, take the right actions,
          and solve the case correctly while the system shows how well the work is going.
        </div>
        <div class="overview-grid">
          <div class="mini"><span>Task</span><strong id="overview-task">Duplicate Charge</strong></div>
          <div class="mini"><span>Difficulty</span><strong id="overview-difficulty">Easy</strong></div>
          <div class="mini"><span>Tasks</span><strong>5</strong></div>
          <div class="mini"><span>Current Score</span><strong id="overview-score">0.00</strong></div>
        </div>
      </div>
      <div class="card scorebox">
        <div class="score-ring" id="score-ring"><span id="live-score">0.00</span></div>
        <div class="score-copy">
          <small>Live performance</small>
          <strong id="score-title">No meaningful progress yet</strong>
          <p id="score-summary">Reset the task and start working through the case step by step.</p>
        </div>
      </div>
    </section>

    <section class="layout">
      <div class="card panel">
        <h2>Case And Actions</h2>
        <div class="task-grid">
          <button class="task-btn active" data-task="duplicate_charge_easy"><strong>Duplicate Charge</strong><small>Easy: refund a duplicate billing event</small></button>
          <button class="task-btn" data-task="refund_outside_window_easy_plus"><strong>Refund Outside Window</strong><small>Easy+: decline politely outside the policy window</small></button>
          <button class="task-btn" data-task="account_takeover_medium"><strong>Account Takeover</strong><small>Medium: urgent security escalation</small></button>
          <button class="task-btn" data-task="sla_breach_escalation_medium_plus"><strong>SLA Breach Escalation</strong><small>Medium+: protect an enterprise SLA</small></button>
          <button class="task-btn" data-task="gdpr_billing_mix_hard"><strong>GDPR + Billing Mix</strong><small>Hard: privacy, billing, and legal tradeoffs</small></button>
        </div>
        <div class="case-box" id="case-display"></div>
        <div class="controls">
          <div class="row">
            <label for="action-type">Action</label>
            <select id="action-type">
              <option value="read_ticket">read_ticket</option>
              <option value="search_policy">search_policy</option>
              <option value="set_classification">set_classification</option>
              <option value="set_priority">set_priority</option>
              <option value="route_ticket">route_ticket</option>
              <option value="add_tag">add_tag</option>
              <option value="draft_reply">draft_reply</option>
              <option value="resolve_ticket">resolve_ticket</option>
            </select>
          </div>
          <div class="row">
            <label for="action-value">Value</label>
            <select id="action-value"></select>
          </div>
          <div class="row">
            <label for="reply-box">Reply Text</label>
            <textarea id="reply-box" placeholder="Use this only when the action is draft_reply. Write the message that would be sent to the customer."></textarea>
          </div>
        </div>
        <div class="actions">
          <button id="step-btn">Run Step</button>
          <button id="reset-btn" class="secondary">Reset Task</button>
          <button id="autoplay-btn" class="ghost">Run Reference Policy</button>
        </div>
        <div class="helper-box">
          <strong>Helpful Guidance</strong>
          <div class="helper-list" id="helper-box">
            <div>Start by reading the customer case carefully.</div>
            <div>Look up the relevant policy before making a decision.</div>
          </div>
        </div>
      </div>

      <div class="card panel">
        <h2>Live Explanation And Results</h2>
        <div class="right-stack">
          <div class="result-banner" id="result-banner"></div>
          <div class="result-grid">
            <div class="detail" id="step-explanation"></div>
            <div class="detail" id="final-summary"></div>
          </div>
          <div class="detail">
            <strong>Evaluation Breakdown</strong>
            <div id="components-panel"></div>
          </div>
          <div class="detail">
            <strong>Step-by-Step Activity</strong>
            <div id="stream-panel" class="timeline"></div>
          </div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const taskButtons = [...document.querySelectorAll('.task-btn')];
    const actionType = document.getElementById('action-type');
    const actionValue = document.getElementById('action-value');
    const replyBox = document.getElementById('reply-box');
    const caseDisplay = document.getElementById('case-display');
    const helperBox = document.getElementById('helper-box');
    const resultBanner = document.getElementById('result-banner');
    const stepExplanation = document.getElementById('step-explanation');
    const finalSummary = document.getElementById('final-summary');
    const componentsPanel = document.getElementById('components-panel');
    const streamPanel = document.getElementById('stream-panel');
    const scoreNode = document.getElementById('live-score');
    const scoreRing = document.getElementById('score-ring');
    const scoreTitle = document.getElementById('score-title');
    const scoreSummary = document.getElementById('score-summary');
    const overviewTask = document.getElementById('overview-task');
    const overviewDifficulty = document.getElementById('overview-difficulty');
    const overviewScore = document.getElementById('overview-score');

    let currentTask = 'duplicate_charge_easy';
    let latestObservation = null;
    const actionValueOptions = {
      read_ticket: [],
      search_policy: ['refund_policy', 'account_security', 'enterprise_sla', 'privacy_deletion', 'billing_disputes'],
      set_classification: ['billing_refund', 'security_incident', 'privacy_request'],
      set_priority: ['low', 'normal', 'high', 'urgent'],
      route_ticket: ['billing_ops', 'trust_and_safety', 'privacy_legal'],
      add_tag: ['refund', 'duplicate_charge', 'account_takeover', 'identity_verification', 'privacy', 'billing_dispute', 'retention_review'],
      draft_reply: [],
      resolve_ticket: []
    };

    const referencePlans = {
      duplicate_charge_easy: [
        { action_type: 'search_policy', query: 'refund_policy' },
        { action_type: 'set_classification', classification: 'billing_refund' },
        { action_type: 'set_priority', priority: 'normal' },
        { action_type: 'route_ticket', route: 'billing_ops' },
        { action_type: 'add_tag', tag: 'refund' },
        { action_type: 'add_tag', tag: 'duplicate_charge' },
        { action_type: 'draft_reply', reply_text: 'Hi Jamie, I reviewed order BIL-1042 and billing can process the refund within the 30-day window. We are routing this to billing to reverse the duplicate charge.' },
        { action_type: 'resolve_ticket' }
      ],
      refund_outside_window_easy_plus: [
        { action_type: 'search_policy', query: 'refund_policy' },
        { action_type: 'set_classification', classification: 'billing_refund' },
        { action_type: 'set_priority', priority: 'normal' },
        { action_type: 'route_ticket', route: 'billing_ops' },
        { action_type: 'add_tag', tag: 'refund' },
        { action_type: 'draft_reply', reply_text: 'Hi Nina, I reviewed invoice INV-7781 and this request is outside the 30-day refund window at 47 days. We cannot approve a refund under the current billing policy, but billing can answer any follow-up questions.' },
        { action_type: 'resolve_ticket' }
      ],
      account_takeover_medium: [
        { action_type: 'search_policy', query: 'account_security' },
        { action_type: 'search_policy', query: 'enterprise_sla' },
        { action_type: 'set_classification', classification: 'security_incident' },
        { action_type: 'set_priority', priority: 'urgent' },
        { action_type: 'route_ticket', route: 'trust_and_safety' },
        { action_type: 'add_tag', tag: 'account_takeover' },
        { action_type: 'add_tag', tag: 'identity_verification' },
        { action_type: 'draft_reply', reply_text: 'Hi Avery, this is being treated as an urgent trust case. Please secure your email, rotate any reused password, and verify your identity so trust and safety can recover the account safely.' },
        { action_type: 'resolve_ticket' }
      ],
      sla_breach_escalation_medium_plus: [
        { action_type: 'search_policy', query: 'enterprise_sla' },
        { action_type: 'set_classification', classification: 'security_incident' },
        { action_type: 'set_priority', priority: 'urgent' },
        { action_type: 'route_ticket', route: 'trust_and_safety' },
        { action_type: 'add_tag', tag: 'identity_verification' },
        { action_type: 'draft_reply', reply_text: 'Hi Priya, we are escalating this enterprise case urgently because the SLA is at risk. An owner is now assigned and we will provide a meaningful update as quickly as possible.' },
        { action_type: 'resolve_ticket' }
      ],
      gdpr_billing_mix_hard: [
        { action_type: 'search_policy', query: 'privacy_deletion' },
        { action_type: 'search_policy', query: 'billing_disputes' },
        { action_type: 'search_policy', query: 'enterprise_sla' },
        { action_type: 'set_classification', classification: 'privacy_request' },
        { action_type: 'set_priority', priority: 'high' },
        { action_type: 'route_ticket', route: 'privacy_legal' },
        { action_type: 'add_tag', tag: 'privacy' },
        { action_type: 'add_tag', tag: 'billing_dispute' },
        { action_type: 'add_tag', tag: 'retention_review' },
        { action_type: 'draft_reply', reply_text: 'Hi Morgan, we will process your request to delete account data and send it to privacy legal for review. We also opened a billing investigation for the March charges. Some records may be retained for legal obligations while we investigate.' },
        { action_type: 'resolve_ticket' }
      ]
    };

    function setActiveTask(taskId) {
      currentTask = taskId;
      taskButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.task === taskId));
    }

    function taskTitleFromId(taskId) {
      const titles = {
        duplicate_charge_easy: 'Duplicate Charge',
        refund_outside_window_easy_plus: 'Refund Outside Window',
        account_takeover_medium: 'Account Takeover',
        sla_breach_escalation_medium_plus: 'SLA Breach Escalation',
        gdpr_billing_mix_hard: 'GDPR + Billing Mix'
      };
      return titles[taskId] || taskId;
    }

    function refreshActionValueOptions() {
      const kind = actionType.value;
      const options = actionValueOptions[kind] || [];
      actionValue.innerHTML = '';
      if (!options.length) {
        actionValue.disabled = true;
        actionValue.innerHTML = '<option value="">No extra value needed</option>';
      } else {
        actionValue.disabled = false;
        actionValue.innerHTML = options.map(value => `<option value="${value}">${value}</option>`).join('');
      }
      replyBox.disabled = kind !== 'draft_reply';
      if (kind !== 'draft_reply') replyBox.value = '';
    }

    function scoreMeaning(score) {
      if (score >= 0.9) return 'Excellent handling';
      if (score >= 0.75) return 'Good handling';
      if (score >= 0.5) return 'Partial progress';
      if (score > 0) return 'Early progress';
      return 'No meaningful progress yet';
    }

    function nextStepAdvice(obs) {
      const parts = obs.component_scores || {};
      if ((parts.research || 0) < 0.5) return 'Look up the relevant policy first.';
      if ((parts.classification || 0) < 1) return 'Set the correct case type.';
      if ((parts.priority || 0) < 1) return 'Set the right urgency.';
      if ((parts.route || 0) < 1) return 'Send the case to the correct team.';
      if ((parts.tags || 0) < 1) return 'Add the right labels so the case is easier to process.';
      if ((parts.reply || 0) < 1) return 'Write a better reply to the customer.';
      if ((parts.resolved || 0) < 1) return 'Close the case once everything important is done.';
      return 'This case is complete. Reset and try another task.';
    }

    function userSummary(obs) {
      const score = Number(obs.grader_score || 0);
      const resolved = obs.done ? 'The case is closed.' : 'The case is still open.';
      return `${scoreMeaning(score)}. ${resolved} ${nextStepAdvice(obs)}`;
    }

    function userFriendlyMetricName(name) {
      const map = {
        classification: 'Understood the case',
        priority: 'Set correct priority',
        route: 'Routed to correct team',
        research: 'Looked up correct policy',
        tags: 'Added correct labels',
        reply: 'Wrote appropriate reply',
        resolved: 'Closed case properly'
      };
      return map[name] || name;
    }

    function simpleStatus(value) {
      if (value >= 0.99) return '<span class="status good">Done</span>';
      if (value > 0) return '<span class="status warn">In progress</span>';
      return '<span class="status bad">Not done</span>';
    }

    function actionReason(payload) {
      const type = payload.action_type;
      if (type === 'read_ticket') return 'This helps understand the customer problem before doing anything else.';
      if (type === 'search_policy') return 'This checks the rules before making a decision.';
      if (type === 'set_classification') return 'This labels the problem correctly.';
      if (type === 'set_priority') return 'This decides how urgent the case is.';
      if (type === 'route_ticket') return 'This sends the case to the right internal team.';
      if (type === 'add_tag') return 'This adds useful labels for tracking and routing.';
      if (type === 'draft_reply') return 'This prepares the message that would go to the customer.';
      if (type === 'resolve_ticket') return 'This closes the case after the important work is complete.';
      return 'This updates the case.';
    }

    function actionLabel(payload) {
      const type = payload.action_type;
      if (type === 'search_policy') return 'Looked up policy';
      if (type === 'set_classification') return 'Set case type';
      if (type === 'set_priority') return 'Set urgency';
      if (type === 'route_ticket') return 'Routed case';
      if (type === 'add_tag') return 'Added label';
      if (type === 'draft_reply') return 'Wrote reply';
      if (type === 'resolve_ticket') return 'Closed case';
      return 'Read ticket';
    }

    function actionValueText(payload) {
      return payload.query || payload.classification || payload.priority || payload.route || payload.tag || payload.reply_text || '';
    }

    function timelineHtml(action, why, result) {
      return `
        <div class="timeline-entry">
          <strong>${action}</strong>
          <div class="why">Why: ${why}</div>
          <div class="what">Result: ${result}</div>
        </div>
      `;
    }

    function renderObservation(obs) {
      latestObservation = obs;
      const score = Number(obs.grader_score || 0);
      const difficulty = String(obs.difficulty || '').replace(/^./, c => c.toUpperCase());
      const ringDegrees = Math.max(0, Math.min(360, score * 360));

      scoreNode.textContent = score.toFixed(2);
      overviewScore.textContent = score.toFixed(2);
      overviewTask.textContent = taskTitleFromId(obs.task_id);
      overviewDifficulty.textContent = difficulty;
      scoreRing.style.background = `conic-gradient(var(--accent) ${ringDegrees}deg, #dfe9ee ${ringDegrees}deg)`;
      scoreTitle.textContent = scoreMeaning(score);
      scoreSummary.textContent = userSummary(obs);

      caseDisplay.innerHTML = `
        <strong>${obs.ticket.subject}</strong>
        <div>${obs.ticket.body}</div>
        <div class="case-meta">
          <div class="meta-pill">Case ID: ${obs.ticket.ticket_id}</div>
          <div class="meta-pill">Customer: ${obs.ticket.customer_name}</div>
          <div class="meta-pill">Region: ${obs.ticket.region}</div>
          <div class="meta-pill">Tier: ${obs.ticket.account_tier}</div>
          <div class="meta-pill">SLA: ${obs.ticket.sla_hours_remaining}h</div>
        </div>
      `;

      helperBox.innerHTML = `
        <div>Next best move: ${nextStepAdvice(obs)}</div>
        <div>System guidance: ${obs.guidance}</div>
      `;

      resultBanner.innerHTML = `
        <h3>${scoreMeaning(score)}</h3>
        <p>${userSummary(obs)}</p>
      `;

      stepExplanation.innerHTML = `
        <strong>What happened and why</strong>
        <div class="subtle">${obs.guidance}</div>
        <div class="subtle" style="margin-top:10px;">Live score: ${score.toFixed(2)} / 1.00</div>
        <div class="progress"><div style="width:${Math.max(0, Math.min(100, score * 100))}%"></div></div>
      `;

      finalSummary.innerHTML = `
        <strong>${obs.done ? 'Final Result' : 'Current Result'}</strong>
        <div class="subtle">Case status: ${obs.done ? 'Closed' : 'Still in progress'}</div>
        <div class="subtle">Classification: ${obs.current_classification || 'not set yet'}</div>
        <div class="subtle">Priority: ${obs.current_priority || 'not set yet'}</div>
        <div class="subtle">Route: ${obs.current_route || 'not set yet'}</div>
        <div class="subtle">Reply written: ${obs.latest_reply ? 'Yes' : 'No'}</div>
        <div class="subtle">Mistakes found: ${(obs.violations || []).join(', ') || 'none so far'}</div>
      `;

      componentsPanel.innerHTML = `
        ${Object.entries(obs.component_scores || {}).map(([name, value]) => `
          <div class="breakdown-item">
            <div class="break-row">
              <div>${userFriendlyMetricName(name)}</div>
              <div>${simpleStatus(Number(value))}</div>
            </div>
            <div class="bar"><div style="width:${Math.max(0, Math.min(100, Number(value) * 100))}%"></div></div>
          </div>
        `).join('')}
      `;
    }

    async function resetTask() {
      const response = await fetch('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: currentTask })
      });
      const payload = await response.json();
      renderObservation(payload.observation);
      streamPanel.innerHTML = timelineHtml('Task reset', 'A new case is ready to work on.', 'You can now take the first action.');
    }

    function buildActionPayload() {
      const kind = actionType.value;
      const payload = { action_type: kind };
      const value = actionValue.disabled ? '' : actionValue.value;
      const reply = replyBox.value.trim();
      if (kind === 'search_policy' && value) payload.query = value;
      if (kind === 'set_classification' && value) payload.classification = value;
      if (kind === 'set_priority' && value) payload.priority = value;
      if (kind === 'route_ticket' && value) payload.route = value;
      if (kind === 'add_tag' && value) payload.tag = value;
      if (kind === 'draft_reply' && reply) payload.reply_text = reply;
      return payload;
    }

    async function sendStep(payload) {
      const response = await fetch('/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) {
        streamPanel.innerHTML = timelineHtml(
          'Action not accepted',
          'This action is not appropriate at this stage.',
          'Please reset the task or choose a different action.'
        ) + streamPanel.innerHTML;
        return;
      }
      renderObservation(data.observation);
      streamPanel.innerHTML = timelineHtml(
        `${actionLabel(payload)}${actionValueText(payload) ? ' -> ' + actionValueText(payload) : ''}`,
        actionReason(payload),
        data.observation.guidance
      ) + streamPanel.innerHTML;
    }

    async function autoplay() {
      streamPanel.innerHTML = timelineHtml(
        'Reference policy started',
        'The system is showing a strong example trajectory.',
        'Watch how the score and explanation change after each step.'
      ) + streamPanel.innerHTML;
      for (const action of referencePlans[currentTask]) {
        await sendStep(action);
        if (latestObservation?.done) break;
      }
    }

    taskButtons.forEach(btn => btn.addEventListener('click', async () => {
      setActiveTask(btn.dataset.task);
      await resetTask();
    }));

    actionType.addEventListener('change', refreshActionValueOptions);
    document.getElementById('reset-btn').addEventListener('click', resetTask);
    document.getElementById('step-btn').addEventListener('click', async () => sendStep(buildActionPayload()));
    document.getElementById('autoplay-btn').addEventListener('click', autoplay);

    refreshActionValueOptions();
    resetTask();
  </script>
</body>
</html>
        """
    )


@app.get("/ping")
async def ping() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
async def http_reset(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        observation = http_env.reset(task_id=(payload or {}).get("task_id"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "grader_score": observation.grader_score,
        "observation": observation.model_dump(),
        "reward": observation.reward,
        "done": observation.done,
    }


@app.post("/step")
async def http_step(action: SupportTriageAction) -> Dict[str, Any]:
    try:
        observation = http_env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "grader_score": observation.grader_score,
        "observation": observation.model_dump(),
        "reward": observation.reward,
        "done": observation.done,
    }


@app.get("/state", response_model=SupportTriageState)
async def http_state() -> SupportTriageState:
    try:
        return http_env.state
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
