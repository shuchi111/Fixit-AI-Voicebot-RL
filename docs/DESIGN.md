# Design Document: Voicebot Reinforcement Learning

This project builds a conservative offline reinforcement learning system for a real-estate voicebot. It learns a turn-level dialogue policy from 1,500 unlabeled transcript logs, evaluates the learned policy before deployment, and exposes human controls for inspection, freezing, and rollback.

## MDP Formulation

The episode is one call. This matches the turn-level decision problem: the bot chooses what type of response to make at each AI assistant turn, and the outcome of those choices is observable within the call transcript. I rejected a customer-lifetime episode because the data only contains isolated calls, not persistent customer identity or future outcomes. I also rejected a single call-level action because it would hide the actual sequential decision points where the bot can recover from objections, busy signals, and repeated questions.

The state is an interpretable vector extracted before each bot decision. `src/features.py` computes transcript-derived signals such as filled slots, objection score, busy score, engagement, hostility, repeated bot behavior, collision-like turns, and normalized turn position. `src/policy/discretizer.py` maps those features into compact state keys such as `s2|o1|b0|e1|h0|c0|r0|t3`. This is intentionally not a full-text embedding state. A full transcript embedding would be tempting, but harder to inspect, harder to constrain, and too easy to overfit in a small offline dataset.

The action space is abstract dialogue acts, implemented in `src/models.py` and inferred in `src/action_infer.py`: greet, ask budget, ask location, ask timeline, handle objection, acknowledge busy and defer, clarify repeat, provide information, graceful exit, and ask property type. I rejected free-form text generation as the RL action because the transcripts do not provide enough coverage for safe natural-language generation. In production, these actions would map to reviewed response templates or prompt branches.

The reward combines turn-level shaping and terminal call outcome proxies in `src/reward.py`. Turn rewards penalize collisions, repeated questions, ignored objections, ignored busy signals, and hostility increases. They reward customer substance, engagement, objection handling, and busy handling. Terminal rewards estimate whether the call gathered useful buyer information, ended with a good deferral, ended prematurely, or ended hostile. I rejected terminal-only reward because it would make credit assignment too weak for short noisy transcripts, and I rejected supervised labels because the transcript logs are unlabeled.

## Reward Hacking

The reward is a proxy for a good call, not the true business objective. Three policies can game it.

First, a policy could ask many slot-filling questions quickly to maximize terminal success from filled slots while annoying the customer. The design partially catches this with repeated-question penalties, busy handling penalties, hostility penalties, and premature/wasted-engaged terminal penalties.

Second, a policy could overuse `ACK_BUSY_DEFER` whenever it sees any uncertainty. That avoids penalties from pitching to a busy caller, but may miss good leads. The reward catches some of this because successful calls need customer substance and useful slots, but it can still slip through when a caller sounds mildly busy but would have continued.

Third, a policy could overuse `GRACEFUL_EXIT` to avoid negative turn rewards. The design catches obvious premature exits with a terminal penalty, but it is still vulnerable when transcripts are short and ambiguous. That is why the learned policy is constrained against the behavior policy, inspected by humans, and evaluated with uncertainty before use.

I am explicit about these weaknesses in the OPE report and in the human gate: a policy is not accepted just because a scalar reward improves.

## Exploration Without Victims

The system does not deploy random exploration to real callers. Day one behavior remains the current scripted or behavior policy estimated from logs (`BehaviorPolicy`). Learning happens offline from existing logged calls. `src/train.py` fits the behavior policy on the train split, then trains a tabular Q-learning policy over logged transitions. The learner is conservative: `src/policy/constraints.py` only switches away from the behavior action when Q improvement is large enough and the action remains near the logged behavior support.

In production, I would put the current scripted policy plus safety overrides in front of users on day one. New learned policies would be trained offline, inspected, evaluated with off-policy estimators, and only then promoted behind a human gate. Rare or unsupported actions should be explored first in a simulator or small manually reviewed shadow mode, not randomly on live calls.

## Trusting the Signal

A correction like "no, I meant Monday" is a precise local signal: the bot misunderstood or failed to track the user's answer. In this implementation, correction-like behavior contributes through objection, repeat, and mismatch features, and the reward penalizes ignored objections or repeated questions immediately. This kind of signal gets strong turn-level weight because it identifies a specific bad decision.

A post-call CSAT score would be weaker if it existed because it is delayed, noisy, and may reflect price, inventory, customer mood, or call quality outside the policy's control. The provided dataset does not include CSAT, so I do not pretend it does. If added later, I would use it as a low-weight terminal reward and keep precise transcript events as stronger local signals. The current reward is built only from observable transcript and turn metadata.

## Credit Assignment

Credit assignment is handled with temporal-difference Q-learning plus shaping rewards. A 12-turn call that escalates at the end does not blame only the last action. Immediate penalties fire on earlier turns where the bot repeats itself, ignores an objection, pitches despite a busy signal, or increases hostility. The discounted return then propagates later consequences backward through the call.

The failure mode is that shaped rewards encode my assumptions. If hostility builds slowly from a good early discovery question followed by later bad handling, the terminal penalty can still reduce the value of early states. Conversely, a policy can receive positive shaping for gathering slots even if the final customer experience is poor. This is why reward breakdowns are saved with trajectories, and why the policy is not trusted without inspection and OPE.

## Offline Evaluation That Lies

The primary offline estimator is Doubly Robust evaluation with bootstrap confidence intervals, implemented in `src/eval/ope.py` and run through `src/eval/run_eval.py`. IPS/SNIPS is also reported, using propensities from the behavior policy. The evaluation uses a held-out test split of calls and clips importance weights to reduce variance.

The core assumptions are support and no hidden confounding. Support means the evaluation policy must mostly choose actions that the logged behavior policy took in similar states. No hidden confounding means the transcript-derived state must capture the reasons why actions were chosen and why outcomes happened. Both assumptions are imperfect here.

A realistic failure case is a rare high-intent buyer state where the current bot almost always asks budget, but the learned policy chooses `HANDLE_OBJECTION` based on sparse evidence. If that action has little or no logged support, IPS can collapse to near-zero effective samples, and DR can become overconfident if the fitted value model extrapolates incorrectly. The project reports this honestly: the OPE report includes confidence intervals and diagnostics such as the number of nonzero weighted episodes.

## Should the Loop Be Fully Autonomous?

I would not let this learning loop run fully autonomously in production. The premise is self-improving, but this domain is high-risk: bad policies annoy real customers, lose leads, and can hide behind noisy proxy rewards. The loop should be autonomous up to candidate generation, not deployment.

The human gate sits after training and offline evaluation, before policy activation. A reviewer inspects the learned action distribution and top state changes with `uv run python run.py --inspect`, compares versions with `uv run python run.py --inspect --diff baseline learned_v1`, applies hard freeze rules with `uv run python run.py --freeze configs/freeze_rules.yaml`, and can roll back with `uv run python run.py --rollback baseline`.

The cost is slower learning: a 1-3 day review cycle instead of continuous automatic updates. The benefit is that policy mistakes are cheap to stop and undo, which is essential for a customer-facing voicebot. This project optimizes for cautious improvement, not uncontrolled exploration.

## What I Would Do With 3x Time

I would build a small calibrated dialogue simulator for rare actions and counterfactual stress tests, add manual review labels for a stratified sample of calls, and replace the tabular state discretizer with a constrained interpretable model that can generalize better while still producing human-readable explanations.
