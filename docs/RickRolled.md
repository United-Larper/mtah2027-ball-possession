# Frame-level ball possession from automatic tracking

**Team RickRolled · More Than A Hack 2027, Initial Challenge · November 2026**

Code: https://github.com/harithnabilf/mtah2027-ball-possession

## 1. Summary

We label each frame as home possession, away possession or dead ball with classifiers trained
only on other matches, a decoder that turns their per-frame probabilities into possessions, and
a layer of evidence from the broadcast recording and the Smart Tagging whose weight is fitted
without labels. On held-out external matches, with their tracking degraded to resemble this
match, the system scores spellF1 0.680 against 0.327 for the organisers' nearest-player baseline.
On this match it outputs 318 possessions and labels 44.9% of graded frames dead ball.

## 2. What the data told us before we built anything

We measured the tracking without looking at any label. Four findings set the design:

- **The ball is lost in long blocks, not short gaps.** 40.7% of graded frames have no ball, almost
  all in about 130 blocks of 5 seconds or more. These blocks cluster around stoppages: players'
  mean speed is under 1.0 m/s on 47% of block frames, against 8% of ball-present frames.
- **The nearest player is weak evidence.** The team of the nearest player changes 29 times per
  minute of ball-present play.
- **The broadcast recording can be synchronised exactly.** Matching the two ball tracks gives one
  constant offset per half, accurate to one frame. Player positions confirm it: 87% of broadcast
  players have a same-team tactical player within 2 m at the found offset, 49% at 25 frames away.
- **Replays and close-ups sit inside stoppages.** Player speed falls from 2.2 to 1.4 m/s in the five
  seconds before the broadcast leaves the wide shot and stays low until it returns.

## 3. Approach

**Training on other matches.** The rules forbid labels on this match, so every model is trained
on ten external matches (section 5). Their tracking is clean, so we degrade it to resemble this
match: tracks smoothed to the same second-difference statistics, the ball removed in long blocks
that start at stoppages and end a few seconds after the restart, and players outside a panning
field of view dropped. The simulator's parameters are fitted to label-free statistics of this
match and randomised across training copies, because several of them are uncertain.

**Features.** 333 per-frame features that need no player identity, computed in a frame where the
home team attacks towards +x: ball speed and direction changes, distances from the ball to the
nearest players of each team, touch events per team and the time since and until each team's
touches, what the ball was doing before and after a gap, each team's shape, depth and movement,
and causal and anti-causal averages over 0.4 to 10 seconds.

**Two decisions, two models.** A first version used one three-class model. Checked against the
public five minutes it almost never labelled a stoppage while the ball was tracked. We traced
this to the external data: the ball is frozen during every stoppage in DFL and blanked in the
Metrica samples, so "the ball is moving" meant "alive" with certainty in training, while in this
match a tracked ball keeps moving during stoppages. We therefore split the problem:

- **Alive or dead** is decided by a model that sees player movement, whether the ball is tracked
  and where it is, but never how it moves or how close players are to it (154 features).
- **Home or away** is decided by a second model trained on alive frames only, with all features.
- Three **transition detectors** (team change, alive to dead, dead to alive) score each frame.

All five are LightGBM classifiers. No parameter was fitted on the public window; the check
changed which information the alive/dead model is allowed to see, and that restriction follows
from the external datasets, not from this match.

**Decoding.** A Viterbi pass over the three states per half. Switching state at a frame costs a
constant minus a multiple of the matching detector's log-odds, so a change of possession is
cheap where a detector fires and expensive elsewhere. This keeps real short possessions and
removes flicker. The eight decoder parameters are tuned on out-of-fold predictions for the
external matches.

**Evidence from the broadcast and the tags.** Four per-frame streams are added to the decoder's
scores: the broadcast state (unmapped, replay or close-up, replay with a pitch view, live without
ball, live with ball), the phase of a set-piece window, the goal window, and the team named by an
attacking-phase tag. The table of each stream given the state is fitted by EM against the
classifiers' own confident frames, with no labels. Because a 20-second replay is one observation
and not 500, the streams are down-weighted by a factor fixed in advance (0.3) from a synthetic
study. Where the tactical ball is missing and the broadcast shows live play with the ball, the
broadcast ball fills the gap (5,511 frames).

## 4. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical tracking | Main input: players, ball, speeds |
| Broadcast tracking and homography | Synchronised to the tactical timeline; ball fill and broadcast-state evidence |
| Smart Tagging | Kick-off rule; set-piece, goal and attacking-phase windows as evidence |
| Tactical homography | Analysis only: camera position, field-of-view statistics for the simulator |
| Both videos | Not used. Decoding is fast, but we had no labelled way to validate a ball detector |
| Audio | Tried for whistle detection and dropped: the tactical audio is very quiet and candidates did not validate |
| Creating Chances, Ball in the Box, Players in the Box tags | Dropped: fixed windows that do not track possession |

## 5. Training data

None of it comes from the provided match.

| Dataset | Matches | Labels | Who labelled |
|---|---|---|---|
| DFL / IDSSE open dataset (Bassek et al. 2025, CC BY 4.0) | 7, about 1.0 M frames | `BallStatus` and `BallPossession` on every frame | The data provider (DFL) |
| Metrica Sports sample data | 3, about 0.43 M frames | Derived by our code from Metrica's event files | Metrica (events); our rule (per-frame labels) |

Our Metrica rule: the team of the most recent pass, carry, shot, recovery or set piece has the
ball; the ball is dead between an event that stops play and the next set piece. DFL counts
briefer touches as possession than Metrica's events do, so we relabel any DFL possession shorter
than 1.0 s that sits between two alive stretches of the other team. That threshold was chosen
because it makes DFL's possession-length distribution match the Metrica samples (387 against 386
possessions per match, median 4.4 s in both); it changes 0.22% of frames. We added no hand labels.

## 6. Possession rules implemented

- **Controlled possession.** A team change needs evidence across frames: the home/away model's
  probabilities must outweigh the switching cost, which the team-change detector lowers at a
  real hand-over. A sub-second touch inside the other team's possession is not a change.
- **Loose and contested balls** stay with the last team: with no new evidence the decoder does
  not switch, and the training labels carry the same convention.
- **Dead ball** is called from player movement, ball availability and position, the broadcast
  state and the tag windows. Frames before each half's tagged kick-off are dead.
- **Restarts.** The dead-to-alive detector times the restart; the home/away model names the team.
  Set-piece windows only add evidence, because the tags mark a window, not an instant.
- **No abstention** inside the graded halves; `X` is written only outside them.

## 7. Validation

Grouped five-fold cross-validation over the ten external matches, scored with the official scorer.

| System (held-out matches) | spellF1 clean | spellF1 degraded | macroF1 degraded |
|---|---|---|---|
| Organisers' baseline, ±50 frames | 0.341 | 0.327 | 0.690 |
| Our models, per-frame argmax | 0.450 | 0.430 | 0.913 |
| Our models, constant-cost Viterbi | 0.709 | 0.673 | 0.914 |
| Our models, learned-transition Viterbi | **0.715** | **0.680** | **0.915** |

Scores are similar on DFL (0.680) and on the Metrica samples (0.679). The decoder parameters were
tuned on the same out-of-fold pool, so these figures are slightly optimistic. On degraded
held-out matches the alive/dead model has dead-ball recall 0.92 with the ball present and 0.97
with it missing. The broadcast and tag evidence cannot be cross-validated, because the external
matches have neither.

As a check of alignment and label meaning only, the final file scores macroF1 0.898 and spellF1
0.739 on the public five minutes (16 predicted possessions, 17 true). Seventeen possessions
cannot rank methods and we did not use this to choose anything.

## 8. Limitations

- **Label convention.** Our training labels come from DFL and from Metrica's events, not from
  Metrica's corrected tracking. If the true convention is coarser, we over-split possessions.
- **Stoppage timing without a ball.** When the ball is not tracked, the start and end of a
  stoppage come from player movement and broadcast cuts, which lag the whistle by a second or two.
- **Slow build-up.** The alive/dead model cannot see ball movement, so slow passing among
  defenders can look like a stoppage where the broadcast gives no help.
- **Simulator assumptions.** We assume ball-missing blocks follow stoppages and approximate the
  field-of-view dropout; the simulated matches are easier than the real one.
- **Fixed evidence weight.** The down-weighting of broadcast and tag evidence is set in advance,
  not learned, and the fitted tables inherit any bias in the classifiers.
- **With more time:** detect the ball in the tactical video inside the missing blocks, using the
  tracked frames as pseudo-labels; replace per-frame trees with a sequence model; and label a few
  outside matches in Metrica's convention to remove the label mismatch.

## 9. Reproduction

Put the data bundle in `./data` or set `MTAH_DATA_DIR`, then:

```
uv sync
uv run python scripts/run.py
```

This writes `outputs/RickRolled.csv` with SHA-256 `d118760450ec0a3c…f3d4d46d` (full value in the
README). Models and fitted tables are committed, so the run has no training and no randomness; a
fresh clone reproduced the file byte for byte. The README lists the commands that rebuild the
models from the external data.

## 10. Disclosure

The code was written with an AI coding assistant. In line with the competition terms, no match
data or extract was sent to any hosted language model: the assistant saw only aggregate outputs
such as scores, counts and summary statistics. No language model is part of the pipeline.
