"""RPC adapter for the existing measurement/decision trace evaluator."""

from dataclasses import asdict
import json
from pathlib import Path

from .core import State, advance, task_complete
from .online import OpportunityBook, physical_choice, score_trace


class WeighingEvaluation:
    def __init__(self, contract, instance):
        self.contract = contract
        self.instance = instance
        self.reset()

    def reset(self):
        self.state = State()
        self.book = OpportunityBook(self.contract, self.instance, "native")
        self.trace = []
        self.pending = None
        self.invalid_sensor = False
        self.safety_violation = False
        self._scores = None

    def emit(self, row):
        self.trace.append(row)
        self._scores = None

    def update(self, measured, detail=None):
        advance(self.contract, self.instance, self.state, measured)
        detail = asdict(measured) if detail is None else detail
        step = measured.step
        self.emit({"type": "measurement", "step": step, "measurement": detail})
        self.invalid_sensor |= not measured.sensor_valid
        self.safety_violation |= not measured.sample_untouched or not measured.no_hand_press
        fingers = {int(key): value for key, value in detail.get("finger_contacts", {}).items()}
        touching_weight = any(value.get("left_finger_contact") or value.get("right_finger_contact")
                              for value in fingers.values())
        candidate = (physical_choice(fingers, self.book.current_signature[0])
                     if self.book.current_signature is not None else None)
        if self.pending is not None and self.state.reading_stable and measured.released:
            action, started = self.pending
            changed = ((action["weight_g"] in self.state.supported_weights_g) ==
                       (action["type"] == "add"))
            self.emit({"type": "operation_result", "step": step, "action": action,
                       "started_step": started, "status": "confirmed" if changed else "failed",
                       "physical_supported_weights_g": list(self.state.supported_weights_g)})
            self.pending = None
            opened = (self.book.update(self.state, step, evidence_step=step) if changed else
                      self.book.retry_after_failure(self.state, step, evidence_step=step))
        else:
            opened = self.book.update(
                self.state, step, evidence_step=step,
                operation_in_flight=self.pending is not None or
                (touching_weight and self.book.first_choice is None))
        if opened:
            self.emit(opened)
        if candidate and self.book.first_choice is None and self.pending is None:
            choice = self.book.record_choice(candidate, step,
                                            evidence_step=self.book.current_evidence_step,
                                            source="physical_interaction")
            if choice:
                self.emit(choice)
                self.pending = (candidate, step)
                self.emit({"type": "operation_start", "step": step,
                           "action": candidate, "source": "physical_interaction"})

    @property
    def scores(self):
        if self._scores is None:
            self._scores = score_trace(self.contract, self.instance, self.trace)
        return self._scores

    @property
    def success(self):
        # Avoid replaying the full trace at every 120-Hz physics tick. Always
        # confirm a potential terminal state with the original offline audit.
        return bool(not self.invalid_sensor and not self.safety_violation and
                    task_complete(self.contract, self.instance, self.state) and
                    self.scores["valid_success"])

    def status(self):
        scores = self.scores
        success = bool(scores["valid_success"])
        return {
            "phase": ("success" if success else "sensor_invalid" if self.invalid_sensor else
                      "settling" if not self.state.reading_stable else "weighing"),
            "success": success, "hold_steps": self.state.final_steps,
            "required_steps": self.contract.final_steps,
            # v1 has no scalar score. This is only the common client's 0/100
            # display; all original metrics remain authoritative below.
            "score": 100 if success else 0, "max_score": 100,
            "score_definition": "100 * valid_success (RPC compatibility)",
            "weighing_scores": dict(scores),
        }

    def save(self, directory, manifest):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (directory / "events.jsonl").open("w") as stream:
            for row in self.trace:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        (directory / "scores.json").write_text(json.dumps(self.scores, indent=2) + "\n")
