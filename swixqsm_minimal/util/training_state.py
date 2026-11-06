"""Current-run phase and model-selection bookkeeping, independent of training.

Call complete_step only after a successful optimizer/training step. Loading a
source checkpoint supplies no evidence about its earlier training history;
that provenance must remain separate from this run's counters and phase names.
"""
import copy
import math
from numbers import Integral, Real


_PHASES = ("l1", "warmup", "gan")
_COUNTERS = ("generator_updates", "discriminator_updates", "adversarial_generator_updates")


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _empty_counts():
    return {key: 0 for key in _COUNTERS}


class TrainingProgress:
    """Bookkeep L1, warmup, and observed adversarial phases of one new run.

    Epoch indices are zero based. Lower validation MAE wins; exact ties retain
    the earlier selected checkpoint. Every phase has a separate best file in
    addition to the global best.pt, so a warmup global best cannot be mistaken
    for a selected adversarial checkpoint.
    """
    def __init__(self, epochs, warmup_epochs, adv_weight):
        self.epochs = _integer(epochs, "epochs", 1)
        self.warmup_epochs = _integer(warmup_epochs, "warmup_epochs", 0)
        self.adv_weight = _finite(adv_weight, "adv_weight", nonnegative=True)
        if self.adv_weight > 0 and self.epochs <= self.warmup_epochs:
            raise ValueError("A GAN run requires epochs > warmup_epochs so that adversarial training is entered")
        self._updates = _empty_counts()
        self._updates_by_phase = {phase: _empty_counts() for phase in _PHASES}
        self._updates_by_epoch = {}
        self._validated = {}
        self._best_by_phase = {phase: None for phase in _PHASES}
        self._global_best = None
        self._latest_observed_epoch = -1

    @property
    def generator_updates(self):
        return self._updates["generator_updates"]

    @property
    def discriminator_updates(self):
        return self._updates["discriminator_updates"]

    @property
    def adversarial_generator_updates(self):
        return self._updates["adversarial_generator_updates"]

    def _epoch(self, epoch_index):
        epoch_index = _integer(epoch_index, "epoch_index", 0)
        if epoch_index >= self.epochs:
            raise ValueError("epoch_index is outside this run's requested epochs")
        return epoch_index

    def _chronological(self, epoch_index):
        if epoch_index < self._latest_observed_epoch:
            raise ValueError("Training progress cannot move to an earlier epoch")

    def phase(self, epoch_index):
        epoch_index = self._epoch(epoch_index)
        if self.adv_weight == 0:
            return "l1"
        return "warmup" if epoch_index < self.warmup_epochs else "gan"

    def weight(self, epoch_index):
        return self.adv_weight if self.phase(epoch_index) == "gan" else 0.

    def complete_step(self, epoch_index):
        """Count one actually completed training step in the current run.

        A GAN step is the existing combined D/G update. This helper does not
        execute optimizers and cannot infer success from weights or loss values.
        The caller must invoke it only after train_step succeeds.
        """
        epoch_index = self._epoch(epoch_index)
        self._chronological(epoch_index)
        if epoch_index in self._validated:
            raise ValueError("Cannot add training steps after this epoch's validation was recorded")
        phase = self.phase(epoch_index)
        epoch_counts = self._updates_by_epoch.setdefault(epoch_index, _empty_counts())
        active = ("generator_updates",) if phase != "gan" else _COUNTERS
        for key in active:
            self._updates[key] += 1
            self._updates_by_phase[phase][key] += 1
            epoch_counts[key] += 1
        self._latest_observed_epoch = epoch_index

    def _require_observed_gan(self, epoch_index):
        if (self.phase(epoch_index) == "gan"
                and self._updates_by_epoch.get(epoch_index, {}).get("adversarial_generator_updates", 0) == 0):
            raise ValueError("A GAN epoch cannot be recorded or saved without actual adversarial updates in that epoch")

    def record_validation(self, epoch_index, score):
        """Record val physical participant-equal MAE and return best file names.

        Validation must follow this epoch's training. Repeated validation of an
        epoch or going backwards is rejected to keep saved-role attribution
        unambiguous. An L1/warmup initial evaluation with no updates can still
        be reported honestly with zero counters; GAN selection always requires
        observed adversarial updates in that epoch.
        """
        epoch_index = self._epoch(epoch_index)
        score = _finite(score, "validation score")
        self._chronological(epoch_index)
        if epoch_index in self._validated:
            raise ValueError("Validation was already recorded for this epoch")
        self._require_observed_gan(epoch_index)
        phase = self.phase(epoch_index)
        record = {"phase": phase, "epoch_index": epoch_index, "epoch": epoch_index + 1,
                  "score": score, "actual_adv_weight": self.weight(epoch_index)}
        files = []
        if self._global_best is None or score < self._global_best["score"]:
            self._global_best = dict(record)
            files.append("best.pt")
        previous = self._best_by_phase[phase]
        if previous is None or score < previous["score"]:
            self._best_by_phase[phase] = dict(record)
            files.append(f"best_{phase}.pt")
        self._validated[epoch_index] = dict(record)
        self._latest_observed_epoch = epoch_index
        return files

    def snapshot(self, epoch_index, units=None):
        """Return plain JSON/weights_only-safe metadata for the current epoch.

        No infinity sentinel is serialized. The phase describes only this run,
        even when initialization came from previously adversarial weights.
        """
        epoch_index = self._epoch(epoch_index)
        self._chronological(epoch_index)
        self._require_observed_gan(epoch_index)
        return {
            "schema": "swixqsm.training_progress", "version": 1,
            "epoch_index": epoch_index, "epoch": epoch_index + 1,
            "epochs_requested": self.epochs, "warmup_epochs_requested": self.warmup_epochs,
            "adv_weight_requested": self.adv_weight,
            "current_run_phase": self.phase(epoch_index), "actual_adv_weight": self.weight(epoch_index),
            **self._updates,
            "updates_by_phase": copy.deepcopy(self._updates_by_phase),
            "current_epoch_updates": dict(self._updates_by_epoch.get(epoch_index, _empty_counts())),
            "epochs_with_updates": sorted(self._updates_by_epoch),
            "validated_epoch_indices": sorted(self._validated),
            "best_by_phase": copy.deepcopy(self._best_by_phase),
            "global_best": copy.deepcopy(self._global_best),
            "selection_split": "val", "selection_metric": ("stored_participant_equal_mae" if units == "stored" else "physical_participant_equal_mae"),
            "selection_metric_description": ("Unverified original-storage-scale MAE aggregated per participant and then averaged equally"
                                              if units == "stored" else "Physical-unit MAE aggregated per participant and then averaged equally"),
            "selection_mode": "min", "tie_policy": "strict improvement; retain earlier selection",
            "phase_scope": "current_run_only; source history separate",
            "source_history_inferred": False,
            "gan_training_observed": self.adversarial_generator_updates > 0,
        }
