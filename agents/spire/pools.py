"""Which encounters an act can still roll. A fight faced in an act does not come again there, so "remaining" = the act variant's
weak + regular + elite pool minus what the run has fought in it. The pools are the game's own (spire-zero SimWorker `list`, v0.113.1).
The boss is not in it (RunState.boss has it)."""
from __future__ import annotations

POOLS: dict[str, list[str]] = {
    "OVERGROWTH": "FUZZY_WURM_CRAWLER_WEAK NIBBITS_WEAK SHRINKER_BEETLE_WEAK SLIMES_WEAK CUBEX_CONSTRUCT_NORMAL FLYCONID_NORMAL FOGMOG_NORMAL INKLETS_NORMAL MAWLER_NORMAL NIBBITS_NORMAL OVERGROWTH_CRAWLERS RUBY_RAIDERS_NORMAL SLIMES_NORMAL SLITHERING_STRANGLER_NORMAL SNAPPING_JAXFRUIT_NORMAL VINE_SHAMBLER_NORMAL BYGONE_EFFIGY_ELITE BYRDONIS_ELITE PHROG_PARASITE_ELITE",
    "UNDERDOCKS": "CORPSE_SLUGS_WEAK SEAPUNK_WEAK SLUDGE_SPINNER_WEAK TOADPOLES_WEAK CORPSE_SLUGS_NORMAL CULTISTS_NORMAL FOSSIL_STALKER_NORMAL GREMLIN_MERC_NORMAL HAUNTED_SHIP_NORMAL LIVING_FOG_NORMAL PUNCH_CONSTRUCT_NORMAL SEAPUNK_NORMAL SEWER_CLAM_NORMAL TWO_TAILED_RATS_NORMAL PHANTASMAL_GARDENERS_ELITE SKULKING_COLONY_ELITE TERROR_EEL_ELITE",
    "HIVE": "BOWLBUGS_WEAK EXOSKELETONS_WEAK THIEVING_HOPPER_WEAK TUNNELER_WEAK BOWLBUGS_NORMAL CHOMPERS_NORMAL EXOSKELETONS_NORMAL HUNTER_KILLER_NORMAL LOUSE_PROGENITOR_NORMAL MYTES_NORMAL OVICOPTER_NORMAL SLUMBERING_BEETLE_NORMAL SPINY_TOAD_NORMAL THE_OBSCURA_NORMAL DECIMILLIPEDE_ELITE ENTOMANCER_ELITE INFESTED_PRISMS_ELITE",
    "GLORY": "DEVOTED_SCULPTOR_WEAK SCROLLS_OF_BITING_WEAK TURRET_OPERATOR_WEAK AXEBOTS_NORMAL CONSTRUCT_MENAGERIE_NORMAL FABRICATOR_NORMAL FROG_KNIGHT_NORMAL GLOBE_HEAD_NORMAL OWL_MAGISTRATE_NORMAL SCROLLS_OF_BITING_NORMAL SLIMED_BERSERKER_NORMAL THE_LOST_AND_FORGOTTEN_NORMAL KNIGHTS_ELITE MECHA_KNIGHT_ELITE SOUL_NEXUS_ELITE",
}
POOLS = {k: v.split() for k, v in POOLS.items()}


class Faced:
    """The encounters a run has fought in its current act."""

    def __init__(self):
        self.act, self.seen = None, set()

    def combat(self, run: dict, encounter: str | None) -> None:
        self._act(run)
        if encounter:
            self.seen.add(encounter)

    def remaining(self, run: dict) -> list[str]:
        self._act(run)
        return [e for e in POOLS.get(run.get("act_id") or "", []) if e not in self.seen]

    def _act(self, run: dict) -> None:
        if run.get("act") != self.act:
            self.act, self.seen = run.get("act"), set()
