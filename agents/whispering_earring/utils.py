import random
import re
from buffer_env import BufferDecision
from env import Sts2ActionError

SEED_WORDS = ["falcon", "ember", "wolf", "raven", "thorn", "comet", "storm", "harbor"]

HEALTHY_HP_FRACTION = 0.5
LOW_HP_SLOPE = 3.0

HP_WEIGHT = 0.75
BOSS_HP_WEIGHT = 0.2
BOSS_WIN_FLOOR = 0.8

GOLDEN_COMPASS_GAMMA = 0.9
GOLDEN_COMPASS_ELITE_BONUS = 2.0
GOLDEN_COMPASS_BOSS_BONUS = 3.0

def random_seed() -> str:
    return f"{random.choice(SEED_WORDS)}-{random.randint(0, 999_999)}"


# Strips ANSI/terminal control sequences (CSI, OSC, lone ESC) and other C0 control bytes (except
# \t/\n) out of relayed worker output before it reaches the real terminal -- see _tag_worker_stderr.
CONTROL_SEQUENCE_RE = re.compile(
    r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])"
    r"|[\x00-\x08\x0b-\x1f\x7f]"
)

# Sts2Headless's [SIM]/[INFO] logging is all free text through one generic Log() call in the
# engine 
MILESTONE_RE = re.compile(
    r"Creating RunState|Rooms generated|Run launched|relics finalized|Entered Act|"
    r"EnterRoom:|Post-combat:|Moving to map coord|Card reward pending|Selected card reward|"
    r"Bundle selection pending|has won against encounter|has lost"
)

# CoreCLR's own unhandled-exception/fatal-error dump. Without this, a CLR crash was silently dropped
# by _MILESTONE_RE in non-verbose mode, leaving a worker that just silently died or wedged with zero visible output.
CRASH_RE = re.compile(
    r"Unhandled [Ee]xception|Fatal error\.|Process terminating|Stack overflow\.|"
    r"Assertion failed|Segmentation fault|SIGABRT|SIGSEGV"
)


WORKER_TAG_COLORS = [31, 32, 33, 34, 35, 36, 91, 92, 93, 94, 95, 96]  # ANSI red..bright-cyan

META_ELIGIBLE_DECISIONS = frozenset({"card_reward", "rest_site", "event_choice"})
META_DRAIN_TIMEOUT = 120.0

# Slightly wrong, change later
def combat_value(decision: BufferDecision) -> float:
    if not decision.player_max_hp:
        return 1.0
    hp_frac = max(0.0, decision.player_hp) / decision.player_max_hp
    if decision.is_last_of_act:
        return BOSS_WIN_FLOOR + BOSS_HP_WEIGHT * hp_frac
    if hp_frac >= HEALTHY_HP_FRACTION:
        return hp_frac
    shortfall = HEALTHY_HP_FRACTION - hp_frac
    return HEALTHY_HP_FRACTION - shortfall * LOW_HP_SLOPE

def bypass_card_reward(session) -> None:
    # A branch's own reward pick is discarded either way -- only the main line's adopted actions
    # matter.
    raw = session._session.Action("skip_card_reward", None)
    if session._decision(raw).decision == "error":
        raise Sts2ActionError(session._session.LastError or "unknown error")