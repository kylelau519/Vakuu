# headless_runner

Launching and supervising headless instances of the game for trial generation.

**Superseded for the primary route.** `/Users/kylelau519/Programming/sts2-cli` (outside this repo)
drives headless trials without launching Godot at all, so it never hits the crash documented below —
there's no display driver, no asset preload, nothing to SIGABRT. See
`.claude/skills/vakuu/SKILL.md`'s "Current phase" section. This directory's findings stay useful as a
record of the mod-based dead end, and as reference if that route is ever revisited. It is tested `sts2-cli` can run on multiple terminals at the same time.