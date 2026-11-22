"""Vakuu: an AI that plays Slay the Spire 2, trained by self-play on spire-zero.

    engine/          spire-zero binding: the hosted game (host.py), its Rust combat engine (engine.py), id tables
    buffer/          the fixed-width decision record every model reads: layout, encoders, token decode
    common/          entity vocabulary and embeddings shared by the nets
    claws/           in-combat policy + value net, its data generation, training and evaluation
    golden_compass/  run-level hazard net (map path, card rewards, shops, rests), its data generation and training
    branch/          paired-branch evaluation of out-of-combat choices (extra training signal for the compass)
    macro/           the heuristics that carry a run through the screens no net decides
"""
