# choice_paradox

Card reward, removal, and upgrade picker. Currently, Baalorlord tier list card picking (`policy.py`)
is used as a placeholder, in order to train whispering_earring with a reasonable deck. The Act 1
boss struggle may not be a combat-skill problem at all -- it may be that this placeholder produces
decks Claws can't win with no matter how well it plays them.

The planned fix is self-play branch-and-score training, the same scheme that trains Claws -- not
supervised learning on scraped human run data (rejected: STS2 is too young/complex a game for that
data to be sufficient, and even with enough of it, human outcomes are confounded by human combat
skill rather than Claws's). See [`model/README.md`](model/README.md) for the design (not
implemented yet).
