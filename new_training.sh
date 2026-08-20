export vakuu="/Users/kylelau519/Programming/Vakuu/"
rm $vakuu/agents/whispering_earring/data/raw/*.json


python $vakuu/agents/whispering_earring/facility.py --trials 40 --branch-workers 12 --branches-per-combat 3 --main-workers 4
python $vakuu/agents/claws/model/materialize.py $vakuu/agents/whispering_earring/data/raw/*.json --out dataset.pt
python $vakuu/agents/claws/model/train.py $vakuu/agents/claws/model/dataset.pt --epochs 20 --out $vakuu/agents/claws/model/claws.pt


