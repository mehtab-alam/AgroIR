
# How to run the script


```sh

python -m scripts.run_extraction \
  --corpus data/corpus.csv \
  --tm-seed data/TM+_seed_terms.csv \
  --owt-seed data/OWT+_seed_terms.csv \
  --av-seed data/AV+_seed_terms.csv \
  --outdir output
  --use-llm \
  --llm-base-url http://localhost:1234 \
  --llm-model mistral-7b-instruct-v0.3

```


