import argparse

"""
CLI arguments for random seeds
"""
randomness_parser = argparse.ArgumentParser(add_help=False)
randomness_parser.add_argument("--seed", type=int, default=42)
randomness_parser.add_argument("--torch-seed", type=int, default=43)

"""
CLI arguments for speech encoders, including for managing which device they
are on
"""
ssl_parser = argparse.ArgumentParser(add_help=False)
# currently --ckpt_path in label-audio.py
ssl_parser.add_argument("--ssl-model", default="facebook/hubert-large-ll60k",
                        help="also try others like facebook/w2v-bert-2.0")
ssl_parser.add_argument("--layer", type=int, default=22)
ssl_parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
ssl_parser.add_argument("--compile", action="store_true")  # I guess?
ssl_parser.add_argument("--pooling-width", type=int, default=1, help="1 recovers no pooling")
ssl_parser.add_argument("--pooling-type", choices=["mean", "max"], default="mean")
ssl_parser.add_argument("--cpu", action="store_true", help="only useful for debugging")

"""
CLI arguments for loading datasets (currently not included: --max-hours and
--hours-per-shard, which are specified in save-features.py
"""
dataset_parser = argparse.ArgumentParser(add_help=False)
dataset_parser.add_argument("--config", nargs="+")
dataset_parser.add_argument("--dataset-weights", nargs="*", type=float,
                            help="weights for dataset sampling")
dataset_parser.add_argument("--resample-to", type=int, default=16000)
dataset_parser.add_argument("--start-ix", type=int, default=0,
                            help="For slicing an HF dataset (start index in the corpus)")
dataset_parser.add_argument("--n-examples", type=int, default=0,
                            help="Number of examples to take, starting with start-ix")
dataset_parser.add_argument("--batch-size", type=int, default=1,
                            help="Number of seconds if token_batching==True, otherwise number of sentences")
dataset_parser.add_argument("--token-batching", action="store_true")
dataset_parser.add_argument("--num-workers", type=int, default=1)
dataset_parser.add_argument("--example-lengths", default=None)

"""
CLI arguments for DSUs
"""
dsu_parser = argparse.ArgumentParser(add_help=False)  # as in for the kmeans model stuff
dsu_parser.add_argument("--kmeans-model", default="/mnt/scratch-artemis/kshitij/clustering/kmeans_model/3datsets_combined_kmeans_5000")
dsu_parser.add_argument("--as-indices", action="store_true")
dsu_parser.add_argument("--no-dedup", action="store_true")
