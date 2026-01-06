#!/usr/bin/env python

from setuptools import setup

# many of these can probably be relaxed
requirements = [
    "torch>=2.8.0",
    "transformers>=4.57.1",
    "datasets>=2.19.2,<4.0.0",
    "vllm>=0.11.0",
    "protobuf>=5.29.3",
    "scikit-learn>=1.3.2",
    "librosa>=0.10.2.post1",
    "whisper-normalizer>=0.0.8",
    "evaluate>=0.4.2",
    "jiwer>=4.0.0",
    "starlette>=0.49.1",
    "npy-append-array>=0.9.19"
]

setup(name='SpireLM',
      version='0.1.1',
      description='LLM for speech and text tasks',
      author='Ben Peters',
      author_email='benzurdopeters@gmail.com',
      url='https://github.com/utter-project/SpireLM',
      packages=['spire'],
      install_requires=requirements,
      license="MIT"
     )
