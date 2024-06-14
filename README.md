# iterator_cacher
* Simple python decorator to transparently cache each element of an iterable.

# Why?
* Imagine you want to create LLM embeddings for 10 sentences in a single call. With most caching mechanism, if you now ask those 10 sentences with another sentence on top (so an iterable of 11), the caching will not be used for the 10 already known values. Same if you ask only 9 of the 10.

# Getting started
* ` python -m pip install -e .`
TODO_tutorial
