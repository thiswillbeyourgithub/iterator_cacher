import time
from typing import List
from string import ascii_letters
import numpy as np
import code

from iterator_cacher import IteratorCacher


# lt = ascii_letters[:list(ascii_letters).index("A")]
lt = ascii_letters

def speedtest(store):
    def meta_wrapper(func):
        def wrapper(*args, **kwargs):
            t = time.time()
            out = func(*args, **kwargs)
            t =  time.time() - t
            store[0] += t
            return out
        return wrapper
    return meta_wrapper

times = [[0], [0]]

@speedtest(store=times[0])
@IteratorCacher(
    cache_location="test",
    iter_list=["texts"],
    verbose=True,
    unpacking_func=lambda ar: ar.tolist(),
    repacking_func=lambda l: np.array(l),
)
def test1(texts: List[str], y, z) -> np.ndarray:
    # fake embeddings
    print(f"Computing {texts}, {y}, {z}")
    assert texts, texts
    embeds = [lt.index(l[0]) for l in texts]
    ar = np.array(embeds)
    return ar

@speedtest(store=times[1])
def test2(texts: List[str], y, z) -> np.ndarray:
    # fake embeddings
    print(f"Computing {texts}, {y}, {z}")
    assert texts, texts
    embeds = [lt.index(l[0]) for l in texts]
    ar = np.array(embeds)
    return ar

def p():
    input("Press any key to continue.")

third = lt[:len(lt) // 3]
half = lt[:len(lt) // 2]
kwargs = {"y": 10, "z": 20}

# cache half of letters
out1 = test1(texts=third, **kwargs)
out2 = test1(texts=third, **kwargs)
out3 = test1(texts=half, **kwargs)
out4 = test1(texts=lt, **kwargs)
out2 = test1(texts=lt + lt, **kwargs)

out1 = test2(texts=third, **kwargs)
out2 = test2(texts=third, **kwargs)
out3 = test2(texts=half, **kwargs)
out4 = test2(texts=lt, **kwargs)
out2 = test2(texts=lt + lt, **kwargs)

time_with_caching = times[0]
time_wo_caching = times[1]


breakpoint()
