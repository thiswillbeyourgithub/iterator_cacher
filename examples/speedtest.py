import time
from typing import List
from string import ascii_letters
import numpy as np
import joblib
from functools import wraps

from iterator_cacher import IteratorCacher

verbose = False


# lt = ascii_letters[:list(ascii_letters).index("A")]
lt = ascii_letters

cache_dir = joblib.Memory("test_iterator", verbose=verbose)
cache_dir.clear()

def speedtest(store):
    def metawrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t = time.time()
            out = func(*args, **kwargs)
            t =  time.time() - t
            store[0] += t
            return out
        return wrapper
    return metawrapper

times = [[0], [0]]

def actual_exec(texts: List[str], y, z, silent=False) -> np.ndarray:
    # fake embeddings
    if not silent:
        print(f"Computing {texts}, {y}, {z}")
    assert texts, texts
    embeds = [lt.index(li[0]) for li in texts]
    ar = np.array(embeds)
    time.sleep(0.1)
    return ar

test2 = speedtest(times[1])(actual_exec)

test1 = speedtest(times[0])(
    IteratorCacher(
            memory_object=cache_dir,
            iter_list=["texts"],
            verbose=verbose,
            res_to_list=lambda ar: ar.tolist(),
    )(actual_exec)
)

def p():
    input("Press any key to continue.")

third = list(lt[:len(lt) // 3])
half = list(lt[:len(lt) // 2])
kwargs = {"y": 10, "z": 20}

# cache half of letters
for test in range(10):
    out1 = test1(texts=third, **kwargs)
    out2 = test1(texts=third, **kwargs)
    out3 = test1(texts=half, **kwargs)
    out4 = test1(texts=lt, **kwargs)
    out2 = test1(texts=lt + lt, **kwargs)

    out1 = test2(silent=True, texts=third, **kwargs)
    out2 = test2(silent=True, texts=third, **kwargs)
    out3 = test2(silent=True, texts=half, **kwargs)
    out4 = test2(silent=True, texts=lt, **kwargs)
    out2 = test2(silent=True, texts=lt + lt, **kwargs)


time_with_caching = times[0]
time_wo_caching = times[1]

print(f"\nFinal times:\nCached version: {time_with_caching}\nNon cacher version: {time_wo_caching}")

