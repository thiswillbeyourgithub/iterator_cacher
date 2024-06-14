from typing import List
from string import ascii_letters
import numpy as np
import code

from iterator_cacher import IteratorCacher


lt = ascii_letters[:list(ascii_letters).index("A")]

@IteratorCacher(
    cache_location="test",
    iter_list=["texts"],
    verbose=True,
    unpacking_func=lambda ar: ar.tolist(),
    repacking_func=lambda l: np.array(l),
)
def test(texts: List[str], y, z) -> np.ndarray:
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
print("Expect 1 print of 1/3 of the alphabet")
out1 = test(texts=third, **kwargs)
p()

print("Expect no print")
out2 = test(texts=third, **kwargs)
p()

print("Expect 1 print for the letters not in 1/3 but in 1/2 of the alphabet")
out3 = test(texts=half, **kwargs)
p()

print("Expect 1 print for the letters of the 2nd 1/2 of the alphabet")
out4 = test(texts=lt, **kwargs)
p()

print("Expect no print")
out2 = test(texts=lt + lt, **kwargs)
p()
