from iterator_cacher import IteratorCacher
from pathlib import Path
import litellm
from string import ascii_letters
from joblib import Memory
from functools import partial

lt = ascii_letters[:list(ascii_letters).index("A")]

cache_location = Path("test_iterator")
cache_location.mkdir(parents=True, exist_ok=True)
cache_location = Memory(cache_location, verbose=False)
cached = IteratorCacher(
    memory_object=cache_location,
    iter_list=["input"],
    verbose=True,
    res_to_list = lambda out: out.to_dict()["data"],
)(litellm.embedding)
embedder = partial(cached, model="openai/text-embedding-3-small")

def p():
    input("Press any key to continue.")


third = lt[:len(lt) // 3]
half = lt[:len(lt) // 2]
kwargs = {"y": 10, "z": 20}

# cache half of letters
print("Expect 1 print of 1/3 of the alphabet")
out1 = embedder(input=list(third))
p()

print("Expect no print")
out2 = embedder(input=list(third))
p()

print("Expect 1 print for the letters not in 1/3 but in 1/2 of the alphabet")
out3 = embedder(input=list(half))
p()

print("Expect 1 print for the letters of the 2nd 1/2 of the alphabet")
out4 = embedder(input=list(lt))
p()

print("Expect no print")
out2 = embedder(input=list(lt + lt))
p()
