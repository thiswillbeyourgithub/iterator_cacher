import dill
from typing import Callable, List
from functools import wraps
from pathlib import Path

from typeguard import typechecked
import joblib


@typechecked
def IteratorCacher(
    cache_location: str,
    iter_list: List[str],
    unpacking_func: Callable,
    repacking_func: Callable,
    verbose: bool = False,
    ) -> Callable:

    def meta_wrapper(func: Callable) -> Callable:

        func_hash = joblib.hash(dill.dumps(func))

        dir = Path(cache_location) / str(func) / func_hash
        mem = joblib.Memory(dir, verbose=False)

        @wraps(func)
        @mem.cache(ignore=["cacher_code"])
        def memory_handler(
            cacher_code,
            func_hash,
            *args,
            **kwargs,
            ):
            """
            wrapper around func.
            func_hash is not used internally but allows to distinguish functions.
            cacher_code can have several values:
                if False:
                    The result is already cached or we're just checking if it is, so should crash if actually running
                if None:
                    No cached value, please compute
                else:
                    is the value that has to be recomputed

            """
            assert not args, f"Non keyword args are not supported"
            if cacher_code is not None and cacher_code is not False:
                return cacher_code
            assert cacher_code is not False, f"cached result was about to be recomputed"
            assert cacher_code is None, f"Was about to compute a value even though cacher_code is not None"

            out = func(**kwargs)
            assert hasattr(unpacking_func(out), "__iter__"), "The computed value must be an iterable!"

            return out

        @wraps(func)
        def wrapper(*args, **kwargs) -> Callable:
            assert not args, (
                "Only keyword arguments are supported for the IteratorCacher decorator"
            )

            for il in iter_list:
                assert il in kwargs, f"Iterator {il} is not present in kwargs"
                assert hasattr(kwargs[il], "__iter__"), f"Object at key {il} is not an iterator"
                assert il != "cacher_code", "The name 'cacher_code' is used internaly in the decorator"

            assert len(set([len(kwargs[il]) for il in iter_list])) == 1, "Not all iterators have the same length"

            n_items = len(kwargs[iter_list[0]])

            # create each arg
            all_kwargs = []
            for i in range(n_items):
                default = kwargs.copy()
                for il in iter_list:
                    default[il] = kwargs[il][i]
                all_kwargs.append(default)

            # check which items are already in cache
            states = [
                memory_handler.check_call_in_cache(
                    cacher_code=False,
                    func_hash=func_hash,
                    **item,
                )
                for item in all_kwargs
            ]
            dones = [item for it, item in enumerate(all_kwargs) if states[it]]
            todos = [item for it, item in enumerate(all_kwargs) if not states[it]]

            if verbose:
                print(f"Number of cached values: '{len(dones)}'")
                print(f"Number of values to compute: '{len(todos)}'")

            if todos:
                # argument to use to compute all missing values
                todo_kwargs = kwargs.copy()
                for il in iter_list:
                    todo_kwargs[il] = (
                            [
                                item[il]
                                for item in todos
                            ]
                    )

                # compute missing values
                new_values = memory_handler(
                    cacher_code=None,
                    func_hash=func_hash,
                    **todo_kwargs,
                )

                # upack the output into an iterable
                new_parsed = unpacking_func(new_values)

                assert len(new_parsed) == len(todos)

            # add values to cache and reconstruct full output at the same time
            result_list = []
            for i in range(n_items):
                item = all_kwargs[i]
                if states[i]:
                    # was already cached, fetch the value and store it
                    val = memory_handler(
                        cacher_code=False,
                        func_hash=func_hash,
                        **item,
                    )
                else:
                    # was computed by batch, we need to store it now
                    val_to_cache = repacking_func([new_parsed[todos.index(item)]])
                    val = memory_handler(
                        cacher_code=val_to_cache,
                        func_hash=func_hash,
                        **item,
                    )
                    # sanity check
                    assert val is val_to_cache
                    # for good measure: retrieval test
                    check = memory_handler(
                        cacher_code=None,
                        func_hash=func_hash,
                        **item,
                    )

                    try:
                        assert val == check 
                    except AssertionError:
                        raise
                    except Exception:
                        assert type(val) == type(check)

                result_list.append(val)

            # assemble the result as per the user liking
            out = repacking_func(result_list)

            # final checks
            assert len(result_list) == n_items
            assert all(type(v) == type(out) for v in result_list)

            return out

        wrapper.wrapped_func = memory_handler

        return wrapper
    return meta_wrapper

IteratorCacher.__VERSION__ = "0.0.1"

if __name__ == "__main__":
    # for testing and demonstration purposes

    from string import ascii_letters
    import numpy as np
    import code


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
        input()

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

    breakpoint()

