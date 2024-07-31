import dill
from typing import Callable, List, Union, Optional
from functools import wraps
from pathlib import Path, PosixPath

from beartype import beartype
import joblib

__VERSION__: str = "0.1.1"

@beartype
def IteratorCacher(
    cache_location: Union[str, PosixPath],
    unpacking_func: Callable,
    repacking_func: Callable,
    iter_list: Optional[List[str]] = None,
    verbose: bool = False,
    ) -> Callable:

    @beartype
    def meta_wrapper(func: Callable) -> Callable:

        func_hash = joblib.hash(dill.dumps(func))

        dir = Path(cache_location) / str(func) / func_hash
        dir.parent.mkdir(exist_ok=True, parents=True)
        mem = joblib.Memory(dir, verbose=False)

        @wraps(func)
        @beartype
        @mem.cache(ignore=["cacher_code"])
        def memory_handler(
            cacher_code: Optional[bool],
            func_hash: str,
            *args,
            **kwargs,
            ) -> List[bool]:
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
            assert not args, f"Non keyword args are not supported but received {args}"
            if cacher_code is not None and cacher_code is not False:
                return cacher_code
            assert cacher_code is not False, "cached result was about to be recomputed"
            assert cacher_code is None, "Was about to compute a value even though cacher_code is not None"

            out = func(**kwargs)
            assert hasattr(unpacking_func(out), "__iter__"), "The computed value must be an iterable!"

            return out

        @wraps(func)
        def wrapper(
            iter_list: List[str] = iter_list,
            *args,
            **kwargs,
            ) -> Callable:
            assert not args, (
                f"Only keyword arguments are supported for the IteratorCacher decorator, received {args}"
            )
            if iter_list is None:
                iter_list = []
                for k, v in kwargs.items():
                    if hasattr(v, "__iter__"):
                        iter_list.append(k)

            assert iter_list, "no argument iter_list given an not iterable in kwargs found"
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
            dones = []
            todos = []
            [
                dones.append(item) if states[it] else todos.append(item)
                for it, item in enumerate(all_kwargs)
            ]

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

        wrapper.memory_handler = memory_handler
        wrapper.check_call_in_cache = memory_handler.check_call_in_cache
        wrapper.func = func
        wrapper.ignore = memory_handler.ignore

        return wrapper
    return meta_wrapper

IteratorCacher.__VERSION__ = __VERSION__
