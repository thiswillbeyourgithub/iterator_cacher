from typing import Callable, List, Union, Optional, Any
from functools import wraps
from pathlib import Path, PosixPath
import inspect

from beartype import beartype
import joblib

__VERSION__: str = "1.0.2"

class CachingCodes:
    "Used to tell the memory_handler what to do"
    pass
class CrashIfNotCached(CachingCodes):
    "The value is assumed cached so if this is detected inside the memory handler it will crash"
    pass
class DoComputeValue(CachingCodes):
    "Ask to compute the value using the func"
    pass
class ReturnThisValue(CachingCodes):
    "Return the value directly, in effect storing the value in the cache"
    def __init__(self, value):
        self.value = value

def memory_handler_(
    cacher_code: Union[CachingCodes, Any],
    func_hash: str,
    user_func: Callable,
    kwargs: Any,
    ) -> List[Union[bool, Any]]:
    """
    Sort of like a wrapper around func.
    The idea is that this function is cached using joblib, but joblib is
    asked to ignore the argument cacher_code, so we can compute many values
    at once then cache each individual value by passing it as cacher_code and
    returning instantly.

    Arguments:
    ----------
    func_hash is not used internally but allows to distinguish functions.

    user_func: Callable
        used if needed to actually compute something

    cacher_code can have several values:
        if of class CrashIfNotCached:
            The result is assumed to be already cached, so should crash if the function is actually called instead of just checking the cache.
        elif of class DoComputeValue:
            Do compute the value
        elif of class ReturnThisValue:
            the value that has to be stored in the cache is in the .value attribute
    Returns:
    --------
    the value, either computed or from the cache

    """
    if isinstance(cacher_code, CrashIfNotCached):
        raise Exception(f"CrashIfNotCached: for kwargs '{kwargs}'")
    elif isinstance(cacher_code, ReturnThisValue):
        return cacher_code.value
    else:
        assert isinstance(cacher_code, DoComputeValue), f"Expected cacher_code DoComputeValue, not {cacher_code}"

    out = user_func(**kwargs)

    return out

@beartype
def IteratorCacher(
    memory_object: joblib.Memory,
    res_to_list: Callable,
    combine_res: Optional[Callable] = None,
    iter_list: Optional[List[str]] = None,
    batch_size: int = 500,
    verbose: bool = False,
    debug: bool = False,
    ) -> Callable:
    """
    Note that args are not supported, you have to send each argument as a
    keyword argument.

    Parameters
    ----------
    memory_object: a joblib.Memory object
        it has to be given by the user otherwise the cache gets seemingly
        recomputed each time python is run.

    res_to_list: Callable
        a callable (ideally a lambda function) that takes the output
        of func and turns it into a list. For example if your function is
        turning strings into numpy arrays, res_to_list could be "lambda ar: ar.squeeze().to_list()"

    combine_res: Callable, optional
        a callable (ideally a lambda function) that
        takes in a list made of outputs of res_to_list, appended one after the
        other and turns it into a single variable. For example in the same
        example this could be "lambda l: np.array([np.array(li) for li in l])"
        If not given, you receive a single list containing the outputs after
        res_to_list was called on each output.

    iter_list: List[str], optional
        list of strings. For example if your function call
        will have kwargs '{"embedding_model": myname, "text_list": mylist}
        then iter_list should be ["text_list"]. It is used by IteratorCacher
        to determine which list among the args should be used to call the
        func by batch. iter_list can contain multiple names but
        then each of those iterables must contain the same number of elements.
        It was tested with lists but might work with other iterables.
        If iter_list is not given, we assume all iterables of the kwargs are
        implied, but this will fail if they don't have have the same length.

    batch_size: int, default 500
        if calling with many input arguments, call the func by batch of this size

    verbose: bool

    debug: bool, default False
        implies verbose=True
        Does many sanity check, which makes the whole thing slower

    """
    if debug:
        verbose = True

    def p(message: str) -> None:
        "print if verbose is set"
        if verbose:
            print(message)

    @beartype
    def meta_wrapper(func: Callable) -> Callable:
        # hash the functions and parameters to distinguish functions
        to_hash = []
        try:
            to_hash.append(joblib.hash(func))
        except Exception as err:
            raise Exception(f"Failed to hash func: '{err}'")
        to_hash.append(joblib.hash(iter_list))
        for userfunc in [res_to_list, combine_res]:
            try:
                to_hash.append(joblib.hash(userfunc))
            except Exception:
                if "lambda" in str(userfunc):
                    stringfunc = inspect.getsource(userfunc).strip().split("=", 1)[1].split("lambda ", 1)[1].strip()
                    while stringfunc.endswith(",") or stringfunc.endswith(")"):
                        stringfunc = stringfunc[:-1]
                    to_hash.append(joblib.hash(stringfunc))
                else:
                    raise
        func_hash = joblib.hash(to_hash)
        p(f"Function hash: {func_hash}")

        try:
            p(f"memory_handler_ hash: {joblib.hash(memory_handler_)}")
        except Exception as err:
            p(f"Couldn't hash memory_handler_: '{err}'")
        memory_handler = memory_object.cache(
            ignore=["cacher_code", "user_func"],
        )(memory_handler_)

        def wrapper(
            iter_list: List[str] = iter_list,
            *args,
            **kwargs,
            ) -> Callable:
            "actual wrapper for the function of the user"
            assert not args, (
                f"Only keyword arguments are supported for the IteratorCacher decorator, received {args}"
            )
            if iter_list is None:
                iter_list = []
                for k, v in kwargs.items():
                    if hasattr(v, "__iter__"):
                        iter_list.append(k)
            p(f"Will use iter_list: {iter_list}")

            assert iter_list, "no argument iter_list given an not iterable in kwargs found"
            for il in iter_list:
                assert il in kwargs, f"Iterator {il} is not present in kwargs"
                assert hasattr(kwargs[il], "__iter__"), f"Object at key {il} is not an iterator"
                assert il != "cacher_code", "The name 'cacher_code' is used internaly in the decorator"

            assert len(set([len(kwargs[il]) for il in iter_list])) == 1, "Not all iterators have the same length"

            n_items = len(kwargs[iter_list[0]])
            p(f"Number of items in iter_list received: {n_items}")

            # create each arg
            all_kwargs = []
            for i in range(n_items):
                default = kwargs.copy()
                for il in iter_list:
                    default[il] = [kwargs[il][i]]
                all_kwargs.append(default)

            # check which items are already in cache
            states = [
                memory_handler.check_call_in_cache(
                    cacher_code=None,
                    func_hash=func_hash,
                    user_func=func,
                    kwargs=item,
                )
                for item in all_kwargs
            ]

            # sanity check
            if debug:
                for ist, sta in enumerate(states):
                    if sta is True:
                        memory_handler(
                            cacher_code=CrashIfNotCached(),
                            func_hash=func_hash,
                            user_func=func,
                            kwargs=all_kwargs[ist],
                        ) is sta
                    elif sta is False:
                        assert memory_handler.check_call_in_cache(
                            cacher_code=DoComputeValue(),
                            func_hash=func_hash,
                            user_func=func,
                            kwargs=all_kwargs[ist],
                        ) is sta
                        assert memory_handler.check_call_in_cache(
                            cacher_code="whatever",
                            func_hash=func_hash,
                            user_func="something",
                            kwargs=all_kwargs[ist],
                        ) is sta
                        try:
                            failed = None
                            memory_handler(
                                cacher_code=CrashIfNotCached(),
                                func_hash=func_hash,
                                user_func=func,
                                kwargs=all_kwargs[ist],
                            )
                            failed = False
                        except Exception:
                            failed = True
                        assert failed, "Sanity check failed"
                    else:
                        raise ValueError(sta)

            # find out which value are cached and which value have to be computed
            dones = []
            todos = []
            [
                dones.append(item) if states[it] else todos.append(item)
                for it, item in enumerate(all_kwargs)
            ]

            p(f"Number of already cached values: '{len(dones)}'")
            p(f"Number of values to compute: '{len(todos)}'")

            if todos:
                # aggregate the arguments that are shared for all calls
                # with arguments from iter_list (i.e. that are batch specific)
                todo_kwargs = kwargs.copy()
                for il in iter_list:
                    assert il in todo_kwargs
                    todo_kwargs[il] = []
                batches = [todo_kwargs.copy()]
                for item in todos:
                    for il in iter_list:
                        if len(batches[-1][il]) >= batch_size:
                            batches.append(todo_kwargs.copy())
                            batches[-1][il] = item[il]
                        else:
                            batches[-1][il].extend(item[il])
                        assert len(item[il]) == 1
                if len(todos) > 1 and batch_size > 1:
                    assert len(batches[0][il]) > 1

                p(f"Number of batches: {len(batches)}")
                p(f"Sample of arguments before actual call: {str(batches[0])[:1000]}")

                # compute missing values for each batch, turn the result
                # into a list, then aggregate all those lists into one
                new_parsed = []
                for ib, b in enumerate(batches):
                    p(f"Number in batch: {len(b[il])}")
                    p(f"Hash of batch #{ib}: {joblib.hash(b)}\nvalue: {str(b)[:100]}")
                    assert all(il in b for il in iter_list)
                    new_values = memory_handler(
                        cacher_code=DoComputeValue(),
                        func_hash=func_hash,
                        user_func=func,
                        kwargs=b,
                    )
                    if debug:
                        memory_handler(
                            cacher_code=CrashIfNotCached(),
                            func_hash=func_hash,
                            user_func=func,
                            kwargs=b,
                        ) is new_values

                    # upack the output into an iterable
                    temp_parsed = res_to_list(new_values)
                    p(f"Sample after unpacking: {str(temp_parsed)[:1000]}")
                    assert isinstance(temp_parsed, list), "The computed value must be produce a list!"
                    assert len(temp_parsed) == len(b[il]), f"Unexpected list values: parsed: {len(temp_parsed)} ; in arg: {len(b[il])}"
                    assert isinstance(temp_parsed, list), f"Expected type list after unpacking but got {type(temp_parsed)}"
                    new_parsed.extend(temp_parsed)

                assert len(new_parsed) == len(todos)

            # store the value of each iteration of the iter_lists in the cache
            # and reconstruct the final list with all the values in the right order
            result_list = []
            for i in range(n_items):
                item = all_kwargs[i]
                if states[i]:
                    # was already cached, fetch the value and store it
                    assert all(il in item for il in iter_list)
                    val = memory_handler(
                        cacher_code=CrashIfNotCached(),
                        func_hash=func_hash,
                        user_func=func,
                        kwargs=item,
                    )

                    # sanity check
                    if debug:
                        test = memory_handler(
                            cacher_code=CrashIfNotCached(),
                            func_hash=func_hash,
                            user_func=func,
                            kwargs=item,
                        )
                        assert test is val or test == val or joblib.hash(test) == joblib.hash(val)
                else:
                    # was computed in a batch, we need to store it now
                    val = new_parsed[todos.index(item)]
                    val2 = memory_handler(
                        cacher_code=ReturnThisValue(value=val),
                        func_hash=func_hash,
                        user_func=func,
                        kwargs=item,
                    )
                    assert val2 is val or val2 == val or joblib.hash(val2) == joblib.hash(val)

                result_list.append(val)

            # final checks
            assert len(result_list) == n_items

            # assemble the result as per the user liking
            if combine_res is not None:
                out = combine_res(result_list)
                return out
            else:
                return result_list


        wrapper = wraps(func)(wrapper)
        wrapper.iterator_cacher_memory_handler = memory_handler
        wrapper.user_func = func
        wraps.iterator_cacher_memory = memory_object

        return wrapper
    return meta_wrapper

IteratorCacher.__VERSION__ = __VERSION__
