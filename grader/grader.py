import builtins
import inspect
import logging
import sys
import time
import traceback

from termcolor import colored


class CheckFailed(Exception):  # noqa: N818
    def __init__(self, why):
        self.why = why

    def __str__(self):
        return self.why


class ContextManager:
    def __init__(self, on, off):
        self.on = on
        self.off = off

    def __enter__(self):
        self.on()

    def __exit__(self, exc_type, exc_value, traceback):
        self.off()


def list_all_kwargs(**kwargs):
    all_args = [{}]
    for k, v in kwargs.items():
        new_args = []
        for i in v:
            new_args.extend([dict({k: i}, **a) for a in all_args])
        all_args = new_args
    return all_args


def case(func, kwargs=None, score=1, extra_credit=False, timeout=1000):
    """
    Use @Case(score, extra_credit) as a decorator for member functions of a Grader
    A test case can return a value between 0 and 1 as a score
    If the test fails it should raise an assertion
    The test case may optionally return a tuple (score, message)

    Args:
        timeout (int): timeout in milliseconds
    """
    if kwargs is None:
        kwargs = {}

    def wrapper(self):
        n_passed = 0.0
        total = 0.0
        msg = ""
        error = ""

        # yields multiple for multi-case
        for a in list_all_kwargs(**kwargs):
            try:
                tick = time.time()
                v = func(self, **a)
                elapsed = time.time() - tick

                if elapsed > timeout / 1000:
                    raise TimeoutError(f"Timeout after {elapsed:.2f} s")

                if v is None:
                    v = 1
                elif isinstance(v, tuple):
                    v, msg = v
                else:
                    assert isinstance(v, float), f"case returned {v!r} which is not a float!"

                n_passed += v
            except TimeoutError as e:
                msg = str(e)
            except NotImplementedError:
                msg = "Not Implemented"
            except AssertionError as e:
                msg = e.args[0]
            except CheckFailed as e:
                msg = str(e)
            except Exception as e:
                msg = type(e).__name__
                error = traceback.format_exc()

            total += 1

        final_score = int(n_passed * score / total + 0.5)
        msg = f"{final_score} / {score} {msg}"

        return final_score, msg, error

    wrapper.score = score
    wrapper.extra_credit = extra_credit
    wrapper.__doc__ = func.__doc__
    wrapper.func = func

    return wrapper


class Case:
    def __init__(self, score=1, extra_credit=False, timeout=500):
        self.score = score
        self.extra_credit = extra_credit
        self.timeout = timeout

    def __call__(self, func):
        return case(func, score=self.score, extra_credit=self.extra_credit, timeout=self.timeout)


class MultiCase:
    def __init__(self, score=1, extra_credit=False, **kwargs):
        self.score = score
        self.extra_credit = extra_credit
        self.kwargs = kwargs

    def __call__(self, func):
        return case(func, kwargs=self.kwargs, score=self.score, extra_credit=self.extra_credit)


class Grader:
    def __init__(self, module, logger, verbose=False):
        self.module = module
        self.logger = logger
        self.verbose = verbose

    @classmethod
    def get_all_cases(cls, sort=True):
        ret = []

        for n, f in inspect.getmembers(cls):
            if not hasattr(f, "score"):
                continue

            line_num = inspect.getsourcelines(f.func)[1]
            ret.append((n, f, line_num))

        if sort:
            ret.sort(key=lambda x: x[2])

        return ret

    @classmethod
    def has_cases(cls):
        return len(cls.get_all_cases()) > 0

    @classmethod
    def total_score(cls):
        return sum(f.score for n, f, line_num in cls.get_all_cases())

    def run(self, logger):
        score = 0
        total_score = 0

        for _, f, _ in self.get_all_cases():
            s, msg, error = f(self)
            score += s

            if self.verbose:
                log = logger.info if msg == "passed" else logger.warn
                log(f"  - {f.__doc__.strip():<50} [ {msg} ]")

                if error:
                    logger.error(f"{error.rstrip()}")

            if not f.extra_credit:
                total_score += f.score

        return score, total_score


def grade(g, assignment_module, logger, verbose):
    try:
        grader = g(assignment_module, logger, verbose)
    except Exception:
        if verbose:
            logger.error(f"Your program crashed\n{traceback.format_exc()}")

        return 0, g.total_score()

    return grader.run(logger)


def grade_all(assignment_module, logger, verbose=False):
    score = 0
    total_score = 0

    def get_all_subclasses(cls):
        all_subclasses = []

        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))

        return all_subclasses

    for g in get_all_subclasses(Grader):
        if g.has_cases():
            logger.info(g.__doc__)

            s, ts = grade(g, assignment_module, logger, verbose)

            if verbose:
                logger.info(f" --------------------------------------------------    [ {s:3d} / {ts:3d} ]")
            else:
                logger.info(f" * {g.__doc__:<50}  [ {s:3d} / {ts:3d} ]")

            total_score += ts
            score += s

    logger.info(f"Total                                                    {score:3d} / {total_score:3d}")

    return score


def load_assignment(logger, assignment_path: str, pre_import_fn=None):
    import atexit
    import importlib
    import tempfile
    import zipfile
    from pathlib import Path
    from shutil import rmtree

    assignment_path = Path(assignment_path)

    if assignment_path.is_dir():
        module_path = assignment_path
        module_dir = module_path.parent
        module_name = module_path.stem

        sys.path.insert(0, str(module_dir))

        return importlib.import_module(module_path.stem)
    elif assignment_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(assignment_path) as f:
            module_dir = Path(tempfile.mkdtemp())
            atexit.register(lambda: rmtree(module_dir))
            f.extractall(module_dir)

            module_names = list(module_dir.glob("*/"))

            if len(module_names) != 1:
                logger.error(f"Malformed zip file, expected one top-level folder, got {len(module_names)}")
                return None

            module_name = module_names[0].name
    else:
        raise ValueError(f"{assignment_path} should be a directory or zip")

    sys.path.insert(0, str(module_dir))

    if pre_import_fn is not None:
        pre_import_fn()

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        logger.error(f'Import error "{e!s}"')
    except Exception as e:
        logger.error(f'Failed to load your solution: "{e!s}"')


class RuntimeFormatter(logging.Formatter):
    _COLOR = {
        "ERROR": "red",
        "WARNING": "yellow",
        "INFO": "white",
        "DEBUG": "green",
    }

    def __init__(self, *args, disable_color: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.disable_color = disable_color
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = time.time() - self.start_time
        _, remainder = divmod(elapsed_seconds, 3600)
        mins, remainder = divmod(remainder, 60)
        secs, ms = divmod(remainder, 1)

        record.elapsed = f"{int(mins):02d}:{int(secs):02d}:{int(1000 * ms):03d}"

        output_raw = super().format(record)

        if self.disable_color:
            return output_raw

        prefix = output_raw[: output_raw.find(record.getMessage())]
        output = output_raw[output_raw.find(record.getMessage()) :]
        # color full line
        output_lines = [colored(line, self._COLOR[record.levelname]) for line in output.splitlines()]
        # add prefix
        prefix_color = colored(prefix, self._COLOR[record.levelname])
        output_lines = [prefix_color + line for line in output_lines]

        return "\n".join(output_lines)


def init_loggers(log_path: str, show_debug=False, disable_color=False):
    formatter = RuntimeFormatter("[%(levelname)-8s %(elapsed)s] %(message)s", disable_color=disable_color)

    if log_path is not None:
        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w")]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]

    logger = logging.getLogger("grader")
    logger.setLevel(logging.DEBUG if show_debug else logging.INFO)

    for h in handlers:
        h.setFormatter(formatter)
        logger.addHandler(h)

    def patched_print(*args, **kwargs):
        logger.debug(" ".join(str(a) for a in args))

    builtins.print = patched_print

    return logger


def run():
    import argparse

    parser = argparse.ArgumentParser("Grade your assignment")
    parser.add_argument("assignment", default="homework")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-vv", "--very_verbose", action="store_true")
    parser.add_argument("--log_path", default=None)
    parser.add_argument("--disable_color", action="store_true", default=False)

    args = parser.parse_args()

    logger = init_loggers(args.log_path, show_debug=args.very_verbose, disable_color=args.disable_color)

    print("Loading assignment")
    assignment = load_assignment(logger, args.assignment)
    if assignment is None:
        return 0

    print("Loading grader")
    total_score = grade_all(assignment, logger, args.verbose or args.very_verbose)

    return total_score
