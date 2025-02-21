from . import tests  # noqa

try:
    from .safe_grader import run

    print("Testing grader loaded.")
except ImportError:
    from .grader import run

    print("Val grader loaded.")

run()
