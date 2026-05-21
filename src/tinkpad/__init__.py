"""tinkpad — browse / probe / copy Tinker checkpoints.

Public helpers usable from any Python script (training scripts, notebooks):

    from tinkpad import register_current_run
    register_current_run(run_id)             # name = current folder
    register_current_run(run_id, "my-exp")   # explicit name
"""
__version__ = "0.3.0"

from .helpers import register_current_run

__all__ = ["__version__", "register_current_run"]
