The workspace scripts are used as integration tests and smoke tests, with
good examples being @autolens_workspace/scripts/imaging/modeling.py
and @autolens_workspace/scripts/interferometer/modeling.py.
PYAUTOFIT_TEST_MODE=2
These run with PYAUTOFIT_TEST_MODE=2, which skips sampling but still performd
a likelihood evaluation to test the model.

However, we would benefit from these smoke tests running a lot faster as they
are a core part of ensuring the software from top to bottom is working. WE ultimately
want all scritps, if possible, to run really fast.

Can you run the abvoe two modeling.py scripts using PYAUTOFIT_TEST_MODE=2 and give a break
down of where the run time is? We can then work towards making them run a lot faster
and thus having quick smoke tests and integration tests.