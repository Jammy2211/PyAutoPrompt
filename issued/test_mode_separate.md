PYAUTOFIT_TEST_MODE has become a bit of a catch all term for things I dont want to run when testing,
it incldues disable some visualization and posiition resmapling in autolens as opposed to just
making the sampler run faster.

Furthermore, there is also PYAUTOARRAY_OUTPUT_MODE which outputs images to hard disk, which si use.

Can you look at all PYAUTOFIT_TEST_MODE clauses and work out which are not doing a sampler speed up.
Can you then suggest names for us to use for them all.

I also want us to move to environment variables just being PYAUTO, so PYAUTO_TEST_MODE, PYAUTO_OUTPUT_MODE,
we will also rename PYAUTO_WORKSPACE_SMALL_DATASETS to PYAUTO_SMALL_DATASETS.