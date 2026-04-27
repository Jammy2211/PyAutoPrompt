PYAUTOFIT_TEST_MODE is used to make integration scripts run faster, and it does the following:

- Makes the sampler use only 50 samples if its Nautilus, or for other samples some other value that reduces run time.
- Disables all visualization so images are not output for speed up.
- Disables some other features to speed things up.

The problem is that running Nautilus with PYAUTOFIT_TEST_MODE=1 still takes some time and
means integration testing, especially smoke tests, still has some time to run.

In test mode, can you run the file @autolens_workspace_test/scripts/imaging/model_fit.py and
do an assessment if whether there is a way we can have test mode run but in a way which does
not require Nautilus itself to be run. The truth is we just want the output folder state to be output
as if its run so that we can use test mode for testing.

One part of behaviour we do want is currently test mode does call the likelihood function, and thus it does
make sure that the likelihood function can be called correct in these scripts. This will be where some run time
is lost, but it is important to make sure the likelihood function can be called correctly in test mode. 

Can you investigate the behaviour for two cases, where the likelihood function is called when you try to bypass
and nautilus and one where its not called at all. Compare their runt imes, we may make it so test mode
can do either and we customize tests to use one or the other depending on the test.