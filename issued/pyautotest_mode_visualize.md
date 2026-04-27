Currently, runs using PYAUTO_TEST_MODE do not output
visualizaiton. For some smoke tests, they need visualiation
to pass. Can you retain the default behaviour being that
PYAUTO_TEST_MODE disables visualization, but add an over ride
env variabiel PYAUTO_TEST_MODE_VISUALIZE which if it is 1
means visualization is performed. Caan you add this
to the build/env_vars.yaml of the scripts fits_make / png_make
in autogalaxy_workspace adn autolens_workspace,and remove
them from no_run