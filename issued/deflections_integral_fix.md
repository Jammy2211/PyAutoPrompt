The followng PR Was meant to move all deflections_via_integral methods out of the source code
and into the @autolens_workspace_test/scripts/mass_via_integral folder:

https://github.com/PyAutoLabs/PyAutoGalaxy/pull/324

The expectation is it would move the functions themselves, including the full calculation, into this folder
so the tests there could use it.

However, it did not move the functions and their calculations themselves from the source code, meaning these tests did not
pass. Can you dig up the PR, look at its history, get the alculations and move them to this test package?