I have a skill /verify_install which goes through a number of steps to check an install appears to work
recently expanded to include a load of extra python verisons.

Extend it to:

1) Also include python 3.13 (I need to install this).
2) Also include PyAutoGalaxy 3.12 and 3.13 (e.g. pip install autogalaxy)
3) For each install downloads the autogalaxy_workspace or autolens_workspace according to the docs and runs start_here.py to confirm it works.


do you think we could turn this skill also into a standalone bash sacript and command we alias and can thus run outside claude, e.g. step by step or end to end