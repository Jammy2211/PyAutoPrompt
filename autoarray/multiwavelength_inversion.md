Can create a list of InversionMatrix objects for each dataset + tracer and then have a new class which takes a list of
these objects and creates a combined curvature_reg matrix from them, which solves for all datasets simultaneously.

This is genuine because it can reuse all existing functionality and doesn't require diting of the inner code
to get multi dataset fits working.

The regularization matrix could addition addition multi-data regularization terms added.

Multi analysis API can easily be used for model parameterization, with CombineAnalysis overwritten to define
the log likelihood fuinction (which maybe uses a special type of Fit object).

[This was not written as a claud Prompt, expan dfully the day I use it]