The dependencies of many core libraries, for example numpy, scipy, matplotlib and ultimately JAX,
are capped or tied based on the code API. Its probably been like this for a long time, and looking to update
these dependencies is a good idea, but must be balanced against source code tweaks and updates.

Can you do an assessment of whether we can udpate the version of these libraries whilst maintaining a stable
build where all github actions run and autobuild works ok. Dont just focus on the core libraries listed, but
also do an assessment of other key libraries like astropy, scikit-image, scikit-learn. Priotize simplicity if
necessary, but ultimately I think a version sweep update is long overdue.