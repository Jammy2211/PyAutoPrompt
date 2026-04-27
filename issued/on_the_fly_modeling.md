When I perform sampling, I am able to produce on-the-fly images and output, which currently overwrites
the method:

    def perform_quick_update(self, paths, instance):
        raise NotImplementedError

In @PyAutoFit/autofit/non_linear/analysis.py

An example of how this is used in autolens to show lens modeling is:

    def perform_quick_update(self, paths, instance):
        """
        Perform a quick visualization update during non-linear search fitting.

        This method is called intermittently while the sampler is running to produce
        the `subplot+fit` plots of the current maximum-likelihood model fit. The intent
        is to provide fast feedback (without waiting for the full run to complete) so that
        users can monitor whether the fit is behaving sensibly.

        The plot appears both in a matplotlib window (if running locally) and is also saved to the
        `output` folder of the output path.

        Parameters
        ----------
        paths : af.DirectoryPaths
            Object describing the output folder structure where visualization files
            should be written.
        instance : model instance
            The current maximum-likelihood instance of the model, used to generate
            the visualization plots.
        """

        self.Visualizer().visualize(
            analysis=self,
            paths=paths,
            instance=instance,
            during_analysis=True,
            quick_update=True,
        )

Which is in @PyAutoGalaxy/autogalaxy/analysis

Where the visualizer object has been adapted to be reused but only call a subset of key functions and outputs for a quick update.

This works ok, but it is stop-start whereby a sampler has to stop to perform visualization. For certian users I have
heard matplotlib can cause display issues in Jupyter Notebooks.

My questions are, can this functionality be updated to output the viuslaizaiton on a separate process while sampling
contiues? Even with JAX and GPU? And can I put any safety checks in or improve it to perform more generally on Notebooks.
