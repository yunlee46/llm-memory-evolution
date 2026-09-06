# Process
1. Restate the task to yourself as a list of concrete requirements.
2. Decide the data model: which columns you will use, how they are cleaned, and what is known at prediction time.
3. Decide the feature construction so that every feature for a row uses only earlier information.
4. Write the code top to bottom: imports, helpers, feature builder, then a single entry point that fits on the training rows and predicts the others.
5. Re-read the requirements list and confirm each one is implemented, including output shape, range and determinism.
6. Emit the deliverable in the requested format only.
