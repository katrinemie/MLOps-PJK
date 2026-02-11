Modul 2
Exercises/Tasks: 

Continue building your MLOps pipeline, including:

Create at least one other branch (e.g., development) besides the main branch in your git project.
Setup local pre-commits and add some checks (e.g., Flake, hardcoded API keys, file size)
Add unit tests to the core functionality of your project codebase 
Setup a CI/CD framework to orchestrate the "Local MLOps Workflow Example" described in MM2. You can use Jenkins, Github Actions or another tool you might prefer. However, I recommend that you stay within the AAU network and use the local resources for simplicity. 
Make your pipeline trigger based on a new commit, e.g. using Jenkins to initiate execution of unittests
Automate the building of a Docker container and push this to a registry if tests pass. Remember to tag the image with the Git commit hash.
Automate the training of a new model version
Implement lineage e.g. via MLFlow or WandB
Implement automatic evaluation of your trained model
Automatically add the trained model to a model registry e.g. MLFlow, if performance criteria are met
Deploy the model if it works as intended and log the deployment e.g. to MLFlow.
Store the model card that produced this model in e.g. MLFlow
Enable branch protection, and automatically merge new features to main (e.g. merges must require a successful complete run of your MLOps pipeline). You can handle this with Jenkins
Note that you are not expected to finish all of these tasks during today's exercise session, but continuously integrate additional features to your pipeline during the course.

Documentation

In addition to briefly discussing the relevant topics covered in this lecture and detailing how you've applied specific methods in your MLOps project (i.e., by solving the exercises above), your report must also include documentation of the following items.

D2.1: An overview of the implemented Continous ML pipeline, preferably in a flow chart including e.g. pre-commit checks, unit-tests, model testing, training.
Please also explain each step in the pipeline, and the logic behind it, and its purpose.
In relation to Lineage in MLOps, explain how much of your pipeline is tracked, recorded and logged.
D2.2: The code coverage percentage of your implemented unit tests. Document with a screenshot.
D2.3: Include summaries or screenshots of experiment tracking dashboards, key metrics, and comparisons between different runs or model versions.