# How to contribute

Thank you for contributing xarray-dataclasses!
If you have some ideas to propose, please follow the contribution guide.
We use [GitHub flow][github-flow] for developing and managing the project.
The first section describes how to contribute with it.
The second and third sections explain how to prepare a local development environment and our automated workflows in GitHub Actions, respectively.


## Contributing with GitHub flow

### Create a branch

First of all, [create an issue][issues] with a simple title and get an issue ID (e.g., `#24`).
For example, if you propose to add functions for plotting something, the title of the issue would be `Add dataset module`.
Using a simple verb (e.g., add, update, remove, fix, ...) in the present tense is preferable.

Then fork the repository to your account and create **a branch whose name begins with the issue ID** (e.g., `#24-dataset-module` or simply `#24`).
We do not care about the naming of it (except for the issue ID) because the branch will be deleted after merging with the main branch.

### Add commits

After you update something, commit your change with **a message which starts with the issue ID**.
Using a simple verb in the present tense is preferable.

```shell
git commit -m "#24 Add dataset module"
```

Please make sure that your code (1) is formatted by [Black][black], (2) is checked by [Flake8][flake8], (3) passes the tests (`tests/test_*.py`) run by [pytest][pytest], and (4) passes the static type check run by [Pyright][pyright].
They are necessary to pass the status checks when you create a pull request (see also [the section of GitHub Actions](#github-actions)).

If you add a new feature, please also make sure that you prepare tests for it.
For example, if you add the plotting module (`xarray_dataclasses/dataset.py`), write the series of test functions in `tests/test_dataset.py`.

If you write a Python docstring, follow [the Google style][napoleon-google] so that it is automatically converted to a part of API docs by [Sphinx][sphinx].

### Open a Pull Request

When your code is ready, [create a pull request (PR)][pull-requests] to merge with the main branch.
Without special reasons, the title should be the same as that of the issue.
Please specify the issue ID in the comment form so that it is linked to the PR.
For example, writing `This PR closes #24.` at the beginning of the comment would be nice.

### Discuss and review your code

Your code is reviewed by at least one contributor and checked by the automatic status checks by [GitHub Actions][github-actions].
After passing them, your code will be merged with the main branch.
That's it!
Thank you for your contribution!

## Development environment

We manage the development environment (i.e., Python and JavaScript and their dependencies) with [Poetry][poetry] and [Node.js][nodejs].
After cloning the repository you forked, you can setup the environment by the following command.

```shell
poetry install
npm install
```

## GitHub Actions

### Testing, linting, and formatting

We have [a test workflow][test-workflow] for testing, static type checking, linting, and formatting the code.
It is used for status checks when a pull request is created.
If you would like to check them in local, the following commands are almost equivalent (the difference is that the workflow is run under multiple Python versions).

```shell
poetry run pytest docs tests xarray_dataclasses
poetry run flake8 docs tests xarray_dataclasses
poetry run black --check docs tests xarray_dataclasses
npm run pyright docs tests xarray_dataclasses
```

### Publish to PyPI

We have [a PyPI workflow][pypi-workflow] for publishing the package to [PyPI][pypi].
When [a release is created][release], the workflow is triggered and the package is automatically built and uploaded to PyPI.

### Deploy docs

We have [a GitHub Pages workflow][gh-pages-workflow] for publishing the HTML docs.
When [a release is created][release], the workflow is triggered and the docs are automatically built and deployed to [the gh-pages branch][gh-pages-branch].


[black]: https://black.readthedocs.io/en/stable/
[direnv]: https://direnv.net/
[flake8]: https://flake8.pycqa.org/en/latest/
[gh-pages-workflow]: https://github.com/astropenguin/xarray-dataclasses/blob/main/.github/workflows/gh-pages.yml
[gh-pages-branch]: https://github.com/astropenguin/xarray-dataclasses/tree/gh-pages
[github-actions]: https://github.com/astropenguin/xarray-dataclasses/actions
[github-flow]: https://guides.github.com/introduction/flow/
[issues]: https://github.com/astropenguin/xarray-dataclasses/issues?q=is%3Aissue
[napoleon-google]: https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google
[nodejs]: https://nodejs.org/
[poetry]: https://python-poetry.org/
[pull-requests]: https://github.com/astropenguin/xarray-dataclasses/pulls?q=is%3Apr
[pypi]: https://pypi.org/project/xarray-dataclasses/
[pypi-workflow]: https://github.com/astropenguin/xarray-dataclasses/blob/main/.github/workflows/pypi.yml
[pyright]: https://github.com/microsoft/pyright
[pytest]: https://docs.pytest.org/en/stable/
[release]: https://github.com/astropenguin/xarray-dataclasses/releases
[sphinx]: https://www.sphinx-doc.org/en/master/
[test-workflow]: https://github.com/astropenguin/xarray-dataclasses/blob/main/.github/workflows/tests.yml
[vs-code]: https://code.visualstudio.com/
