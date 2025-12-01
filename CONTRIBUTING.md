# Contributing
We greatly appreciate your interest in contributing to ChildProject! We are always on the lookout for more engineers to help standardization and open source efforts in the field of linguistics.

# Setting up the Conda Virtual Environment
We recommend using miniconda to manage the environment expected of ChildProject development. You must therefore install miniconda from the anaconda website.

Once installed, you can create the environment with

```
conda env create -f [path to .env file] -p
```

The .env files for ChildProject can be found in the env folder. If you then run `conda env list` you should see the environment in your env list. You can then run `conda activate [path in env list or env name]` to enter the conda virtual environment.

NOTE: You may somewhere read instructions to automatically source/activate conda after the shell starts up. For those using Python who are used to other package managers, you must not do so, as conda does not play well with other package managers (in particular because both conda and other managers tend to alter the shell's PATH variable. I'm expect that conda uproots many other things too). What this means is that when you're working on something else, running `conda` should print `command not found: conda`. I myself found it best to create a shortcut to activate conda on the fly.

There are of course other ways to use a virtual environment, but seeing that our repository doesn't natively support e.g., Poetry or uv, you will have to do some extra complicated bookkeeping.

## Running Tests
As always, you can run `pytest` from the root folder, or `pytest [glob pattern]`.

## Making Docs
Step into the `docs` folder, making sure to have an active virtual environment. Then run `make clean`, `make html`, `touch build/html/.nojekyll`. You can make a shortcut or live-reload using the `sphinx-autobuild` package if you want with `sphinx auto-build source build/html`. Open `docs/source/build/html/index.html`