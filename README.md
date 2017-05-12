# official code

just some random hacking re-work of official code

## install

first, I'm assuming you have `conda` installed. if not,
[go do it](https://www.continuum.io/downloads)!

Given that, simply use the provided `environment.conda` file

``` shell
cd path/to/code
conda env -f environment.conda
```

that should be it.

## using

You can run the code via

``` shell
python officialCode.py
```

or from a python shell

``` python
import officialCode
officialCode.main()
```

## configuration

All configuration has been moved to the file `config.py`. I've dropped some of
the unused stuff and reformatted some things to be more natural data structures
(changing lists of tuples to `dict`s, etc.). You can make any change you want to
that file (easy future work is to allow users to pass a different config file
for different behavior (and to cache that ish))
