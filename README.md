# Token-Log-Hatch-Params

## Requirements

* NodeJS 12.x.x
* Python 3.8

## Installing Dependencies

```
pip3 install -r requirements.txt
jupyter labextension install @pyviz/jupyterlab_pyviz
```

## Installing TECH
```
pip install -e ./
```

## Packaging
Following the guidelines here: https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/creation.html#arranging-your-file-and-directory-structure

```
TECH/
    CHANGES.txt
    docs/
    LICENSE
    README.md
    setup.py
    tech/
        __init__.py
        tech.py
        utils.py
        test/
            __init__.py
            test_tech.py
            test_utils.py
    app/
        __init__.py
        app.py
        example_app.py
    proposals/
        ygg.ipynb
        vitor.ipynb
    examples/
        test_tech.ipynb
        dress_rehearsal.ipynb
```