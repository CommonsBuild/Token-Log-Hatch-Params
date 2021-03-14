# TEC Hatch Params Dashboard

This dashboard is a tool to collect community opinions on the proper parameterization of the Token Engineering Commons.

## Requirements

- NodeJS 12
- Python 3.8

## Installing Dependencies

```
pip3 install -r requirements.txt
pip install -e ./
jupyter labextension install @pyviz/jupyterlab_pyviz
```

## Running Locally

```
panel serve --show apps/hatch.py [--dev tech/* apps/*]
```

## Resources

- [The Hatch Overview](https://forum.tecommons.org/t/the-hatch-tl-dr/272)
- [Understanding each Hatch Parameter](https://forum.tecommons.org/t/tec-test-hatch-implementation-specification/226)
- [About the Token Engineering Commons](https://tecommons.org/)
