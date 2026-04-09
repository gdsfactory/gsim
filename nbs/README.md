# Building Docs from Notebooks

Notebooks are committed **with outputs**. CI only converts and deploys — it does not re-execute them.

## Run a notebook for docs

```bash
just nbrun-docs nbs/my_notebook.ipynb
```

This executes the notebook, strips absolute paths from outputs, and converts it to markdown. Plotly renders as
self-contained HTML and PyVista renders as static images.

You can run multiple at once:

```bash
just nbrun-docs nbs/meep_ybranch.ipynb nbs/palace_cpw.ipynb
```

## Preview docs locally

```bash
just serve    # http://localhost:8080/gsim/
```

## Add a new notebook

1. Create `nbs/my_notebook.ipynb` and develop it normally
1. Run it for docs: `just nbrun-docs nbs/my_notebook.ipynb`
1. Add an entry in `mkdocs.yml` under `nav` pointing to `nbs/my_notebook.md`
1. Commit the executed notebook

## Update an existing notebook

1. Edit and run the notebook normally in Jupyter
1. Re-run for docs: `just nbrun-docs nbs/my_notebook.ipynb`
1. Commit
