# Configuration file for the Sphinx documentation builder.
from datetime import datetime

# -- Project information -----------------------------------------------------

project = """Psycop Model Training"""

author = "Martin Bernstorff"
github_user = "MartinBernstorff"
repo_url = f"https://github.com/{github_user}/{project.lower()}"

copyright = f"{datetime.now().year}, {author}"  # noqa

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinx.ext.githubpages",
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Do not execute the notebooks when building the docs (turned off by default)
nb_execution_raise_on_error = True  # Raise an exception when a cell execution fails
nb_execution_mode = "cache"  # Execute the notebooks only if the source has changed

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_title = """Psycop Model Training"""
html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"

html_show_sourcelink = True

html_context = {
    "display_github": True,  # Add 'Edit on Github' link instead of 'View page source'
    "github_user": github_user,
    "github_repo": project.lower(),
    "github_version": "main",
    "conf_py_path": "/docs/",
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}


html_theme_options = {
    # adds github footer icon with link to repo
    "footer_icons": [
        {
            "name": "GitHub",
            "url": repo_url,
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": repo_url,
    "source_branch": "main",
    "source_directory": "docs/",
    "light_logo": "icon.png",
    "dark_logo": "icon.png",
    "light_css_variables": {
        "color-brand-primary": "#ff5454",
        "color-brand-content": "#ff7575",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff8f8f",
        "color-brand-content": "#ff8f8f",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

pygments_style = "monokai"
pygments_dark_style = "monokai"
