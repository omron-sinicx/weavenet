# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import inspect
import subprocess

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'WeaveNet'
copyright = '2023, OMRON SINIC X Corp.'
author = 'Atsushi Hashimoto'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import sys, os

sys.path.append(os.path.abspath('../src/'))

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon','sphinx_multiversion','sphinx.ext.linkcode']
smv_tag_whitelist = r'^v\d+\.\d+\.\d+$'            # Include tags like "v2.1.023"
smv_branch_whitelist = r'^.*$'
smv_remote_whitelist = None                   # Only use local branches
smv_released_pattern = r'^tags/.*$'
smv_outputdir_format = '{ref.name}'


# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"
autodoc_default_options = {
#    'members': 'var1, var2',
    'member-order': 'bysource',
#    'special-members': '__init__',
    'undoc-members': False,    
    'exclude-members': 'training',#__weakref__'    
}

# Don't show class signature with the class' name.
#autodoc_class_signature = "separated"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sizzle'
#html_theme = 'sphinx_rtd_theme'

#extensions.append("sphinxjp.themes.basicstrap")
#html_theme = 'basicstrap'
#html_theme = 'groundwork'
html_theme = 'renku'
#html_theme = 'python_docs_theme'

html_static_path = ['static']


# a better implementation with version: https://gist.github.com/nlgranger/55ff2e7ff10c280731348a16d569cb73


linkcode_revision = "v1.0.0"
try:
    # lock to commit number
    cmd = "git log -n1 --pretty=%H"
    head = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    linkcode_revision = head

    # if we are on master's HEAD, use master as reference
    cmd = "git log --first-parent {} -n1 --pretty=%H".format(linkcode_revision)
    master = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    if head == master:
        linkcode_revision = "v1.0.0"

    # if we have a tag, use tag as reference
    cmd = "git describe --exact-match --tags " + head
    tag = subprocess.check_output(cmd.split(" ")).strip().decode('utf-8')
    linkcode_revision = tag

except subprocess.CalledProcessError:
    pass

linkcode_url = "https://github.com/omron-sinicx/weavenet/tree/" \
               + linkcode_revision + "/{filepath}#L{linestart}-L{linestop}"

def linkcode_resolve(domain, info):
    
    if domain != 'py' or not info['module']:
        return None

    modname = info['module']
    topmodulename = modname.split('.')[0]
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None
    '''
    try:
        modpath = pkg_resources.require(topmodulename)[0].location
        filepath = os.path.relpath(inspect.getsourcefile(obj), modpath)
        if filepath is None:
            return None
    except Exception:
        return None
     '''
    filepath = os.path.relpath(inspect.getsourcefile(obj), './')

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1
        
    return linkcode_url.format(
        filepath=filepath, linestart=linestart, linestop=linestop)
