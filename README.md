`pydpiper` is a set of Python modules that offers programmatic control over pipelines. 

It is very much under active development. The paper describing the framework can be found here (note that the internals have changed significantly over time):

http://www.ncbi.nlm.nih.gov/pubmed/25126069

We kindly ask you to reference this paper when using the code.
For instructions on installing Pydpiper and optionally configuring it for your HPC environment, see the INSTALL file.

Application modules that utilize the pipeline class definitions are currently in applications folder. These applications may be moved to a separate repository at a later date.

`pydpiper` supports config files (lowest precedence), environment variables, and command-line flags (highest precedence) in a mostly uniform way via the [ConfigArgParse](https://pypi.python.org/pypi/ConfigArgParse) module.
For examples, see the `config` directory; note that unlike the command line, values in key-value pairs must not be quoted (e.g., `--queue-type=sge`, not `--queue-type='sge'`).
The config file should also be accessible to any remote machines.
You can specify a default configuration file location (e.g., for a site-wide default) with the (otherwise undocumented) environment variable `PYDPIPER_CONFIG_FILE`.
Similarly, you can place modified command templates for external tools (see pydpiper/templates) in directories specified by the environment variable PYDPIPER_TEMPLATE_PATH (as usual, a ':'-separated list of directories).
Also note that if `OMP_NUM_THREADS` is not set, we set it to 4 to reduce memory usage by Numpy/OpenBLAS, which could in principle affect some tool; you can still set this yourself (including to an empty value) to change this behaviour.

Pydpiper currently has three backends: local execution (not recommended), a default server-executor engine using qbatch for cluster submission and Pyro5 for RPC (the default), and a newer, more versatile, cluster backend using Makeflow; see the INSTALL file for details.  When using Pyro5-based backends, you can also use environment variables to override our configuration defaults for the underlying Pyro library, except for
`$PYRO_SERVERTYPE` and `$PYRO_LOGFILE`; in particular, you may wish to change `$PYRO_LOGLEVEL`, since this also controls
the verbosity of some of the application's own logging.  See [the Pyro5 documentation](https://pyro5.readthedocs.io) for more options.  When using the Makeflow backend, you may pass arbitrary options to Makeflow via --makeflow-opts=....

You may customize external commands (even changing the programs used entirely) via placing edited versions of the command templates (using Jinja2 syntax) in pydpiper/templates into a location on PYDPIPER_TEMPLATE_PATH.  Beware that this can break the pipeline in various ways e.g. generating nonlinear transforms in a linear registration step will crash the pipeline, while creating affine transforms with nontrivial Jacobian in the rigid registration steps will silently produce incorrect results.

Pydpiper supports containerised execution of external commands via the --use-singularity, --container-path=..., and --container-args=... flags.

*** *** ***
When your pipeline is running using a Pyro5-based backend, you can verify the state of your pipeline using the following tool (as of version 1.8):

check_pipeline_status.py uri

*** *** ***
Run a somewhat comprehensive test of the software:

https://wiki.mouseimaging.ca/display/MICePub/Pydpiper#Pydpiper-Pydpipertesting-MBMandMAGeT
