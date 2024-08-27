# BAND SDK v0.2 Community Policy Compatibility for `jitr`


> This document summarizes the efforts of current and future BAND member packages to achieve compatibility with the BAND SDK community policies.  Additional details on the BAND SDK are available [here](https://raw.githubusercontent.com/bandframework/bandframework/main/resources/sdkpolicies/bandsdk.md) and should be considered when filling out this form. The most recent copy of this template exists [here](https://raw.githubusercontent.com/bandframework/bandframework/main/resources/sdkpolicies/template.md).
>
> This file should filled out and placed in the directory in the `bandframework` repository representing the software name appended by `bandsdk`.  For example, if you have a software `foo`, the compatibility file should be named `foobandsdk.md` and placed in the directory housing the software in the `bandframework` repository. No open source code can be included without this file.
>
> All code included in this repository will be open source.  If a piece of code does not contain a open-source LICENSE file as mentioned in the requirements below, then it will be automatically licensed as described in the LICENSE file in the root directory of the bandframework repository.
>
> Please provide information on your compatibility status for each mandatory policy and, if possible, also for recommended policies. If you are not compatible, state what is lacking and what are your plans on how to achieve compliance. For current BAND SDK packages: If you were not fully compatible at some point, please describe the steps you undertook to fulfill the policy. This information will be helpful for future BAND member packages.
>
> To suggest changes to these requirements or obtain more information, please contact [BAND](https://bandframework.github.io/team).
>
> Details on citing the current version of the BAND Framework can be found in the [README](https://github.com/bandframework/bandframework).


**Website:** [github.com/beykyle/jitr](https://github.com/beykyle/jitr), [pypi.org/project/jitr/](https://pypi.org/project/jitr/)
**Contact:** [beyerk@frib.msu.edu](mailto:beyerk@frib.msu.edu)
**Description:** Just-in-time-compiled solver for the Shrödinger equation using the calculable R-Matrix method on a Lagrange-Legendre mesh 

### Mandatory Policies

**BAND SDK**
| # | Policy                 |Support| Notes                   |
|---|-----------------------|-------|-------------------------|
| 1. | Support BAND community GNU Autoconf, CMake, or other build options. |Full| `jitr` is a Python package and provides a pyproject.toml file for installation, using the common `setuptools` backend. This is compatible with the pip installer. GNU Autoconf or CMake are unsuitable for a Python package.|
| 2. | Have a README file in the top directory that states a specific set of testing procedures for a user to verify the software was installed and run correctly. | Full| README explains full test procedure.|
| 3. | Provide a documented, reliable way to contact the development team. |Full| The `jitr` team can be contacted through the public [issues page on GitHub](https://github.com/beykyle/jitr/issues) or via an e-mail to [the jitr team](https://github.com/beykyle/jitr/blob/main/SUPPORT.rst).|
| 4. | Come with an open-source license |Full| Uses 3-clause BSD license.|
| 5. | Provide a runtime API to return the current version number of the software. |Full| The version can be returned within Python via: `jitr.__version__`.|
| 6. | Provide a BAND team-accessible repository. |Full| https://github.com/beykyle/jitr |
| 7. | Must allow installing, building, and linking against an outside copy of all imported software that is externally developed and maintained .|Full| `jitr` does not contain any other package's source code within. Dependencies are listed in [requirements.txt](https://github.com/beykyle/jitr/blob/main/requirements.txt) |
| 8. |  Have no hardwired print or IO statements that cannot be turned off. |Full| |


### Recommended Policies

| # | Policy                 |Support| Notes                   |
|---|------------------------|-------|-------------------------|
|**R1.**| Have a public repository. |Full|  [https://github.com/beykyle/jitr](https://github.com/beykyle/jitr) |
|**R2.**| Free all system resources acquired as soon as they are no longer needed. |Full| |
|**R3.**| Provide a mechanism to export ordered list of library dependencies. |Full| Dependencies are listed in [requirements.txt](https://github.com/beykyle/jitr/blob/main/requirements.txt) |
|**R4.**| Document versions of packages that it works with or depends upon, preferably in machine-readable form.  |Full|  |
|**R5.**| Have SUPPORT, LICENSE, and CHANGELOG files in top directory.  |Full|  |
|**R6.**| Have sufficient documentation to support use and further development.  |Full| `jitr` has examples in [examples/](https://github.com/beykyle/jitr/tree/main/examples) |
|**R7.**| Be buildable using 64-bit pointers; 32-bit is optional. |Full| Package supports both 32 and 64 bit under same API.|
|**R8.**| Do not assume a full MPI communicator; allow for user-provided MPI communicator. |N/a| |
|**R9.**| Use a limited and well-defined name space (e.g., symbol, macro, library, include). |Full| `jitr` uses the `jitr` namespace |
|**R10.**| Give best effort at portability to key architectures. |Full| `jitr` is tested on a variety of architectures as part of its continuous integration via [github actions](https://github.com/beykyle/jitr/tree/main/.github/workflows)|
|**R11.**| Install headers and libraries under `<prefix>/include` and `<prefix>/lib`, respectively. |Full| The standard Python installation is used for Python dependencies. This installs external Python packages under `<install-prefix>/lib/python<X.Y>/site-packages/`.|
|**R12.**| All BAND compatibility changes should be sustainable. |Full| |
|**R13.**| Respect system resources and settings made by other previously called packages. |Full| `jitr` does not modify system resources or settings.|
|**R14.**| Provide a comprehensive test suite for correctness of installation verification. |Full| tests live in [test/](https://github.com/beykyle/jitr/tree/main/test) and are run via [pytest](https://docs.pytest.org/en/stable/)|
