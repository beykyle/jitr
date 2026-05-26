# jitr.folding

`jitr.folding` contains reusable folding infrastructure plus the JLM/JLMB
optical-potential implementation.

## Layout

- `density.py` — general parametric and tabulated density helpers
- `folding.py` — generic Gaussian folding machinery
- `nuclear_matter_self_energy.py` — reusable Lane-decomposed nuclear-matter
  self-energy abstractions
- `jlm.py` — JLM-specific analytical self-energy, JLMB parameters, Coulomb
  shift handling, and `JLMPotential`

## Public API

- Generic helpers live under `jitr.folding`
- JLM-specific helpers live under `jitr.folding.jlm`

That means imports such as:

```python
from jitr.folding import TabulatedNMSelfEnergy, TwoParameterFermiDensity
from jitr.folding.jlm import (
    JLMPotential,
    JLMSelfEnergy,
    JLMSelfEnergyModelParameters,
    JLMV0Parameters,
    make_jlmb_parameters,
)
```

remain valid even though the extra `jitr/folding/jlm/` package layer has been
removed and replaced with the single module `jitr/folding/jlm.py`.

The JLM model coefficients are now explicit dataclasses, so fitting and UQ can
work against named parameters rather than hard-coded literals. For example:

```python
from dataclasses import replace

model = JLMSelfEnergyModelParameters()
custom_model = replace(model, V0=replace(model.V0, energy_constant=125.0))
se = JLMSelfEnergy("n", model_parameters=custom_model)
```
