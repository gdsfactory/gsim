# VLSIR-GDSFactory Metadata Specification

---

### Implementation Status

- [x] Skywater 130nm CMOS
- [x] Global Foundries 180nm MCU CMOS
- [x] IHP SG13G2 130nm BiCMOS

### Motivation

Electrical PDKs in GDSFactory describe the layout of devices with SPICE models using parameterized cell and foundry templates wrapped in GDSFactory `Component` objects with attributed `Port` objects to allow for routing between devices. In order for SPICE simulators to correctly represent the intended circuit submitted by the engineer, it is essential that each device comes equipped with metadata which `gsim` `vlsir` utilities can render into `vlsir` protobuf and can then be converted by `vlsirtools` into ngspice, Xyce or Spectre netlists ready for simulation.

To do this, metadata is standardized according to the following format:

## Specification

```python
@gdsfactory.cell
def MyPDKDevice(...):

    c = gdsfactory.Component()

    ... # Layout Programming

    # Specify Electrical Ports!

    c.info['vlsir'] = {
        "model" : [DEVICE MODEL NAME HERE] (str),
        "spice_type" : [SEE IMPLEMENTED SPICE TYPES] (str),
        "spice_lib" : [ASSUMED ROOT, PDK/MODELS] (List[str]),
        "port_order" : [SPICE PORTS OF DEVICE] (List[str]),
        "port_map" : {GDSFACTORY PORT : SPICE PORT} (dict[str,str]),
        "params" : {
            "param1" : PARAM1 (str/int/float),
            "param2" : PARAM2 (str/int/float),
            "param3" : PARAM3 (str/int/float)
        }
    }

    return c
```
