# VLSIR-GDSFactory Connection Logic

---

**Status** : Not Implemented

### Proposed Solution

GDSFactory can connect components in one of three ways, `.connect` `Component` method which physically arranges components side-by-side (a feature which makes more sense in photonics), `gdsfactory.routing` utilities which draw routing between components using rules specified in the PDK and a third way is the implicit logical links that inform the routing although this to the best of my knowledge is not a persistent state maintained by GDSFactory.

For minimal overhead - I've decided to implement a global `gsim` solution to rely on the `nets` attributed to the `.YAML` schema after calling `gdsfactory.routing` utilities with the following rule:

1. If a `Component` has [VLSIR Metadata](./METADATA_SPEC.md) it is considered a device
2. If a `Component` does not have VLSIR Metadata it is considered routing
3. We consider all connected non-VLSIR `Instance` to be electrical nodes
4. We can handle recursive `Components` with the use of `SUBCKTs`