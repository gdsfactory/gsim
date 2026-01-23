# VLSIR-GDSFactory Connection Logic

---

### Implementation Status

- [ ] Skywater 130nm CMOS
- [ ] Global Foundries 180nm MCU CMOS
- [ ] IHP SG13G2 130nm BiCMOS

### Proposed Solution

After the [VLSIR Circuit](./CONNECTIONs.md) has been determined, we can now begin the final step of preparing the netlist for a simulator - we note that this often-times can be PDK specific as users may wish to specify simulation corners, use special modelling utilities unique to each PDK and each PDK often contains quirks which don't neatly fit into the [VLSIR Metadata Specification](./METADATA_SPEC.md).

