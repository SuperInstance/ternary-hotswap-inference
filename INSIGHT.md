# Experiment Context

This crate is part of the Flux→PTX experimental suite, testing synergies
between the five layers of the distributed GPU runtime:

1. open-parallel — async runtime (tokio fork)
2. pincher — "Vector DB as runtime, LLM as compiler"
3. flux-core — bytecode VM + A2A agent protocol
4. cuda-oxide — Flux→MIR→Pliron→NVVM→PTX compiler
5. cudaclaw — persistent GPU kernels, warp-level consensus, SmartCRDT

See the full architecture at:
https://github.com/SuperInstance/cuda-oxide/blob/main/GRAND_ABSTRACT.md
