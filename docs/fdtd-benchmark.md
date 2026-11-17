# Directional coupler: four FDTD solvers

<!-- markdownlint-disable MD033 -->

<style>
.md-content__inner img[src*="benchmarks/charts/"] { width: 100%; max-height: 400px; object-fit: contain; }
.fdtd-example-cards li {
  padding: 0 !important; border: 2px solid #087f72 !important; border-radius: .5rem;
  background: color-mix(in srgb, #087f72 5%, var(--md-default-bg-color));
}
.fdtd-example-cards li:hover, .fdtd-example-cards li:focus-within { box-shadow: 0 4px 14px #087f7230; }
.fdtd-example-cards li a { display: block; padding: 1rem 1.1rem; color: var(--md-default-fg-color); }
.fdtd-example-cards strong { display: block; font-size: 1.05em; }
.fdtd-example-cards span { display: block; margin-top: .5rem; color: #087f72; font-weight: 600; }
.fdtd-response-stack { display: grid; grid-template-columns: minmax(0, 1fr); gap: .75rem; }
.fdtd-response-stack img { width: 100%; height: auto; object-fit: contain; }
.fdtd-response-stack img:last-child { max-width: 530.3px; justify-self: center; }
</style>

**Broadband photonic simulation in just over a minute.**
GDSFactory FDTD solved this 37-million-cell directional coupler on cloud CPUs
in 75 seconds. See how its optical response, throughput and cost compare with
commercial CPU, commercial GPU and Meep solvers.

## The problem

A directional coupler controls how much light passes straight through and how
much transfers to a neighboring waveguide. Getting that split right is central
to designing splitters, interferometers and photonic circuits.

This benchmark uses two **500 × 220 nm silicon waveguides**, a **236 nm gap**
and a **20 µm coupling section**, with SiO₂ cladding. Light enters at **o1**;
the **cross port o3** measures |S31|² and the **through port o4** measures |S41|².

![Directional-coupler layout showing input, cross and through ports](assets/benchmarks/charts/device.png)

<div class="admonition quote">
<p class="admonition-title">Based on a published photonic device</p>
<p>Z. Liu and J. K. S. Poon (2025), <em>Comparison of Lumerical FDTD and Tidy3D for three-dimensional FDTD simulations of passive silicon photonic components</em>.<br>
<a href="https://arxiv.org/abs/2506.16665">Read the original paper</a></p>
<p>The device geometry comes from this study. The four-solver comparison below is a separate benchmark.</p>
</div>

## Cross and through transmission

**Close agreement with the commercial CPU result.** At 1550 nm, GDSFactory
predicts **39.1% cross transmission** and **60.9% through transmission**, each
within 0.7 percentage points of the commercial CPU result. The plots show the
full 1500–1600 nm response for all four solvers at 40 nm nominal cell size.

<div class="fdtd-response-stack">
<img src="assets/benchmarks/charts/transmission_40_power.png" alt="Cross S31 and through S41 transmission with a shared legend">
<img src="assets/benchmarks/charts/ey_top_1550_horizontal.png" alt="Horizontal top-view real Ey at 1550 nm">
</div>

## Throughput

**High throughput on CPUs.** GDSFactory FDTD reaches **29.5 GCUPS** on the
largest measured grid. Compare how throughput changes as the simulation grows
from millions to billions of cells.

![Measured throughput across simulation sizes](assets/benchmarks/charts/throughput.png)

<div class="admonition info">
<p class="admonition-title">What is GCUPS?</p>
<p><strong>GCUPS means giga cell updates per second</strong>: billions of simulation cells advanced by one timestep, every second.</p>
<p><strong>GCUPS = cells × timesteps / (seconds × 10⁹)</strong></p>
<p>Higher throughput means more grid updates per second. One GCUPS-hour is 3.6 trillion cell updates.</p>
</div>

## CPU scalability

**Put more CPU cores to work.** See how each solver's throughput changes as
more CPU cores are used.

![GCUPS versus CPU cores](assets/benchmarks/charts/cpu_scaling.png)

## How much does it cost?

**$1.95 per run for this directional coupler**, including the allocated annual
subscription and usage charges. The table compares the cost of a run; the chart
below shows the cost per GCUPS-hour as simulation size increases.

### Cost of the directional-coupler run

| Solver | Recorded runtime | Customer cost / run |
| --- | ---: | ---: |
| GDSFactory FDTD CPU | 75 s | $1.95 |
| Commercial GPU | 31.2 s | $4.90 |
| Commercial CPU | 221.4 s | $4.36 |
| Meep CPU · self-managed | 1,239.3 s | $0.67 |

Meep uses self-managed compute without a solver license fee.

![Customer cost per GCUPS-hour across simulation sizes](assets/benchmarks/charts/customer_cost.png)

Lower dollars per GCUPS-hour means more grid updates for the same budget.
Above 100M cells, the GDSFactory curve illustrates a hypothetical cost scenario
following the commercial GPU trend. Values through 100M cells use the benchmark's
customer rates.

Alternative commercial offerings combine annual subscriptions with pay-as-you-go
usage rates, similarly to GDSFactory+.

## Try it on your next design

Use these notebooks as a starting point for your own photonic layouts.

<div class="grid cards fdtd-example-cards">
<ul>
<li><a href="nbs/fdtd_directional_coupler.md"><strong>Directional coupler</strong><span>View notebook →</span></a></li>
<li><a href="nbs/fdtd_sacher_visible_mmi.md"><strong>Visible-light MMI</strong><span>View notebook →</span></a></li>
</ul>
</div>
