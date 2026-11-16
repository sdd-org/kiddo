---
title: "Kiddo: High-performance, configurable k-d trees for low-dimensional search in Rust"
tags:
  - Rust
  - nearest-neighbour search
  - spatial indexing
  - k-d trees
  - scientific computing
authors:
  - name: Scott Donnelly
    affiliation: "1"
affiliations:
  - name: Independent Researcher
    index: 1

date: 12 November 2026
bibliography: paper.bib
---

# Summary

Kiddo is an open-source Rust library for fast, flexible nearest-neighbour search
in low-dimensional data. Its modular architecture facilitates configurability for
optimizing performance or memory usage for a given use case, or experimenting
with new algorithms and layouts. Performance and ergonomics are guiding principles
of the design.

Like other k-d trees, it partitions space to avoid examining distant points
[@bentley1975]. Kiddo combines this pruning with memory layouts designed for
efficient use of processor caches, vectorised distance calculations, prefetching,
and specialised traversal and result collection. These optimisations target
the cost of each search as well as the number of points examined. Author-run
benchmarks report lower query times than other k-d tree implementations in Rust
as well as implementations in other languages [@spatialbench].

The library supports exact and approximate searches, mutable and immutable
datasets, and several coordinate representations. These choices share a common
tree and query interface, so configurations can be compared without rewriting
the surrounding application. Kiddo is distributed through Cargo under the MIT
or Apache-2.0 licence.

# Statement of need

Spatial searches recur throughout scientific computing. Catalogue matching
requires finding nearby sources; point-cloud processing requires local
neighbourhoods; and particle analyses require selecting neighbours within a
cutoff. Repeating a full scan of a large dataset for each query can make these
operations expensive. Using a spatial index library can reduce the work
while leaving the scientific interpretation of the returned neighbours to the
calling program.

Different workloads impose different requirements: an index built once and
queried many times can benefit from an alternative representation from an index
that accepts insertions and removals. Coordinate precision, memory consumption,
query throughput, and startup costs also matter when an index is reused across
analysis runs.

Kiddo addresses these needs through a common tree and query interface with
configurable storage and traversal strategies. Researchers can select a suitable
representation without implementing their own spatial index. Its scope is
point-based search in low-dimensional coordinate spaces; suitability and
performance still depend on the data distribution, metric, and workload.
Its intended audience also includes researchers developing spatial data
structures, who need to vary algorithms and memory layouts independently while
retaining a common application interface and evaluation framework.

# State of the field

K-d trees are an established technique [@bentley1975], with mature
implementations in several scientific software ecosystems. SciPy offers
nearest-neighbour searching within its Python scientific-computing environment
[@scipy2020; @scipy_kdtree]. The C++ library `nanoflann` provides a header-only
implementation for nearest-neighbour search [@nanoflann2014]. Within Rust,
`kd-tree` provides nearest-neighbour and radius searches over user-defined point
types [@terada_kdtree]. These packages are relevant alternatives when choosing
an index for a particular application.

The SPRK-tree study independently evaluated Kiddo 5.0.3 in a comparison of
spherical range-query implementations [@blasius2026sprk], identifying it among
the strong performers in low-dimensional workloads. These results concern the
pre-v6 implementation; the Spatial Bench comparison below evaluates the
redesigned Kiddo 6.3.0.

Spatial Bench, also developed by Kiddo author Scott Donnelly, provides
author-run comparisons with Rust, C++, and Python implementations [@spatialbench].
A frozen comparison uses Kiddo 6.3.0, `nanoflann` at revision `7812aa0`, and Rust
`nabo` 0.6.0 for single-threaded exact single-neighbour searches over three-dimensional
uniform `f32` point clouds, using squared-Euclidean distance on an AMD Ryzen 5 8500GE.
Across nine tree sizes from 65,536 to 16,777,216 points and 1,000 repeated probes,
`nanoflann`'s recorded median query times were approximately 2.2–2.6 times Kiddo's,
and `nabo`'s were 2.5–3.0 times Kiddo's. The comparison retains fixed configurations
and selects the latest recorded run for each. Timings include adapter and result
handling against prebuilt indexes; construction is excluded. The public records
and methodology document the configurations and timing boundaries.

Kiddo's contribution is a Rust spatial-search library whose internal design is
also intended to support data-structure research and performance tuning. The
v6 redesign makes point storage and stem traversal independent extension points
behind a shared tree and query interface. This lets researchers compare compatible
strategies without maintaining separate search implementations or adapting their
application to each experiment. Achieving this within Kiddo required replacing
the divisions between earlier tree implementations, rather than adding isolated
query methods. Maintaining and redesigning the existing package also preserves
its connection to established research users. This architectural objective
complements the application-facing functionality of the alternatives above;
comparative performance still requires matched semantics, data, and build settings.

# Software design

Version 6 was a substantial rewrite, developed over approximately a year to
address limitations of earlier versions. Previously, mutable and immutable trees
and floating-point and fixed-point implementations were separated. The
[v6 release notes](https://github.com/sdd-org/kiddo/blob/408e63820bea80811b21ed20c77ff545842d2bd9/CHANGELOG.md#600---2026-07-27)
describe the transition to one generic tree with independent stem and leaf
strategies. The goal was an easier-to-use, more coherent library with a foundation
for correctness, API stability, performance tuning, and experimentation.

The architecture makes Kiddo a configurable k-d tree laboratory. A researcher
can hold point storage fixed while changing stem ordering, traversal, or
prefetch behaviour, or vary leaf storage while retaining a compatible stem
strategy. Shared query semantics, reference tests, and benchmark infrastructure
support comparisons of individual design choices. This separates experimental
changes from the application using the index, reducing the work needed to
explore alternative algorithms and data structures.

Stem strategies include breadth-first Eytzinger layouts and blocked Donnelly
layouts with configurable block sizes, prefetching, and SIMD traversal variants.
Leaf strategies determine storage layout and mutability, allowing read-optimised
and updatable trees within the same architecture. Their separation exposes
memory-access and construction-versus-query trade-offs for investigation.
Compile-time strategy selection avoids runtime trait-object dispatch in the
core traversal, at the cost of more configurations to validate and additional
compilation work. Convenience aliases provide defaults for ordinary use.

Earlier APIs exposed separate methods for combinations of query options. The
v6 builder composes query family, ordering, boundary behaviour, and result
projection, with supported combinations expressed through type constraints.
This reduces method proliferation and is intended to accommodate new options
with fewer breaking changes. Reusing the same query model across strategies
also gives correctness checks a common target. Brute-force comparisons,
regression tests, and scalar paths support validation of specialised kernels;
these checks help assess changes across the supported configuration space.

Generic distance metrics separate stored coordinates from distance output,
permitting wider arithmetic where supported without widening every stored point.
This makes numerical trade-offs explicit, particularly for fixed-point data;
exact searches still follow the chosen arithmetic's rounding and overflow
behaviour. Periodic boundaries support repeating simulation domains, while
optional Rayon threading supports construction and batch queries. Optional
Serde serialization and `rkyv` archived querying let applications reuse built
indexes; memory-mapped access and zero-copy deserialization permit tree data
to be loaded very rapidly with a cold OS page cache and effectively instantly
when the OS page cache is warm.

# Research impact statement

By 10 November 2026, Kiddo had accumulated over eight million downloads across
all versions on crates.io, exceeding the totals of the other dedicated Rust
k-d tree crates identified in registry searches [@cratesio_downloads2026].

Kiddo has documented uses in astronomy, information-theoretic estimation,
neuron morphology comparison, computational oceanography, and molecular
configurational entropy estimation. Nessie is a
Rust-based friends-of-friends galaxy-group finder with R and Python interfaces.
@lambert2025nessie explicitly identify the Kiddo crate in their method for
estimating survey completeness by counting galaxies around target positions.
This work documents an earlier use of Kiddo rather than adoption of v6.

Carlson Moses Büth's `infomeasure-rs` provides entropy, mutual-information,
transfer-entropy, and related estimators. Its accompanying preprint explicitly
identifies Kiddo as underpinning nearest-neighbour searches
[@buth2026infomeasure]. The inspected implementation uses Kiddo's Euclidean and
Chebyshev queries in its CPU nearest-neighbour estimator paths. Downstream
integration also informs library development: the v6 migration exposed an
empty-input construction bug, reported with a fix and regression tests in
[pull request #388](https://github.com/sdd-org/kiddo/pull/388), and motivated
query-overhead investigations in [issue #394](https://github.com/sdd-org/kiddo/issues/394).

Chris L. Barnes's `nblast-rs` implements neuron morphology comparison and supplies
a Kiddo backend enabled by default [@barnes_nblast]. Its implementation uses
nearest-neighbour queries to estimate local tangent directions and match points
between neurons, with exact and approximate search options. Its manifest targets
the Kiddo 4.2.1-compatible series, demonstrating an established integration with
an earlier API rather than migration to v6.

A public user report additionally describes two-dimensional nearest-neighbour
searches in computational oceanography after upgrading from Kiddo v5 to v6
[@oceanography2026]. This report supports actual use in that field; it does not
identify a particular ocean-circulation model or publication.

Helmut Carter and Tom Kurtzman's `molentropy` estimates molecular configurational
entropy from molecular-dynamics trajectories using internal coordinates and
nearest-neighbour mutual-information expansions [@carter2026molentropy]. Its
implementation calls Kiddo for neighbour searches and handles periodic torsion
coordinates through wrapped query images. The inspected manifest targets the
Kiddo 5.3.1-compatible series. Carter also proposed periodic queries in
[pull request #297](https://github.com/sdd-org/kiddo/pull/297), providing a public
example of research-driven engagement with the library.

Carter confirms that Kiddo supplied nearest-neighbour distances for the entropy
calculations in the Flexi-GIST solvation-thermodynamics work
[@carter2026flexigist], including subsequent revisions (personal communication
supplied by Scott Donnelly, 9 November 2026). This paper-specific connection is
author-confirmed; the inspected preprint does not explicitly name Kiddo.

Kiddo has also served as an experimental baseline in spatial-index research:
@blasius2026sprk cite and evaluate it in their ESA 2026 study of spherical range
queries. This establishes scholarly evaluation alongside the application
integrations described above.

# AI usage disclosure

Generative AI was not used in Kiddo versions before v5. The fundamental v5 and v6
architecture, data-structure designs, and algorithmic ideas are the author's
own. OpenAI Codex assisted with aspects of the implementation alongside
hand-written code, and with analysis of compiler-generated assembly. All changes,
human-written or AI-assisted, go through comprehensive continuous-integration
testing. Test coverage is checked manually during pull-request review and by
automated coverage tools.

OpenAI Codex (GPT-6.1-Sol, high reasoning effort) assisted in drafting this
manuscript, bibliography, and citation metadata, using repository inspection and
primary-source searches. Scott Donnelly repeatedly reviewed, edited, and validated
the manuscript and retains responsibility for core design decisions and technical
claims.

# Acknowledgements

Kiddo was developed independently, with no funding or institutional support,
and outside the auspices of the author's employer.

I thank [Carlson Büth](https://github.com/cbueth),
[Chris Barnes](https://github.com/clbarnes),
[Francisco Couzo](https://github.com/franciscouzo),
[Jules Bertholet](https://github.com/Jules-Bertholet), and
[Perry Kundert](https://github.com/pjkundert) for their contributions to Kiddo,
including work on earlier versions. Special thanks to Carlson Büth for suggesting
a JOSS submission in the first place, his encouragement, and his offer to review
the manuscript. I thank the wider contributor community for development, testing,
issue reports, and discussions.

# References
