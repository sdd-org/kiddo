# When an API Type Becomes a Memory Optimization

Kiddo's query builder can express not only which neighbours a query should
find, but also which parts of each result the caller wants to receive. A caller
can request points, stored items, distances, or combinations of those fields.

That sounds like an API convenience. For unsorted radius queries, it is now
also a memory-layout optimization.

The key change is remarkably small:

> Project each accepted result into its requested type before appending it to
> the result vector, rather than collecting a full result vector and projecting
> it afterwards.

Because the projection is represented in Rust's type system, the compiler can
specialize the whole collection path. For an item-only query, the generated
loop still calculates a distance when it needs one for the radius comparison,
but it never stores that distance in the result vector.

There is no runtime mode flag, no duplicated traversal, and no handwritten
item-only algorithm. The existing generic design already contained all the
information required to produce the specialized code. It only needed that
information to reach the point where results are stored.

## The original shape

An unsorted radius query naturally discovers an item and its distance together:

```rust
QueryResultItem<(), T, Distance>
```

The point field is already absent here. The tree traversal knows the item, and
it has just calculated the distance to decide whether the item lies inside the
query radius.

Previously, the eager query path collected those full results first:

```rust
let results: Vec<QueryResultItem<(), T, Distance>> =
    tree.within_unsorted(...);
```

The query builder then applied the requested projection:

```rust
results
    .into_iter()
    .map(project_nearest_without_point::<P, I, Dp>)
    .collect()
```

For an item-only query, this logically produced:

```text
Vec<QueryResultItem<(), T, Distance>>
                    |
                    | project every result
                    v
Vec<QueryResultItem<(), T, ()>>
```

The projection was correct, but it happened too late. The traversal had already
allocated space for full-width elements and written every accepted distance to
that allocation.

The iterator machinery may be able to reuse an allocation in some circumstances,
so this should not be understood as a promise that two separate heap allocations
always occurred. The important fact is that the traversal's output type was the
full result type. Its vector therefore had to be suitable for full-width
elements, and the traversal populated those elements before the projection was
applied.

## The types already knew the answer

Kiddo represents the requested result content with projection types. In
simplified form:

```rust
struct Include;
struct Exclude;

trait ProjectionField<T> {
    type Output;

    fn project(value: T) -> Self::Output;
}

impl<T> ProjectionField<T> for Include {
    type Output = T;

    fn project(value: T) -> T {
        value
    }
}

impl<T> ProjectionField<T> for Exclude {
    type Output = ();

    fn project(_: T) {}
}
```

Calling `without_distances()` does not immediately transform any values. It
changes the query builder's projection type. Conceptually, it changes:

```text
Projection<PointMode, ItemMode, Include>
```

into:

```text
Projection<PointMode, ItemMode, Exclude>
```

That type determines the eventual result type:

```rust
QueryResultItem<
    PointMode::Output,
    ItemMode::Output,
    DistanceMode::Output,
>
```

For the usual item-only radius query, the concrete result is:

```rust
QueryResultItem<(), T, ()>
```

It is still a `QueryResultItem`, preserving a consistent API shape, but both
omitted fields are Rust zero-sized values. For common item types, the vector
element therefore occupies only the space required by the item and its
alignment. For example, in the benchmark configuration:

```text
QueryResultItem<(), u64, f32>  = 16 bytes
QueryResultItem<(), u64, ()>   =  8 bytes
```

The exact layout of a Rust type is not a public ABI promise unless its
representation says otherwise, but the concrete layouts used by the benchmark
can be measured with `size_of`, and `()` itself requires no storage.

## Moving the projection across one boundary

The new collection helper is generic over its final result type and over the
projection function:

```rust
fn within_unsorted_projected<R>(
    ...,
    project: impl FnMut(QueryResultItem<(), T, Distance>) -> R,
) -> Vec<R> {
    let mut results = Vec::<R>::with_capacity(...);

    visit_matching_results(|result| {
        results.push(project(result));
    });

    results
}
```

The query builder supplies the same projection function that it previously
used in `.map(...)`:

```rust
tree.within_unsorted_projected(
    ...,
    project_nearest_without_point::<P, I, Dp>,
)
```

The dataflow is now:

```text
distance calculation
        |
        v
radius comparison ---- rejected
        |
     accepted
        |
        v
type-directed projection
        |
        v
Vec<QueryResultItem<(), T, ()>>
```

There is only one materialized result vector, and its element type is the
caller's final result type.

This is not a separate item-only path. The same function handles every
projection:

- Include the item and distance: `R = QueryResultItem<(), T, Distance>`.
- Include only the item: `R = QueryResultItem<(), T, ()>`.
- Include only the distance: `R = QueryResultItem<(), (), Distance>`.
- Exclude both: `R = QueryResultItem<(), (), ()>`.

The generic parameters select a concrete implementation at compile time.

## What Rust and LLVM do with it

The source still appears to construct a full temporary result:

```rust
|result: QueryResultItem<(), T, Distance>| {
    results.push(project(result));
}
```

It is tempting to conclude that the full result must still be constructed and
then copied into its smaller form. At the source-language level, that is a
reasonable description. It is not necessarily what survives into machine code.

Several compiler mechanisms compose here.

### 1. Monomorphization makes the choices concrete

Rust generates specialized code for the concrete generic parameters used by a
query. An item-only query does not execute a runtime test such as:

```rust
if include_distance {
    ...
}
```

Its compiled instance already has `Dp = Exclude` and:

```rust
<Exclude as ProjectionField<Distance>>::Output = ()
```

The final collection type is therefore statically known to be:

```rust
Vec<QueryResultItem<(), T, ()>>
```

The vector's element size, allocation size, pointer increments, and stores are
all generated for that concrete type.

### 2. Inlining exposes the complete dataflow

The visitor, projection function, `Exclude::project`, and result push are small
generic functions. Once inlined, the optimizer can see a single dataflow rather
than a chain of opaque calls.

For the distance field, it sees the equivalent of:

```rust
let distance = calculate_distance(candidate, query);

if distance <= radius {
    let projected_distance = ();
    push_item_and_unit_distance(candidate.item, projected_distance);
}
```

There is no dynamic dispatch and no unknown callback hiding how the distance is
used.

### 3. Aggregate values can be split apart

LLVM does not have to treat the temporary `QueryResultItem` as an indivisible
object that must live in memory. Scalar replacement of aggregates can represent
its item and distance as independent SSA values.

Conceptually:

```text
temporary result
├── item ───────────────> final Vec store
└── distance ──> project to () ──> nothing
```

The fact that the Rust source passes a struct by value does not imply that a
struct-shaped memory copy must appear in the generated code.

### 4. Dead work after the comparison disappears

`Exclude::project(distance)` ignores its argument and produces `()`. Once the
functions are inlined and the aggregate is split apart, there is no observable
consumer for the distance after the radius comparison.

Dead-code and dead-store elimination can therefore remove:

- Construction of a stored distance field.
- Copies of that field through projection machinery.
- Writes of the distance into the result vector.
- The wider pointer stride and allocation required by a full result element.

The item remains live and is written directly into the final-width vector.

A simplified scalar version of the resulting loop is close to:

```rust
let distance = calculate_distance(candidate, query);

if distance <= radius {
    output.push(candidate.item);
}
```

The real implementation may calculate several distances at once using SIMD and
derive an acceptance mask, but the principle is identical.

## What is _not_ eliminated

The distance calculation itself is not optional for an exact radius query.

Kiddo must still determine:

```rust
distance(candidate, query) <= radius
```

The distance value is live until that decision has been made. Only its life
after the comparison is eliminated.

This distinction matters. The optimization is not "do a radius query without
calculating distances." It is:

> Calculate each distance for exactly as long as the query semantics require,
> then allow it to die instead of serializing it into an externally visible
> result.

That is both more modest and more profound. The algorithm does not change. The
representation of its output becomes faithful to what the caller requested.

## Why unsorted radius queries are the perfect case

An unsorted radius query needs a candidate's distance for one decision: whether
the candidate belongs in the result set. Once accepted, traversal order is
already an acceptable output order. If the caller does not request the
distance, it has no remaining role.

Other query shapes have different constraints:

- A sorted radius query still needs distances to order the accepted results.
- A nearest-_n_ query needs distances to maintain its threshold and select the
  best candidates.
- A query returning point coordinates must obtain and retain those coordinates.

Those queries can still project their public results, but some omitted fields
may remain necessary as internal working data. The optimization applies most
cleanly where a field's last semantic use occurs before collection.

This is a useful general rule:

> Look for values whose final algorithmic use occurs immediately before an
> output boundary. If the output type says the caller does not want them,
> project before crossing that boundary.

## Why this is not benchmark-specific

The motivating comparison made the cost visible because another implementation
returned a narrower result. Dense queries, with tens of thousands of matches,
amplify the cost of writing and moving an unused field.

But the optimization is not tied to that benchmark or to any particular
competitor. It improves a real public operation:

```rust
tree.query(&point)
    .within::<SquaredEuclidean<_>>(radius)
    .unsorted()
    .without_distances()
    .execute()
```

Any application that wants identities but not distances benefits from:

- A result allocation sized for the requested element type.
- Less result memory written.
- Better cache density.
- Less memory bandwidth consumed.
- No intermediate full-width result materialization.

The existing result-capacity hint also becomes more precise in physical terms.
Reserving capacity for `N` results now reserves space for `N` final projected
elements, not `N` full elements that will later be narrowed.

For sparse queries, traversal and distance calculation dominate, so the effect
should be small. For dense queries, result materialization becomes a significant
part of the work, and the narrower representation matters increasingly. That is
the expected shape of a genuine data-movement optimization.

## Why a handwritten fast path was unnecessary

An obvious response would have been to add a second traversal:

```text
within_unsorted_with_distances(...)
within_unsorted_items_only(...)
```

That could avoid the unwanted stores, but it would also duplicate algorithmic
code, increase the testing surface, and risk the two implementations drifting
apart.

The generic projection achieves the same specialization without duplicating
the traversal. Rust provides the vocabulary:

- Generic result types express the desired representation.
- Associated output types turn `Exclude` into `()`.
- Monomorphization creates a concrete item-only instance.
- Zero-sized types make omitted fields occupy no result storage.

LLVM then removes the scaffolding:

- Inlining reveals the projection.
- Aggregate decomposition separates the live item from the dead distance.
- Dead-code and dead-store elimination erase the unused path.

Neither half is sufficient on its own. LLVM cannot optimize through information
that the program hides behind a materialized full-width vector. The type system
cannot improve runtime behavior unless that type information reaches the code
that allocates and writes the results.

The change works because the abstraction boundary moved to the point where both
sides can meet.

## The deeper design lesson

Zero-cost abstraction is sometimes described as writing high-level code that is
"as fast as" a lower-level implementation. This example is more interesting
than that slogan suggests.

The query builder's projection types were originally an API design: they made
result content configurable while keeping invalid or irrelevant choices out of
runtime state. Once propagated into the collector, those same types became:

- An allocation policy.
- An element layout.
- A store-width decision.
- A statement about value liveness.

No new public option was required. The caller had already stated the relevant
fact by choosing `without_distances()`.

The final implementation is smaller than the collection of special cases it
replaces because it aligns three views of the same operation:

1. **Semantic view:** calculate distance to decide membership.
2. **API view:** return only the fields requested by the caller.
3. **machine view:** retain and store only values that remain observable.

When those views line up, specialization stops looking like extra machinery.
It becomes the natural consequence of the types.

That is the magic here—not that the compiler guessed what we wanted, but that
the design told it the truth early enough, and precisely enough, for the
unnecessary work to vanish.
