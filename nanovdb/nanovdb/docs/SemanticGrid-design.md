# Semantic grids: bundling index grids with their sidecars

Design notes for a container/view structure that bundles a NanoVDB index grid with the arrays that give it
meaning (value channels, interior masks, statistics, global parameters), so that the bundle can be passed
around as one semantic object, e.g. "a level set".

Status: design discussion only. Nothing here is implemented. All type names (`GridView`, `Attr`, `Nodes`,
`Compact`, ...) are placeholders. Line references are to `develop/nanovdb` at 5be1b3cd1.

Revision (2026-10-28): section 7.9 tightens the enumerations to four classes, `Global`, `Node<...>`,
`Index<...>` and `Range<...>`. Sections 7.1-7.8 are the first iteration, kept as a possible alternative.
Examples elsewhere in this document still use the first iteration's names; 7.9.8 maps them to the new ones.

Contents:

1. Motivation
2. Goals and non-goals
3. Overview: owner, view, accessor, descriptor
4. The entities of a tree
5. `Attr`: describing one array
6. Tags
7. Enumerations (7.9: the current refinement)
8. `GridView`
9. The owning object and the descriptor
10. Statistics as node-indexed arrays
11. Recipes: narrow-band level set and level set
12. How tools plug in
13. What the design relies on in today's NanoVDB
14. Prior art
15. Open questions
16. Suggested first prototype

---

## 1. Motivation

Some NanoVDB grids are self-contained. A `FloatGrid` level set holds everything about the level set in one
buffer: the SDF values of active voxels, the signed background values of inactive voxels and tiles
(after `signedFloodFill`), the background, the transform, and per-node statistics.

Index grids (in this document, `BuildT == ValueOnIndex` unless stated otherwise) are not self-contained.
The grid holds only the topology, and every active voxel carries an index into a separate value array, the
*sidecar*. The sidecar can live in two places:

- inside the grid's buffer, as one of the blind-data arrays (`createNanoGrid` with `channels > 0`);
- in a separate buffer, which is the common case (fVDB, the CUDA tools).

More arrays are appearing alongside the values. `tools::cuda::PropagateSign` (PR 2361) produces, for an index
grid, per-level interior masks and a list of interior root tiles: the information that a flood-filled
`FloatGrid` stores implicitly in the signs of its inactive values. Together with the topology and the SDF
values, these masks make up one semantic object, a level set, spread over several buffers.

Today nothing ties these pieces together. Each tool takes and returns raw pointers and buffers, and the
caller has to know which array belongs to which grid, how each one is indexed, and how to combine them
(for example: an inactive voxel's sign is the interior-mask bit at whichever tree level the lookup stopped).

## 2. Goals and non-goals

Goals:

- One object that carries a grid and its arrays, and can be passed around as "a level set", "a fog volume",
  "a point set", and so on. Not specific to level sets.
- Parametric and modular: a semantic type is a recipe of components, and components can be added to an
  existing object (interior masks, statistics) without changing its type's core.
- Values can attach to any entity of the tree, not just voxels: nodes at every level, tiles, the root, the
  grid as a whole.
- Where an array is stored (blind data in the grid buffer, or a separate buffer) is invisible to kernels.
- Device code gets a trivially copyable view with `__hostdev__` accessors and no runtime lookup cost.
- Consistency is checkable: an array knows which grid it belongs to and how it is indexed, and mismatches
  are caught when a view is taken, not inside a kernel.

Non-goals (for now):

- Changing the NanoVDB grid format itself. The design uses existing blind data for in-buffer storage.
- Replacing `FloatGrid` and other value grids. They remain self-contained; this is for index grids.
- Mutable topology. A bundle describes arrays over one fixed grid buffer.

## 3. Overview: owner, view, accessor, descriptor

The design separates four things, following NanoVDB's existing split between `GridHandle` (owns the bytes)
and `NanoGrid*` (views them).

| Piece | Lives on | Role |
|---|---|---|
| **Owning object** | host | Holds the `GridHandle` and any separate buffers (e.g. stream-ordered `cuda::Buffer<T, ResourceRef<...>>`). Resolves every array to a pointer and hands out views. |
| **View** (`GridView<BuildT, Attrs...>`) | host and device | Trivially copyable: the grid pointer plus one storage handle per array. Passed by value into kernels. Shared and const. |
| **Accessor** | per thread | Wraps `DefaultReadAccessor<BuildT>` (a node cache, so mutable) plus a reference to the view. Tree lookups go through it. |
| **Descriptor** | with the owner, optionally serialized | Runtime list of the arrays: name, element type, enumeration, count, storage location, provenance. Used for I/O, runtime callers, and checking a view before it is handed out. |

Where an array's bytes live matters only to the owning object. A view built from blind data and a view built
from separate buffers look the same to a kernel.

## 4. The entities of a tree

Every level of the tree has two kinds of entity: the **nodes** themselves, and the **slots** inside each
node.

| Level | Node entity (count) | Slot entity |
|---|---|---|
| Grid | 1 | none |
| Root | 1 | root-table entries (variable count) |
| Upper | upper-node count | 32^3 slots, each a child or a tile |
| Lower | lower-node count | 16^3 slots, each a child or a tile |
| Leaf | leaf count | 8^3 voxels |

Values can attach to any of these:

- **Per node:** statistics (min, max, average, standard deviation), bounding boxes, flags, rank offsets.
- **Per slot:** voxel values, tile values, interior masks.
- **Global:** background, isovalue, narrow-band width.

## 5. `Attr`: describing one array

```cpp
template<typename EnumerationT, typename T, typename TagT>
struct Attr
{
    using Enumeration = EnumerationT;  // which entities, and how each maps to an array position
    using ValueType   = T;             // element type
    using Tag         = TagT;          // compile-time name, unique within one view
    using Storage     = typename EnumerationT::template Storage<T>;  // what a kernel holds to reach the data
};
```

`Attr` is a pure compile-time description, like a column in a schema. It holds no data and is never
instantiated. The data lives in buffers owned by the owning object, and as pointers inside a view.

The enumeration does most of the work. Every enumeration provides the same small interface:

```cpp
struct SomeEnumeration
{
    template<typename T> using Storage = ...;   // T*, one T* per level, a Mask<L>*, or a T held by value
    template<typename GridT>
    static uint64_t count(const GridT&);        // element count (per level, for multi-level enumerations)
    template<typename T>
    __hostdev__ static T fetch(const Storage<T>&, const Location&);   // location -> element
    __hostdev__ static bool covers(const Location&);                   // is this location in the entity set?
};
```

`T` and the enumeration together decide the storage. A `bool` over a `Packed<...>` enumeration is stored as
bits, so its `Storage` is a `const Mask<L>*` and `fetch` tests a bit.

| `Attr` | `Storage` | `fetch(loc)` |
|---|---|---|
| `Attr<Compact<Slots<Leaf>, Active>, float, tag::Sdf>` | `const float*` | `data[leaf.getValue(slot)]` |
| `Attr<Nodes<Leaf>, Stats<float>, tag::Stats<tag::Sdf>>` | `const Stats<float>*` | `data[nodeId]` |
| `Attr<Nodes<Leaf, Lower, Upper, Root>, Stats<float>, ...>` | one pointer per level | `data[loc.level][nodeId]` |
| `Attr<Packed<Slots<Leaf, Lower, Upper>>, bool, tag::Interior>` | one `Mask` array per node level | bit `slot` of the mask of node `nodeId` at `loc.level` |
| `Attr<Global, float, tag::Background>` | `float` | the value |

One `Attr` serves three consumers:

1. the **view**, through `Storage` and `fetch`;
2. the **owning object**, through `Enumeration::count(grid)` and `sizeof(T)`, to allocate buffers;
3. the **descriptor**, through the tag's name, `T` (mapped to a `GridType`) and an enumeration id.

An alternative would let the tag own `T` and the enumeration (`struct Sdf { using ValueType = float; ... }`)
so a view lists only tags. That is shorter, but fixes `Sdf` to one type and one enumeration everywhere.
Keeping `Attr` a triple lets the same tag mean, for example, a `double` SDF in another view. Aliases can
cover the common cases.

## 6. Tags

A tag is the array's name, written as a type: an empty struct, optionally carrying runtime metadata.

```cpp
namespace tag {
struct Sdf        { static constexpr const char* name = "sdf"; };
struct Background { static constexpr const char* name = "background"; };
struct Interior   { static constexpr const char* name = "interior"; };
template<typename ChannelTag> struct Stats {};   // the statistics of one channel
}
```

What a tag is for:

1. **Telling apart arrays that look the same.** An SDF and a density are both `float` per active voxel; only
   the tag distinguishes `get<tag::Sdf>` from `get<tag::Density>`.
2. **Stating what a tool needs.** A tool can require arrays by meaning, regardless of what else a view carries
   or in what order:

   ```cpp
   template<typename ViewT>
   __hostdev__ float marchToSurface(const ViewT& view, Ray ray)
   {
       static_assert(ViewT::template has<tag::Sdf>() && ViewT::template has<tag::Interior>(),
                     "marchToSurface needs a level set: an SDF channel and interior masks");
       ...
   }
   ```

3. **Linking to runtime names.** The tag's `name` (and possibly a `GridBlindDataSemantic`) is what the
   descriptor and file I/O match against when a view is taken.
4. **Zero cost and open to extension.** Lookup by tag resolves at compile time; a misspelled tag is a compile
   error. Anyone (NanoVDB, fVDB, users) can declare a tag without editing NanoVDB.

Alternatives and why not: strings cannot easily be template parameters in C++17 and cost a lookup in
kernels; position in the view's list breaks when components are added or reordered; a shared `enum` is a
closed list only NanoVDB could extend.

**Template tags** express a component applied more than once (USD's multiple-apply schemas), and record
dependencies: `tag::Stats<tag::Sdf>` and `tag::Stats<tag::Density>` are two instances of the statistics
component, and each names the channel it was computed from.

## 7. Enumerations

Sections 7.1-7.8 are the first iteration. Section 7.9 refines and tightens them, and is the current
proposal; the first iteration stays here as a possible alternative and as the reasoning that led to 7.9.

### 7.1 Three independent choices

An enumeration is built from three choices.

**Which entities**

- `Global`: one value for the whole object.
- `Nodes<Levels...>`: the nodes of one or more levels (leaf, lower, upper, root).
- `Slots<Levels...>`: the slots inside the nodes of one or more levels.

**Which subset**, a test on each slot

- `All`, `Active`, `Inactive`.
- `Child` or `Tile`: internal slots with or without a child.
- Combinations, e.g. `ActiveTile`.

**How an entity maps to an array position**

- **Dense:** computed from where the entity sits: `nodeId`, `nodeId * SLOTS + n`, or the root-table index.
  Nothing stored. With a subset, excluded positions exist but are unused.
- **Packed:** `bool` only, one bit per slot, stored as one `Mask<L>` per node.
- **Compact:** the entity's rank among the members of the subset. No wasted space; needs a rank structure
  (7.4).
There is deliberately no free-form "indexed" mapping (each entity holding an index into a shared table, like
USD's indexed primvars). That would break the rule in 7.7. Values that repeat, e.g. a material id per voxel
looking up a palette, are modelled as two ordinary arrays: a per-voxel `uint32_t` array whose values happen
to be indices, and the palette under `Global` or its own flat enumeration. The indirection is in the data,
not in how entities map to positions.

### 7.2 Catalogue

| Enumeration | Size | `fetch` | Example contents |
|---|---|---|---|
| `Global` | 1 | the value | background, isovalue |
| `Nodes<Leaf>` | leaf count | `data[id]` | leaf statistics, leaf bounding boxes |
| `Nodes<Leaf, Lower, Upper, Root>` | one count per level | `data[level][id]` | statistics at every level |
| `Dense<Slots<Leaf>>` | leaf count x 512 | `data[id*512 + n]` | a value for every voxel slot |
| `Packed<Slots<Leaf, Lower, Upper>>` | bits per level | the mask bit | interior masks, per-slot flags |
| `Compact<Slots<Leaf>, Active>` | active voxel count + 1 (slot 0) | the grid's own offsets | a value channel, i.e. what `ValueOnIndex` provides |
| `Compact<AllSlots, All>` | the grid's value count | `ValueIndex` offsets | every value including tiles: a value grid stored as index grid plus channel |
| `Compact<Slots<Lower, Upper>, ActiveTile>` | active tile count | its own rank structure | values of active tiles |
| `RootKeys` | a list of coordinates | search by key | PropagateSign's interior root tiles |

### 7.3 Node ids come from the layout

Each level's nodes are stored contiguously and have one size, so a node's id is a pointer difference:

```cpp
const uint64_t id = nodePtr - tree.getFirstNode<LEVEL>();   // NanoVDB.h:2588, :2607
```

This makes `Nodes<L>`, `Dense` slots and `Packed` slots free: nothing is stored to map an entity to its
position. `ReadAccessor` already caches node pointers, so a lookup that reports the node it stopped at costs
nothing extra. (The division by `sizeof(Node)` inside the pointer subtraction is by a constant and compiles
to a multiply.)

Node ids depend on the order in which the builder stored the nodes. That order is fixed for a given buffer,
but two buffers with the same topology built by different tools could order nodes differently. Per-node
arrays therefore belong to a specific grid buffer, not just to a topology (see 9.3).

### 7.4 Rank structures are themselves arrays

For `Compact<Slots<L>, Filter>`, an entity's position is its rank among all slots that pass the filter,
which depends on how many passing slots precede it in every earlier node:

```
rank = offset[id] + popcount(filterMask[id] bits below n)
```

where `offset` is a prefix sum of the per-node counts. This is how index-grid leaves already work
(`mOffset` and `mPrefixSum` in `LeafIndexBase`, NanoVDB.h:4160 onward).

Because node ids are free (7.3), the rank structure need not live inside the node. It can be two ordinary
arrays: a `Nodes<L>` array of `uint64_t` offsets, and the filter mask (either a mask the node already has,
such as the value or child mask, or a `Packed` array). So:

- **Any subset can be compact without changing the grid format.** Active tiles, inactive voxels, voxels in a
  region: each is one prefix-sum pass when the enumeration is created.
- **An index grid's own offsets are an optimization, not a requirement.** `Compact<Slots<Leaf>, Active>`
  uses the built-in `mOffset` when present; a plain topology grid with a separate offsets array would serve
  the same role.
- **Cost:** one extra load (`offset[id]`) from a separate array, versus reading `mOffset` from the node,
  usually on the same cache line as the mask. Small, but measurable in tight loops: a reason to keep the
  built-in path for active voxels.

Section 7.8 makes this concrete for leaf-level subsets: an additional index layer.

### 7.5 The root

The slots of internal nodes form a fixed grid (32^3, 16^3, 8^3). The root's slots are unbounded, and its table
holds only entries that exist (sorted, NanoVDB.h:3219). PropagateSign's interior root tiles show the
problem: under its precondition that the root holds only child tiles, an interior root tile is a position
that is not in the table at all, so no table index can address it.

The root therefore needs a keyed enumeration, `RootKeys`: entities named by coordinate, found by search.
It is the only enumeration whose `fetch` is not direct arithmetic. As a consequence the interior component
is two enumerations under one tag: `Packed<Slots<Leaf, Lower, Upper>>` for the node levels plus `RootKeys`
for the root.

### 7.6 Three categories of trust

Enumerations differ in how far correctness depends on stored data.

1. **Implied by the layout: nothing stored, cannot be wrong.** `Global`, `Nodes<L>`, `Dense` slots,
   `Packed` slots, the root-table index. Nodes are compact by construction: their ids cannot overlap or
   leave gaps.
2. **Canonical rank: determined by the topology, so stored offsets are a cache.** The format declares that
   compact ranks are canonical: walk the nodes in storage order, take each node's mask bits in order, and
   number them. A rank is then a function of the masks and the node order. Any stored offsets are a cache of
   that function: checkable in one prefix-sum pass, rebuildable from the masks, and overlaps or gaps are
   errors.
3. **Free-form stored mapping: anything goes, so it must be checked.** Only one enumeration is allowed in
   this category: active voxels, through the grid's own offsets (7.7). `RootKeys` stays in category 2 by
   requiring its keys to be sorted and unique. Explicit permutations and per-entity lookup tables are not
   enumerations at all (7.1).

Voxels differ from nodes here. Voxel indices under `mOffset` are stored data, so two leaves could have
overlapping index ranges, or leave gaps, or be out of order. Nodes cannot.

### 7.7 The rule: only active voxels may be free-form

**Every enumeration is layout-implied or canonical, with one exception: the mapping from active voxels to
value positions may be whatever the grid's offsets say.**

Why active voxels are the right exception:

- **Existing grids stay valid.** Index grids with interleaved statistics, or any other hand-built offsets,
  need no conversion; their active-voxel mapping is simply not canonical.
- **There are uses for free-form offsets,** chiefly several grids pointing into one shared array with
  disjoint ranges, for batching. Making that canonical would need a per-grid base offset, a separate
  question.
- **Arrays over active voxels still agree with each other.** They all go through the same grid's offsets,
  as `ChannelAccessor` does today (NanoVDB.h:5884). A free-form mapping is harmless when everyone uses the
  same one; mixing mappings is what goes wrong.
- **Everything new stays canonical.** Per-node arrays, slot masks, compact ranks over other subsets, and
  sorted, unique root keys add no new free-form data.

Contract for the active-voxel offsets:

- **In range:** every index is less than the array's length. Every reader needs this.
- **No two voxels share an index:** needed by any tool that writes one value per voxel. Optional for
  read-only arrays.
- **Gaps are allowed.** That keeps grids with interleaved statistics valid.

The slot-0 convention is part of the definition: a `ValueOnIndex` leaf returns index 0 for an inactive
voxel (NanoVDB.h:4245), and active voxels are numbered from 1, so a channel built by `createNanoGrid` has
`activeVoxelCount + 1` entries with the background in slot 0. Sidecars elsewhere (fVDB, the CUDA tools) are
often sized `activeVoxelCount` and read with an offset of -1. The `Compact<Slots<Leaf>, Active>` enumeration
must fix one convention and record it in the descriptor, because mixing the two is a silent off-by-one. With
slot 0 included, `Background` can simply be `sdf[0]`.

Undecided: whether the descriptor records a flag "active-voxel offsets are canonical", checked in one
prefix-sum pass when the object is built. Many tools (the CUDA tools, `VoxelBlockManager`, fVDB) silently
assume canonical offsets, treating the n-th active voxel's value as `array[n]`. With the flag, such tools
require it once when they take a view, instead of each tool checking for itself or relying on the
assumption silently.

If the flag is set, the grid's own `mOffset` and `mPrefixSum` are, like any canonical offsets, only an
acceleration structure (7.8): a cache that happens to live inside the leaves.

### 7.8 Additional index layers

Some attributes need a second set of voxels with its own consistent numbering. Example: each leaf's active
voxels plus their 6-connected neighbours within the leaf, with every slot in that extended set numbered.

**What a layer needs.** The same three things a `ValueOnIndex` leaf holds, kept as per-leaf arrays:

- a membership mask per leaf (`Mask<3>`): which slots belong to the set;
- an offset per leaf: where the leaf's range starts;
- optionally, the within-leaf prefix sum, for constant-time rank (otherwise a popcount over up to eight words).

All three are indexed by leaf id, so they stay aligned with each other and with the grid for free (7.3). A
lookup always needs both mask and offset, so they are best stored together, one record per leaf:

```cpp
template<uint32_t LOG2DIM>
struct IndexBlock                       // 80 bytes for LOG2DIM = 3
{
    Mask<LOG2DIM> mMask;                // membership; not "active", hence not mValueMask
    uint64_t      mOffset, mPrefixSum;  // as in LeafIndexBase
};

Attr<Nodes<Leaf>, IndexBlock<3>, tag::Layer<tag::Halo>>            // the layer
using HaloVoxels = Compact<Slots<Leaf>, LayerOf<tag::Halo>>;
Attr<HaloVoxels, float, tag::HaloSdf>                               // a channel over the extended set
```

A channel over a layer refers to the layer by tag, so taking a view checks that the layer is present. A
location (leaf id, slot) is enough to read any channel over any layer; a voxel in two layers has two indices,
and nothing converts between them:

```cpp
auto loc = acc.probe(ijk);
float a  = view.get<tag::Sdf>(loc);       // through the grid's mOffset (active voxels)
float h  = view.get<tag::HaloSdf>(loc);   // through the Halo layer's IndexBlock
```

**Layers are canonical.** An additional layer follows the compact contract (7.6, category 2), so the rule in
7.7 still holds. Its offsets are fully determined by the masks and the leaf order (the exclusive prefix sum of
each leaf's popcount, plus a fixed start), so they are only an acceleration structure:

- Random access needs them; a kernel that sweeps all leaves can compute them on the fly with one scan.
- They need not be stored or serialized. The owner can persist only the masks and build the offsets with one
  scan when a view is first taken.
- Offsets the owner built are correct by construction; loaded ones can be checked or rebuilt.

This holds only if the contract fixes the starting value (0, or 1 with slot 0 reserved; use the same
convention as active voxels), the order (leaves in storage order, bits in order), and no sharing of arrays
between grids (batching would need a per-grid base offset).

If the membership is itself a function of the topology, as in the example (each leaf's value mask dilated
within the leaf, which the `MaskShift` helpers in `util/MorphologyHelpers.h` compute), the masks are a cache
too, and the descriptor records that provenance so the layer is stale when the topology changes. If the set
is chosen by hand, the masks are primary data. Whether a layer must contain the active voxels is a recorded
property, not a requirement.

**Sharing code with `LeafData<ValueOnIndex>`.** An `IndexBlock` is exactly the lower part of
`LeafIndexBase` (`mValueMask`, `mOffset`, `mPrefixSum`), without the leaf's identity (`mBBoxMin`,
`mBBoxDif`, `mFlags`), which a layer does not need because it shares the grid's leaves. The rank code in
`LeafData<ValueOnIndex>::getValue` (NanoVDB.h:4240) is exactly what a layer needs. Ways to share it:

- *`IndexBlock` as a base class of `LeafIndexBase`:* rejected. A base class's members come first in memory,
  which would move the mask from byte 16 to byte 0. PNanoVDB hard-codes the leaf layout
  (`PNANOVDB_LEAF_OFF_BBOX_MIN 0`, `PNANOVDB_LEAF_OFF_VALUE_MASK 16`, PNanoVDB.h:1830), as do the C port,
  shaders and existing files. It would be a format change for every index grid.
- *`IndexBlock` as a member of `LeafIndexBase`:* same bytes, but every `->mValueMask` becomes
  `->mIndex.mValueMask`, a broad source change.
- *A view type over the leaf's bytes* (`reinterpret_cast`): works in practice, but relies on aliasing
  assumptions for no benefit over the next option.
- *A shared free function (recommended):*

  ```cpp
  template<uint32_t LOG2DIM>
  __hostdev__ uint64_t rankOf(const Mask<LOG2DIM>& mask, uint64_t offset, uint64_t prefixSum, uint32_t i);
  ```

  `LeafData<ValueOnIndex>::getValue` calls it with its own members, and `IndexBlock` with its own. No format
  change, no renames, and the same tested code for both.

`IndexBlock` should not carry `NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT)`: 32-byte alignment would pad each record
from 80 to 96 bytes. Eight-byte alignment, from its `uint64_t` members, is enough.

**Rejected alternative: a second grid.** Building a second `ValueOnIndex` grid whose leaf masks are the
extended masks gives `mOffset`, `mPrefixSum` and `ChannelAccessor` for free, but duplicates every internal
node, lines up leaf ids only if the builder happens to store leaves in the same order, needs two tree lookups
per query, and leaves two grid identities to keep consistent. The `IndexBlock` array reuses the first grid's
tree to find the leaf.

The first layer stays special only in what its mask means: the grid's `mValueMask` is both membership and
topology (it defines which voxels are active), while a layer's mask is only membership. How the two index is
the same.

### 7.9 Refinement: `Global`, `Node`, `Index`, `Range` (current proposal)

This section tightens 7.1-7.8 into four enumeration classes. Where the two disagree, this section is the
current proposal; 7.1-7.8 remain a possible alternative.

#### 7.9.1 The four classes

| Class | Entities | Position comes from | Examples |
|---|---|---|---|
| `Global` | the object as a whole | nothing: position 0 | background, isovalue, grid-wide statistics |
| `Node<Levels...>` | the nodes of each listed level | the layout: the node's place in the flattened tree | statistics, interior masks, rank records, dense per-slot values |
| `Index<...>` | a subset of the slots of one level | stored offsets (a rank source) | value channels (SDF, colour), channels over additional voxel layers, active-tile values |
| `Range<...>` | a variable number of items per entity | stored offsets plus per-entity ranges (a range source) | point attributes, triangle or Gaussian ids per voxel |

The names diverge deliberately. `Global` and `Node` are named after what the values attach to, and their
positions follow from the layout: nothing is stored, nothing can be wrong. `Index` and `Range` are named after
the mechanism, and their positions go through stored offsets, which may have gaps, overlaps or a
non-monotonic order unless a stated property rules that out (7.9.3). The name alone tells a reader which kind
an array is.

#### 7.9.2 `Node<Levels...>`

- **Each level is indexed separately.** A position is (level, node id), where the id is
  `nodePtr - getFirstNode<L>()` (7.3), and each level's positions run from 0 to that level's node count. No
  enumeration has a position space spanning levels. The interface works per level (`count<L>(grid)`), and
  `fetch` uses `loc.level` to choose the level's range.
- **A multi-level `Attr` is a set of single-level arrays under one tag,** so `get<Tag>(loc)` works at
  whatever level a lookup stopped. Leaving a level out of the list means that level has no data.
- **The record can be anything:** statistics, a mask, a rank record, or a value for every slot of the node.
  Dense per-slot data is simply a per-node record of `SLOTS` values (e.g. `float[512]` per leaf), so no
  separate dense slot enumeration is needed.
- **Element types may depend on the level.** Interior masks are `Mask<3>`, `Mask<4>` and `Mask<5>` at the
  leaf, lower and upper levels. `T` may therefore be a family resolved per level, e.g. a `NodeMask` marker
  whose element at level L is that level's `Mask<LOG2DIM>`:

  ```cpp
  Attr<Node<Leaf, Lower, Upper>, NodeMask, tag::Interior>
  ```

  Since each level is indexed and stored separately, a different type per level is natural, and one tag still
  covers all levels. For masks, `get<Tag>(loc)` returns the node's whole mask; a helper `isOn<Tag>(loc)`
  tests bit `loc.slot` of it.
- **The root level is the root table.** `Node<Root>` enumerates root-table entries by table index:
  `RootData::probeTile(ijk)` (NanoVDB.h:2830) returns the entry, and its index is `tile - tile(0)`, the same
  pointer subtraction as node ids. Table indices are not upper-node ids: the table also holds value tiles.
  The root node's own single record (e.g. grid-wide statistics) is `Global`.

Masks are always `Node` records in this refinement. The first iteration declared them as per-slot booleans
(`Packed<Slots<...>>`), and a marker element type `MaskBit` was considered to trigger packed storage
explicitly (rather than `bool`, which would repeat the `std::vector<bool>` mistake; `bool` would then mean one
byte per slot, which allows writes without atomics). With masks as `Node` records there is only one way to
declare a mask, and neither `MaskBit` nor the overlap between `Nodes<L>` with `Mask<>` and packed slots arises.

#### 7.9.3 `Index<...>`

Values for a subset of the slots of one level, reached through a **rank source**: per node, a membership
mask, an offset, and within-node prefix counts. An `Index` array names its rank source, so taking a view
checks that it is present:

```cpp
Attr<Index<Leaf, BuiltinActive>, float, tag::Sdf>       // the grid's own leaves are the rank source
Attr<Index<Leaf, tag::Halo>,     float, tag::HaloSdf>   // rank source: the Node<Leaf> array tagged Halo
```

**Offsets are laissez-faire, with properties stated on top.** Gaps, overlaps and non-monotonic offsets are
all representable. Each rank source records which of these properties it guarantees, and each tool states
which it needs:

1. **In range:** every position is less than the array's length. Needed by every reader.
2. **No overlaps:** no two entities share a position. Needed by any tool that writes one value per entity.
3. **Canonical:** monotonic, no gaps, rank order (nodes in storage order, bits in order, a fixed start).
   The offsets are then only an acceleration structure: rebuildable from the masks, checkable in one
   prefix-sum pass. Tools that treat a channel as a flat array need this.

Examples: a freshly built grid without statistics is canonical; an old grid with interleaved statistics, or a
growable grid (section 17), has no overlaps but has gaps. This generalizes the "canonical" flag of 7.7, and
replaces 7.7's rule that only active voxels may be free-form: free-form offsets are now a property of a rank
source, not a privilege of active voxels.

**Rank sources:**

- **Leaf, built in:** the grid's own `mValueMask`, `mOffset` and `mPrefixSum`. Restricted to leaf voxels: an
  index grid built with tile indices (`createNanoGrid(..., includeTiles = true)`, the default) numbers tiles
  and voxels in one space spanning levels, which per-level indexing does not allow. Such grids remain
  readable as old grids, as grids with interleaved statistics do; new grids are built without tile indices,
  in line with PropagateSign's precondition #4.
- **Leaf, additional layer:** a `Node<Leaf>` array of `LeafData<ValueOnIndex>` records: a copy of the grid's
  leaves with the mask, `mOffset` and `mPrefixSum` overwritten for the layer, and the statistics flag off.
  It is the same type as the built-in source, so the existing rank code (`getValue`, `valueCount`) applies
  unchanged, and each record carries its leaf's origin, which kernels running over the layer need (nodes
  have no parent pointers). The cost is 96 bytes per leaf rather than the 80 of a bare `IndexBlock`. This
  supersedes 7.8's recommendation (`IndexBlock` plus a shared `rankOf` function), which remains an
  alternative. In a layer, `mValueMask` means membership in the layer, not activity.
- **Internal levels:** a small per-node rank record, not a copy of the tree's internal node type
  (`InternalData<ValueOnIndex>` gives every slot an 8-byte table entry: about 33 KB per lower node and 270 KB
  per upper node):

  ```cpp
  template<uint32_t LOG2DIM>
  struct RankBlock
  {
      uint64_t mOffset;                                  // rank of the node's first member, among this level's nodes
      uint16_t mPrefix[Mask<LOG2DIM>::WORD_COUNT];       // members before each mask word, within the node
  };
  ```

  About 136 bytes per lower node and 1 KB per upper node (about 0.4% of the node); counts per group of 8
  words cut this by 8 at the cost of up to 7 popcounts. The leaf's packing of seven 9-bit counts into one word
  does not scale: a lower node has 63 counts of up to 13 bits. The membership mask comes from the tree for
  tree subsets (active tiles: `value & ~child`), or is stored in the record for layers. Nothing needs this yet;
  it is built only when a tool asks for sparse values at an internal level.
- **Complements come free from canonical sources.** Within a node, the rank of slot `n` in `~mask` is
  `n - rank(n)`, and the complement's node offset is `id * SLOTS - offset(id)` (adjusted for the start
  value). For example, inactive leaf voxels can be ranked from the active voxels' rank data, but only if
  those offsets are canonical.

**Considered and deferred: a dedicated `Index` build type** whose leaf holds only the mask, `mOffset` and
`mPrefixSum`, so that a grid's leaf array is exactly an array of rank records. It saves no memory in the tree
(nodes are 32-byte aligned, so 80 bytes pad to 96, which is what `LeafIndexBase` already is), leaf-parallel
code needs the origin it would drop, and it is a format change (new `GridType`, PNanoVDB and C-port support,
`LeafNode` specializations). If ever pursued, its name would collide with `bundle::Index`.

#### 7.9.4 `Range<...>`

A variable number of items per entity, stored contiguously: an item's position is (entity, k) for
`k < count(entity)`. NanoVDB already has this built into point grids: `LeafData<Point>` (NanoVDB.h:4270
onward) holds `mOffset`, `mPointCount` and `uint16_t mValues[512]`, the cumulative end of each voxel's points,
so voxel `i`'s points are `[first(i), last(i))`, read through `PointAccessor::voxelPoints`.

Uses: point attributes (the `Point*` blind-data semantics), triangle or Gaussian ids per voxel
(`TriangleId`, `GaussianId`), the existing `GridBlindDataSemantic::Range`, fVDB's jagged tensors, and
variable-width stencils such as general sparse-matrix rows. None reduces to `Index`, which gives each member
exactly one position, or to a fixed-size element type.

A `Range` array names its range source, as an `Index` array names its rank source; for point grids it is the
grid's own leaves. The same property ladder applies. Including `Range` lets a bundle describe a point grid with
its attributes, arguably NanoVDB's oldest "grid plus sidecars" case.

#### 7.9.5 The root and sanitized level sets

Convention: **sanitized SDF level sets insert interior root tiles into the root table.** A childless root
tile's existence means interior, so the interior attribute has no root-level part, and no keyed enumeration
is needed:

```cpp
const auto* tile = root.probeTile(ijk);
return tile && !tile->isChild();   // childless entry: interior; absent: exterior; child: continue down
```

- **Sanitized** means the root holds child tiles and inactive interior tiles, nothing else: no exterior
  (+background) value tiles and no active root tiles. A `FloatGrid` converted from OpenVDB can have
  +background root tiles, which sanitizing must drop. The descriptor records the property, and the level-set
  recipe checks it when a view is taken. In an index grid the interior tiles have value index 0 (the
  background slot).
- This matches OpenVDB: after `signedFloodFill`, interior root regions are -background tiles and exterior
  regions are mostly absent.
- **PropagateSign:** its list of interior root-tile origins becomes the input to a root rebuild that produces
  the sanitized grid. Lengthening the root table moves every node, since the table precedes them in the buffer;
  this appears cheap (shift the node region, fix the root tiles' child offsets and the tree header's node
  offsets; offsets between internal nodes and their children are relative and node order is unchanged, so
  node ids and per-node arrays stay valid), but this is not yet verified. The shared root-building helpers
  from #2389 are the natural home.
- A sanitized grid violates PropagateSign's precondition #4 (the root holds only child tiles). Accepting
  sanitized inputs, with childless root tiles as known interior, is a possible future relaxation. Other tools
  that assume the root table holds only child tiles must be checked before they accept sanitized level sets.

#### 7.9.6 Dropped from the first iteration, and why

| First iteration | Status | Reason |
|---|---|---|
| `Slots<...>` with dense positions | dropped | dense per-slot data is a `Node` record of `SLOTS` values; a compact rank over "all slots" equals the dense position anyway |
| `Packed<...>` with `bool`, or a `MaskBit` element type | dropped | masks are `Node` records (7.9.2) |
| `RootKeys` (keyed by coordinate) | dropped | root-table indexing plus the sanitized convention (7.9.5) |
| free-form `Indexed` lookup tables | dropped | lookup tables are data: two ordinary arrays (7.1) |
| a separate batch class above `Global` | deferred | a per-grid base offset can be a `Global` added by `Index`, unless batching becomes central |
| "only active voxels may be free-form" (7.7) | generalized | free-form offsets are a stated property of each rank source (7.9.3) |

#### 7.9.7 Names

- `Node` rather than `Nodal`: in simulation vocabulary "nodal" means values at grid vertices (voxel corners),
  as opposed to cell- or face-centred, exactly the wrong reading for NanoVDB's level-set and MAC-grid users.
  `PerNode` was the runner-up.
- `Node<Leaf>` names an entity and an enumeration at once, like *vertex* and *fragment* in shaders or
  Houdini's point and primitive attributes. It lives in its own namespace (e.g. `bundle::Node`) and appears only
  as `Attr`'s first argument, so it is not confused with NanoVDB's node types.
- `Index` rather than `Indexed`, and `Range` to parallel it (matching `GridBlindDataSemantic::Range`). "Index"
  is crowded in NanoVDB (`ValueIndex`, `ValueOnIndex`, `IndexGrid`); the namespace handles those.

#### 7.9.8 Mapping from the first iteration

Examples elsewhere in this document use the first iteration's names. They translate as follows:

| First iteration | Refinement |
|---|---|
| `Global` | `Global` |
| `Nodes<Levels...>` | `Node<Levels...>` |
| `Dense<Slots<L>>` with element `T` | `Node<L>` with a record of `SLOTS` values of `T` |
| `Packed<Slots<Leaf, Lower, Upper>>`, `bool` (interior masks) | `Node<Leaf, Lower, Upper>`, `NodeMask` |
| `RootKeys` (interior root tiles) | interior root tiles in the root table (7.9.5) |
| `Compact<Slots<Leaf>, Active>` (`ActiveVoxels`) | `Index<Leaf, BuiltinActive>` |
| `Compact<Slots<L>, Subset>` | `Index<L, rank source>` |
| an `IndexBlock` layer with `LayerOf<tag::X>` | `Index<Leaf, tag::X>`, rank source a `Node<Leaf>` array of `LeafData<ValueOnIndex>` |
| point grids (not covered) | `Range<...>` |

## 8. `GridView`

```cpp
template<typename BuildT, typename... Attrs>
struct GridView : detail::AttrHolder<Attrs>...   // one wrapper per Attr, so equal Storage types don't collide
{
    const NanoGrid<BuildT>* mGrid;

    template<typename Tag>
    __hostdev__ static constexpr bool has();

    template<typename Tag>
    __hostdev__ auto get(const Location& loc) const
    {
        using A = detail::FindByTag<Tag, Attrs...>;   // compile-time; ambiguous or missing tags fail to compile
        return A::Enumeration::fetch(static_cast<const detail::AttrHolder<A>&>(*this).mStorage, loc);
    }
};
```

- `BuildT` fixes the topology type, `NanoGrid<BuildT>`.
- Each `Attr` contributes one storage member, held in a distinct wrapper base so two arrays with the same
  `Storage` type do not collide.
- Storage is hand-rolled as one base class per array rather than `std::tuple`, which is not usable in device
  code without extra nvcc flags. `cuda::std::tuple` from CCCL (already a NanoVDB dependency) would also work.
- Tags must be unique within a view.

### 8.1 Locations

A tree lookup ends at a **location**: the level where it stopped, the node id at that level, and the slot
within the node. Every array can be read from a location, so a view needs one lookup and then any number of
fetches:

```cpp
auto acc = view.getAccessor();
auto loc = acc.probe(ijk);                  // where the lookup stopped: level, node id, slot
float v  = view.get<tag::Sdf>(loc);         // defined only if loc is an active voxel
bool  in = view.get<tag::Interior>(loc);    // defined at any level's childless slot
auto  s  = view.get<tag::Stats<tag::Sdf>>(loc.node());
```

`Enumeration::covers(loc)` says whether an array has an element at a location; reading one that does not is
a precondition violation (assert in debug builds).

### 8.2 Accessors

A view is shared and const; tree lookups need a per-thread node cache, so they go through an accessor
(`DefaultReadAccessor<BuildT>` plus a reference to the view). Recipes add semantic methods on top of the
accessor (section 11). This mirrors `ChannelAccessor`, which is an index-grid accessor plus one channel
pointer.

## 9. The owning object and the descriptor

### 9.1 Owning object

On the host, the owning object holds:

- the `GridHandle`;
- separate buffers, e.g. `cuda::Buffer<T, ResourceRef<ResourceT>>` (stream-ordered, as PropagateSign's
  outputs are), or nothing extra when an array lives in the grid's blind data;
- the descriptor.

It allocates arrays from `Enumeration::count(grid)` and `sizeof(T)`, resolves each array to a storage handle,
and hands out views. Composition at the owner level may be compile-time (typed owner per recipe) or runtime
(descriptor-driven, with views checked when taken). The view is compile-time in either case.

### 9.2 Descriptor

A runtime list of component records: name, tag, enumeration id, element type, count (per level), storage
location (blind-data index or external buffer), provenance. This is close to what `GridBlindMetaData`
already records (`mSemantic`, `mDataClass`, `mDataType`, `mValueCount`, `mName`). The descriptor enables:

- **Packing:** writing every component into the grid buffer as blind data, so the whole object is one `.nvdb`
  file. Unpacking is taking a view whose pointers point into the blind data.
- **Checked views:** `obj.view<LevelSetView<ValueOnIndex>>()` checks the records (counts, element types,
  enumerations, the canonical flag if adopted) before handing out a view.
- **Runtime callers** (Python, tools that do not know the C++ types) can inspect the components.

Likely format additions: new `GridBlindDataSemantic` values (e.g. an interior mask), and a way to mark a
blind-data array as per-node rather than per-value, which `GridBlindMetaData` cannot express today.

### 9.3 Grid identity

Every sidecar is valid only for the grid buffer it was computed against (and per-node arrays, for the node
order of that buffer, 7.3). Each record stores the grid's checksum, or another identity for the buffer, and
taking a view checks it. Mismatched sidecars are the class of bug this design most invites; this check is the
defence. "Same active voxels" is not sufficient for per-node arrays.

### 9.4 Derived arrays

Some arrays are computed from others: statistics from a channel, interior masks from signs. The record of a
derived array names its sources (template tags already encode this: `tag::Stats<tag::Sdf>`), so the
descriptor can mark it stale when a source changes, the same way the identity check catches a mismatched
grid.

## 10. Statistics as node-indexed arrays

### 10.1 Today

An index grid with statistics stores them in the value channel itself:

- leaf statistics at `mOffset + 512..515` (`ValueIndex`, NanoVDB.h:4216) or `lastOffset() + 1..4`
  (`ValueOnIndex`, NanoVDB.h:4236), gated by a per-leaf flag bit;
- internal and root statistics as indices stored in the node's `mMinimum`/`mMaximum`/... fields.

This mixes two entity sets (voxels and nodes) in one index space:

- statistics must share the channel's element type;
- every sidecar sized `activeVoxelCount` silently assumes there are no statistics;
- statistics cannot be kept for one channel and omitted for another.

A related oddity: the grid-level `HasMinMax`/`HasAverage`/`HasStdDeviation` flags are always off on index
grids from `createNanoGrid`, yet `tools/cuda/IndexToGrid.cuh` gates copying leaf statistics on
`srcGrid.hasMinMax()` (e.g. :202), so it never copies them.

### 10.2 Proposed

Statistics move out of the value channel into a separate node-indexed array. They describe nodes, not
voxels, so they use the node-id enumeration (category 1 in 7.6): nothing stored, nothing to check beyond each
level's array having that level's node count.

```cpp
template<typename ValueT>
struct Stats { ValueT min, max; FloatType<ValueT> avg, std; };   // avg/std in FloatType, as in NanoVDB's leaf data

Attr<Nodes<Leaf, Lower, Upper, Root>, Stats<float>, tag::Stats<tag::Sdf>>
```

- **One array per level, any subset of levels.** Leaf statistics only is `Nodes<Leaf>`.
- **Grid-wide statistics are the root's entry** (the root level has one node).
- **Definition:** a node's statistics summarize the active values in its whole subtree, not just its own
  slots.
- **Partial statistics** use separate tags (`tag::MinMax<Sdf>`, `tag::Moments<Sdf>`) rather than presence
  flags, so presence is in the type. This replaces the `HasMinMax`/`HasAverage` grid flags for index grids.
- **Linked to the channel** through the template tag, so the descriptor can mark statistics stale.
- **Computed bottom-up:** a reduction from each level to the next. A lower node's child is found by pointer
  subtraction too, so the pass needs no extra structure.

### 10.3 Consequences

- **The value channel loses its gaps.** Statistics were the only reason for them, so new grids' active-voxel
  offsets are canonical in practice. The free-form exception (7.7) remains for batching and old grids.
- **Migration.** One pass over an old grid reads each leaf's four statistics and the internal nodes'
  statistic indices, and writes them to node arrays. The `IndexToGrid` gating above is resolved in the same
  change.
- **Creation defaults.** The planned cleanup of `createNanoGrid`'s index-grid defaults (`includeStats = true`,
  tools/CreateNanoGrid.h:191) becomes: build index grids without statistics in the channel, and compute
  statistics arrays separately when asked.

## 11. Recipes: narrow-band level set and level set

### 11.1 What a `FloatGrid` level set carries

| In a `FloatGrid` level set | In the narrow-band recipe |
|---|---|
| SDF values on active voxels | `Attr<ActiveVoxels, float, tag::Sdf>` |
| background (= half-width x voxel size) | `Attr<Global, float, tag::Background>`, or `sdf[0]` |
| `GridClass::LevelSet`, transform, voxel size | the index grid's header (transform), plus a semantic marking the object as a level set (11.4) |
| per-node min/max/avg/std | optional: `Attr<Nodes<...>, Stats<float>, tag::Stats<tag::Sdf>>` |
| sign of inactive voxels and tiles (after `signedFloodFill`) | dropped; the interior component adds it back |

A narrow-band level set therefore holds strictly less than a flood-filled `FloatGrid`: off the band it
cannot tell inside from outside. Its API says so.

### 11.2 Narrow-band level set

```cpp
using ActiveVoxels = Compact<Slots<Leaf>, Active>;

template<typename ValueT = float, typename BuildT = ValueOnIndex>
struct NarrowBandLevelSetView
    : GridView<BuildT,
               Attr<ActiveVoxels, ValueT, tag::Sdf>,
               Attr<Global,       ValueT, tag::Background>>
{
    __hostdev__ auto getAccessor() const;
};

// accessor API: in-band queries only
__hostdev__ bool   isInBand(const Coord& ijk);                // ijk is an active voxel
__hostdev__ bool   probeValue(const Coord& ijk, ValueT& v);   // true: v = SDF; false: v = background, sign unknown
__hostdev__ ValueT background() const;
```

There is deliberately no plain `getValue` returning `+background` off the band: it would silently claim
that everything off the band is outside.

### 11.3 Level set

Adding the interior component makes a full `getValue` possible:

```cpp
template<typename ValueT = float, typename BuildT = ValueOnIndex>
struct LevelSetView
    : GridView<BuildT,
               Attr<ActiveVoxels,                      ValueT, tag::Sdf>,
               Attr<Global,                            ValueT, tag::Background>,
               Attr<Interior,                          bool,   tag::Interior>>   // Packed node levels + RootKeys
{
    __hostdev__ auto getAccessor() const;
};

// accessor API, in addition to the narrow-band one
__hostdev__ ValueT getValue(const Coord& ijk);   // active: SDF; otherwise -background inside, +background outside
__hostdev__ bool   isInside(const Coord& ijk);   // from the interior bit at the level where the lookup stopped
```

`getValue` is written once, here, instead of in every caller of PropagateSign. The two recipes share the
accessor code; the level set is the narrow-band set plus one component, which is the modular idea in its
smallest form.

### 11.4 Building one from what exists today

- **Everything in one buffer:** `createNanoGrid<SrcGridT, ValueOnIndex>(src, /*channels=*/1,
  /*includeStats=*/false, /*includeTiles=*/false)` produces the topology plus the SDF channel as blind data.
- **Separate SDF buffer:** `channels = 0`, with the SDF array in its own device buffer (e.g. from the CUDA
  tools).

Both give the same view; only the owning object differs.

Marking the object as a level set: `ChannelAccessor` asserts `gridClass() == GridClass::IndexGrid`, so the
grid header cannot carry `GridClass::LevelSet`. The semantic belongs in the descriptor. NanoVDB already has
`GridBlindDataSemantic::LevelSet` (NanoVDB.h:422 onward), and `toGridClass` maps it to `GridClass::LevelSet`
(NanoVDB.h:452), so tagging the SDF channel with it needs no format change.

To confirm before relying on it: that a `ValueOnIndex` build with `includeTiles = false` stores exactly the
active voxel values plus slot 0 in each channel.

## 12. How tools plug in

Tools declare the components they consume and produce. A recipe is complete once all its components are
attached.

- **PropagateSign** consumes the topology and a sign array over active voxels, and produces the interior
  component (`Packed<Slots<Leaf, Lower, Upper>>` plus `RootKeys`), instead of returning a tuple of four
  buffers.
- **A statistics tool** consumes a channel and produces `tag::Stats<Channel>` over the requested levels.
- **Consumers** (ray marchers, samplers, mesh extraction) take a view and `static_assert` the tags they need
  (section 6).

## 13. What the design relies on in today's NanoVDB

| Fact | Where |
|---|---|
| Nodes of each level are contiguous; `getFirstNode<LEVEL>()` returns the first | NanoVDB.h:2578-2617 |
| `ValueOnIndex` leaves return index 0 for inactive voxels (the background slot) | NanoVDB.h:4245 |
| Index-grid leaves rank active voxels with `mOffset` plus a prefix sum | `LeafIndexBase`, NanoVDB.h:4160 onward; rank code in `LeafData<ValueOnIndex>::getValue`, :4240 |
| PNanoVDB hard-codes the leaf layout: bbox at byte 0, value mask at byte 16 | PNanoVDB.h:1830 |
| Leaf statistics are interleaved in the value index space | NanoVDB.h:4216 (`ValueIndex`), :4236 (`ValueOnIndex`) |
| The root's tiles are stored sorted | NanoVDB.h:3219 |
| `ChannelAccessor`: index-grid accessor plus one channel, from blind data or an external pointer | NanoVDB.h:5884 |
| `GridBlindDataSemantic::LevelSet` exists and maps to `GridClass::LevelSet` | NanoVDB.h:422, :452 |
| `createNanoGrid` for index grids: `channels`, `includeStats` (default true), `includeTiles` | tools/CreateNanoGrid.h:189 |
| `IndexToGrid` gates statistics on `hasMinMax()` | tools/cuda/IndexToGrid.cuh:202 |

## 14. Prior art

| Source | What it contributes |
|---|---|
| Houdini attribute classes (point, vertex, primitive, detail) | entity class per attribute; *detail* is `Global` |
| OpenMesh property handles, CGAL `Surface_mesh` property maps | (entity kind, element type, name) per property, stored as its own array: the `Attr` triple |
| USD primvars | name, type, *interpolation* (constant, uniform, vertex, varying, faceVarying): interpolation is the enumeration; element counts are checked against the topology |
| USD indexed primvars (`primvars:foo:indices`) | values stored compactly through an index array; here, lookup tables are modelled as data (7.1), and compact storage as `Compact` enumerations |
| USD typed, applied and multiple-apply API schemas | recipes, add-on components, and the same component applied several times (`tag::Stats<...>`) |
| `UsdVolVolume` with named fields pointing at `UsdVolOpenVDBAsset` | a named bundle whose parts live in external storage; schema separate from storage location |
| `std::mdspan` layout policies, `Kokkos::View<T, Layout, MemorySpace>` | a non-owning, trivially copyable view parameterized by an index-to-offset mapping: the enumeration's role |
| NanoVDB `GridHandle` / `NanoGrid*`, Thrust `device_vector` / raw pointers | owner versus view |
| fVDB GridBatch with JaggedTensor features | topology plus separate arrays over active voxels |
| OpenVDB points `AttributeSet::Descriptor`, NanoVDB `GridBlindMetaData` | a runtime schema kept apart from the data |
| Apache Arrow | schema plus typed columns; booleans stored as packed bits |
| Entity-component systems (EnTT, flecs, Unity DOTS) | typed component arrays found by type at compile time |

What has no direct precedent: in meshes and entity-component systems the entities are flat and their ids are
given. Here the entities form a hierarchy, and ids are derived from node addresses or from a rank
structure. The `Location` and multi-level enumerations are new, and the least supported by prior experience.

USD differs in two ways worth noting: it is typed at runtime (no compile-time view), and its entity sets are
flat and fixed (five interpolation modes), where ours are hierarchical and open-ended. Keeping the descriptor
close to USD's model would make it straightforward to write a bundle out as a `UsdVolVolume`, with components
as fields or attributes.

## 15. Open questions

1. **Names** for the owning object and the view. Suggested: `GridBundle` (owns the bytes) and `GridView`
   (kernel side); "wrapper" was rejected as suggesting one wrapped thing. The enumeration names are settled
   in 7.9.
2. **Offset properties** (7.9.3): how a rank source records its rung (in range, no overlaps, canonical), and
   how a tool states the rung it requires. This generalizes the canonical flag of 7.7.
3. **Slot 0**: whether `Index<Leaf, BuiltinActive>` includes the background slot (7.7).
4. **Owner composition**: compile-time typed owners per recipe, or a runtime descriptor-driven owner.
5. **Which rank sources are first-class**: start with the built-in leaf source and leaf layers; add
   internal-level `RankBlock` sources when a tool needs sparse values at an internal level (7.9.3).
6. **Level-dependent element types** (7.9.2): how `NodeMask`-style families are declared and resolved per
   level. (Per-level counts are settled: each level is indexed separately.)
7. **`Global` storage**: the view holds a pointer (no host sync when building a view; works for values
   computed on the device; can alias `sdf[0]`) or the value itself (a kernel parameter). Leaning towards a
   pointer, with an optional by-value variant.
8. **Format additions**: new `GridBlindDataSemantic` values, and marking a blind-data array as per-node.
9. **Grid identity**: checksum, or a cheaper buffer identity, for the check in 9.3.
10. **Batching**: a per-grid base offset into shared arrays (7.7), and how it interacts with canonical ranks.
11. **Index layers**: whether canonical layers persist their offsets or rebuild them on load. (7.9.3 now
    prefers `LeafData<ValueOnIndex>` records for layers, which makes 7.8's shared `rankOf` function
    unnecessary.)
12. **Sanitized roots** (7.9.5): verify that inserting interior root tiles is the cheap rebuild it appears to
    be, decide where it lives, and whether PropagateSign should later accept sanitized inputs.

## 16. Suggested first prototype

A `NarrowBandLevelSetView<float, ValueOnIndex>` and a `LevelSetView`, built from a `GridHandle`, an SDF buffer
and PropagateSign's interior outputs, plus a leaf-and-lower `Stats` array computed from the SDF channel. This
exercises every enumeration class the level set needs: `Index<Leaf, BuiltinActive>` (the SDF), `Node` records
(the statistics, and the interior masks with a level-dependent `NodeMask` element), and the root table (interior
root tiles, under the sanitized convention of 7.9.5). Check `LevelSetView::getValue` against
`signedFloodFill` on the equivalent `FloatGrid`.

## 17. Note: growable index grids

A blue-sky idea (discussed with Ken) that the design should stay compatible with. Each leaf's range in the
value arrays is provisioned with spare slots after its active voxels (e.g. 8 per leaf), so voxels can later be
activated within existing leaves without rebuilding the arrays. The spare slots have no coordinate and are
never reached through the tree.

How it fits so far:

- Spare slots are gaps in the active-voxel mapping, which the free-form exception for active voxels allows
  (7.7). Such a grid is a valid index grid, but not canonical: after insertions a leaf's capacity depends on
  its history, so the offsets are real data, not a cache.
- Statistics must first move out of the value channel (section 10): today they sit right after a leaf's last
  active value, where the spare slots would go.
- A leaf's capacity is `offset[id+1] - offset[id]` while ranges stay in leaf order, so nothing new is stored.
- Inserting a voxel shifts the later voxels of its leaf by one slot (rank follows bit order) and rewrites the
  leaf's `mPrefixSum`. The work stays within the leaf, but must be applied to every array over active voxels.
  Only the bundle's descriptor knows all of them.
- Tools that treat a channel as a flat array of `activeVoxelCount` values break; the canonical flag (7.7)
  lets them refuse such a grid when they take a view.
- Additional index layers (7.8) could grow the same way, with spare slots in each leaf's range. Their
  offsets would then become real data too, so a growable layer would be a second exception to the rule in
  7.7, to be decided if this is pursued.
- Open: what to do when a leaf runs out of spare slots (rebuild all offsets, or relocate the leaf's range and
  store per-leaf capacities). New leaves are a topology rebuild, which invalidates every per-node array.
  On the GPU, insertions would be batched per leaf.
