# Semantic grids: one object for an index grid and everything that goes with it

A brief overview. Details, alternatives and open questions are in
[SemanticGrid-design.md](SemanticGrid-design.md).

## 1. Motivation

- **A `FloatGrid` keeps everything under one roof:** topology and values in a single buffer, one object to
  pass around.
- **An OnIndex grid holds only the topology.** Its values live in a sidecar, sometimes as blind data in the
  grid's buffer, more often in a separate buffer.
- **The pieces are multiplying.** Level sets will need per-node masks to tell inside from outside in
  *inactive* regions. Other uses, such as grids with statistics or further auxiliary arrays, may follow.
- **We need a semantic object,** e.g. "a level set", that bundles these pieces and is passed around as one,
  while leaving the data layout and ownership flexible.
- **Level sets should not be a privileged concept.** We want an extensible concept that accommodates them,
  with room to grow to other kinds of objects.

## 2. The abstraction: `GridView` (tentative name)

- **One object that brings the topology together with the data that complements it.**
- **It is not the storage.** The data can live anywhere: blind data in the grid's buffer, separate buffers,
  or a mix. `GridView` doesn't own or lay out that data; it is the pathway that integrates it.
- **It defines collective functionality** that needs several pieces at once, e.g. a level set's value at an
  inactive voxel, which combines the topology, the sidecar and the inside/outside masks.
- **It is the single object to pass around,** including into kernels.

```cpp
template<typename BuildT, typename... Attributes>
struct GridView;
```

- `BuildT` is the topology's build type, e.g. `ValueOnIndex`.
- `Attributes` are the species of data that complement the topology. The sidecar is the most obvious one;
  others follow in the next sections.

## 3. Attributes: what complements the topology (examples)

- **Value sidecars,** the familiar case: one value per active voxel (an SDF, a fluid velocity or density,
  a feature vector as in fVDB).
- **Per-node masks, at any level of the tree.** Level sets use them to mark inactive regions as interior:
  inactive voxels in leaves, and tiles in lower and upper nodes.
- **Data we used to keep elsewhere.** A `FloatGrid` stores per-node statistics inside its nodes, and index
  grids park them in the value sidecar. Stats might be better kept as a per-node attribute of their own.
- **Globals (uniforms):** background values (a level set could even have two, one inside and one outside),
  transforms, and other per-object parameters.

Each attribute is one kind of data with a name and a type, attached to a kind of entity: active voxels,
nodes, or the object as a whole.

## 4. The signature of an attribute

```cpp
template<typename EnumerationT, typename T, typename TagT>
struct Attribute;
```

- `EnumerationT`: **what the attribute's data is indexed over**, e.g. the active voxels, the nodes of some
  levels of the tree, or the object as a whole. The options are the subject of the next section.
- `T`: **the data type** of each element, e.g. `float`, `Vec3f`, a statistics struct, a mask.
- `TagT`: **a semantic tag** naming the attribute, e.g. `tag::Sdf`. It tells apart attributes that share
  an enumeration and a type (an SDF and a density are both a `float` per active voxel), and lets code ask
  for an attribute by meaning.

`Attribute` is only a description: it holds no data. Example: an SDF sidecar is indexed over the active
voxels, has type `float`, and is tagged `Sdf`.

## 5. Enumerations: `Global`

- **One element for the whole object.** No indexing: there is nothing to look up.
- **The element can be any type,** including a struct that groups several parameters.
- **Examples:** a background value, or separate interior and exterior values for asymmetric saturation;
  a narrow-band width; aggregate statistics for the grid as a whole.

```cpp
template<typename ValueT>
struct Background { ValueT interior, exterior; };

template<typename ValueT, typename FloatT = float>
struct Stats { ValueT min, max; FloatT avg, std; };

// one background value
Attribute<Global, float, tag::Background>
// or: separate interior and exterior values
Attribute<Global, Background<float>, tag::Background>
// aggregate statistics of the SDF
Attribute<Global, Stats<float>, tag::Stats<tag::Sdf>>
```

## 6. Enumerations: `Node<Levels...>`

- **One element per node,** at each listed level: leaf, lower, upper, or root (where the elements are the
  root-table entries).
- **Indexed by the layout.** A node's index is its position among the nodes of its level, which NanoVDB
  stores contiguously: nothing is stored to find it, and nothing can be inconsistent. This presumes
  `grid.isSequential()`.
- **Each level is indexed separately,** but one attribute can span several levels under one tag, and its
  element type may differ per level.
- **The element can be any per-node record:** statistics, a mask over the node's slots, or even a value for
  every slot.

```cpp
// interior masks: Mask<3>, Mask<4>, Mask<5> by level
Attribute<Node<Leaf, Lower, Upper>, NodeMask, tag::Interior>
// per-node statistics of the SDF
Attribute<Node<Leaf, Lower, Upper>, Stats<float>,
          tag::Stats<tag::Sdf>>
```

## 7. Enumerations: `Index<Level, RankSource>`

- **Values for a subset of a level's slots, through stored offsets:** the value-sidecar case.
- **The rank source** provides what a `ValueOnIndex` leaf holds: `mValueMask`, `mOffset`, `mPrefixSum`.
  `BuiltinRank` uses the grid's own leaves.
- **Additional voxel layers (speculative),** e.g. the active voxels plus a one-voxel halo, would use
  `CustomRank`: the attribute also points at the layer's own `LeafData<ValueOnIndex>` records, shared by
  every attribute over that layer.
- **Offsets are laissez-faire across leaves:** gaps, and leaves in any order. Within a leaf, values are
  sequential. A rank source states whether it is canonical (no gaps, leaves in order); a tool states
  whether it needs that.

```cpp
// the SDF sidecar: ranks from the grid's own leaves
Attribute<Index<Leaf, BuiltinRank>, float, tag::Sdf>
// speculative: two attributes over one extra layer
Attribute<Index<Leaf, CustomRank>, float, tag::HaloSdf>
Attribute<Index<Leaf, CustomRank>, Vec3f, tag::HaloGrad>
```

## 8. Enumerations: `Range<Level, RangeSource>` (tentative)

- **A variable number of items per voxel,** stored contiguously: an item's position is (voxel, k).
- **Point grids do this today with a dedicated build type** (`LeafData<Point>`). Here, a point set is a
  `ValueOnIndex` grid (active voxels: those holding points) plus range records.
- **The range source is records the attribute points at** (`CustomRange`); there is no built-in one.

```cpp
// point attributes over a ValueOnIndex grid
Attribute<Range<Leaf, CustomRange>, Vec3f, tag::Position>
Attribute<Range<Leaf, CustomRange>, float, tag::Radius>
// triangle ids per voxel, with their own range records
Attribute<Range<Leaf, CustomRange>, uint32_t, tag::TriangleId>
```

`Global` and `Node` get their positions from the layout; `Index` and `Range` go through stored offsets.

## 9. Example: a level set

- **A recipe, not a special case:** a particular `GridView` with three attributes.
- **Its collective functionality** is `getValue`. For an active voxel it returns the SDF. Otherwise it uses
  the interior bit of the node where the lookup stopped (or, at the root, whether a childless root-table
  entry exists), and returns the interior or exterior background.
- **Dropping `Interior`** gives a narrow-band-only level set; **adding** `Stats` costs nothing elsewhere.

```cpp
template<typename ValueT = float>
using LevelSetView = GridView<ValueOnIndex,
    Attribute<Index<Leaf, BuiltinRank>, ValueT,
              tag::Sdf>,
    Attribute<Node<Leaf, Lower, Upper>, NodeMask,
              tag::Interior>,
    Attribute<Global, Background<ValueT>,
              tag::Background>>;
```
