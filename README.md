# NVIDIA Displacement Micro-Map BaryFile

Repository for barycentric data fileformat '.bary'

`BARY` is an fileformat that serves as container for micromap data.

Micromaps store per-microvertex or per-microtriangle data for a micromesh
that is the result of evenly subdividing a triangle.

Each subdivided triangle contains its own set of values, there is 
no sharing of values across subdivided triangles.

```
                  V
                  4
                /  \ 
               3 __  8
             /  \  /  \ 
            2 __  7 __ 11
          /  \  /  \  /  \ 
         1 __  6 __ 10 __ 13
       /  \  /  \  /  \  /  \ 
      0 __ 5 __   9 __ 12 __ 14
     W                         U

    Result of subdivision level 2 for 
    triangle W,U,V

    16 microtriangles
    15 microvertices
```

The file format is designed to be directly uploaded to the GPU
and passing data structures into 3D APIs without additional processing.

## bary_core library

This library defines the structs and principle file structure in memory
and uses the `bary` namespace.

It comes with core API functions that operate on pointers to aid 
serialization or validation of the data. No allocations or file IO operations
are performed in this library.

All content of a bary file is expressed via properties, that are idenfitied 
through a v4 UUID.

## bary_utils library

The bary_utils library adds some utility C++ classes that leverage STL containers
and adds basic file loaders and savers using `stdio`.
It makes use of the functions in bary_core to implement many of these utilities
and uses its own namespace `baryutils`.

By default CMAKE does not build this library and you must opt-in.

## New VkFormats for barycentric micromaps

The primary use in 3D APIs for barycentric data are displacement and opacity micromaps.
Displacement can be block-compressed similar to BC/ASTC.

However, any other scalar values can be stored in bary files for exchange,
there just isn't a native 3D API use for them.

`BARY` uses VkFormat enum values, we reserved two Vk extensions (397 and 398) to get the value ranges below.

**! NOTE: These could change, rely only what is provided in the headers for now !**

``` c++
// bary::Format uses VkFormat values
enum class Format : uint32_t {
...
    // opacity encoding (based on VK NV extension reservation 397)

    // block-compressed opacity format
    // for uncompressed 1 or 2 bit data stored in 8-bit
    // valueByteSize = 1
    eOpaC1_rx_uint_block = 1000396000,

    // displacement encoding  (based on VK NV extension reservation 398)

    // block-compressed displacement format
    // for compressed data stored in blocks of 512- or 1024-bit
    // valueByteSize = 1
    // valueByteAlignment = 128
    // minmax as eR11_unorm_pack16
    eDispC1_r11_unorm_block = 1000397000,

    // for uncompressed data 1 x 11 bit stored in 16-bit
    eR11_unorm_pack16 = 1000397001,

    // variable packed format
    // for uncompressed data 1 x 11 bit densely packed sequence of 32bit values.
    // Each triangle starts at a 32-bit boundary.
    // valueByteSize = 1
    // minmax as eR11_unorm_pack16
    eR11_unorm_packed_align32 = 1000397002,
...
};

// encodes 1 or 2 bit opacity maps
enum class BlockFormatOpaC1 : uint16_t
{
    eInvalid    = 0,
    eR1_uint_x8 = 1,
    eR2_uint_x4 = 2,
};

// encodes displacement maps of unorm11 values
enum class BlockFormatDispC1 : uint16_t
{
    eInvalid                 = 0,
    eR11_unorm_lvl3_pack512  = 1,
    eR11_unorm_lvl4_pack1024 = 2,
    eR11_unorm_lvl5_pack1024 = 3,
};
```

## Principle dataflow

Here are some of the core data structures stored within bary files.

``` c++
// gives details about what values are stored
// and how they are laid out across a triangle.
struct bary::ValuesInfo
{
    Format         valueFormat;
    // spatial layout of values across the subdivided triangle
    ValueLayout    valueLayout;
    // per-vertex or per-triangle
    ValueFrequency valueFrequency;
    // how many values there are in total (or bytes for compressed / special packed formats)
    uint32_t       valueCount;
    // compressed or special packed formats must use
    // valueByteSize 1
    uint32_t       valueByteSize;
    // valueByteAlignment must be at least 4 bytes, higher alignment only
    // if it is power of two and either matching valueByteSize, or if special formats
    // demand for it. (e.g. eRG32_sfloat is 8 byte aligned, but eRGB32_sfloat 4 byte)
    uint32_t       valueByteAlignment;
}

// provides key information for every Micromap Triangle
struct bary::Triangle
{
    // valuesOffset must be ascending from t to t+1
    // and are relative to the group that this triangle belongs to
    // for uncompressed: serves as indexOffset (valueFormat agnostic)
    // for compressed / special packed: serves as byteOffset (given valueByteSize is 1 for those)
    uint32_t valuesOffset;
    // the subdivision level of this triangle, influences how much
    // values are relevant to it.
    uint16_t subdivLevel;

    // if the values are stored compressed 
    uint16_t blockFormat;
};

// groups allow to store multiple groups of independent values and triangles
// of same value info in one file.
struct bary::Group
{
    // first/count must be linear ascending and non-overlapping

    uint32_t triangleFirst;
    uint32_t triangleCount;
    uint32_t valueFirst;
    uint32_t valueCount;

    uint32_t minSubdivLevel;
    uint32_t maxSubdivLevel;

    // for UNORM,SNORM,FLOAT values these
    // represent the final value range
    // (value * scale) + bias
    ValueFloatVector floatBias;
    ValueFloatVector floatScale;
};

// typical file content can therefore be accessed via a pointer view
struct bary::BasicPropsView
{
    // mandatory for all
    const Group*      groups         = nullptr;
    uint32_t          groupsCount    = 0;
    const ValuesInfo* valuesInfo     = nullptr;
    const uint8_t*    values         = nullptr;
    const Triangle*   triangles      = nullptr;
    uint32_t          trianglesCount = 0;
    ...
};
```
The following describes their relationship:

- `mesh.`: data typically stored within the 3d mesh file loaded by application
- `bary.`: data stored within a bary file

- `mesh.triangle`: a triangle (triangle/quad) within the geometry (aka baseTriangle), is mapped to 1:
  - `bary.triangle`: a triangle within the barycentric data container, has information where actual values are stored within the bary values. Can be split into **N >= 1**:
    - `bary.blocktriangle`: when block compression is used, splitting of a `bary.triangle` into triangles of lower subdivision that each represent a compressed block.

Looking at the data we store in file:

- `mesh.triangleMapping`: index to map `mesh.triangle` to `bary.triangle` (think like UV coordinate for texture)
  The majority of the content will have unique 1:1 mapping, which allows to skip this mapping buffer. This is mostly meant for micro-instancing 
  some displacements all over within a single mesh.

- `mesh.triangleFlags`: used to generate watertight displacement for each `mesh.triangle`
  - store "half resolution" information, set the n-th bit, if the n-th edge has an adjacent `mesh.triangle` of half resolution `bary.triangle`
  - cannot be stored in the bary file, as each `mesh.triangle` could have different adjacency behahvior but map to the same `bary.triangle`.

- `bary.triangles`: links bary triangles to values.
  - subdivision level of triangle
  - offsets for values
  - block format (if all explicit or implicit subtriangles use the same block format)

- `bary.values` : raw data uploaded in one big blob to GPU, stores values in special barycentric ordering (uncompressed or compressed formats exist)


```  c++
    for baseTriangleIdx in mesh.triangles 
    {
        tri       = bary.triangles[ mesh.barytriangleMapping[ baseTriangleIdx ]  ];

        triValues = & bary.values[ tri.valuesOffset ];

        // these values are either per vertex or per triangle and stored in a canonical
        // spatial order according to the valueLayout

        for (i < computeValueCount(bary.valuesInfo.valueFrequency, tri.subdivLevel))
        {
           value =  triValues[ i ];
        }
    }
```

### Limitations

- Barycentric data is dependent on the winding of the mesh triangle. 
- Currently no support for "mirrored" barycentric data (same data used with a different `mesh.triangle` winding order) like UV textures would allow.

### Open Issues

- Modelling tools currently will not support generating a mapping table between mesh and bary container.
  Modifying such tables under topological changes during modelling operations would also be unfortunate.
  For now we focus on a unique 1:1 mapping of mesh and barycentric triangles.
  
  One relatively easy method to enhance this, is to use the existing three UV-texture coords of a mesh 
  triangle as a single key to identify which triangles should get matching mapping indices. 
  This way we can re-use relationship information available in existing files, as well as use existing 
  UV-tools. We would simply not care about UV-texture coords as such, but only for sake to identify those
  triangles that should use same barycentric data. This way an artist can clone the mesh triangles and
  the relationship, that these share identical values, would stay intact.

## Support Contact

Feel free to file issues directly on the GitHub page or reach out to NVIDIA at
<displacedmicromesh-sdk-support@nvidia.com>