# bary_tool

commandline tool, convert versions etc.

`bary_tool <filename> ... commands ...`

**Supported versions**
- v100

**List of commands**
- `-appendonsave <filename>`: appends another file when save is triggered (only standard properties)
- `-save <filename>`: saves new file, but only preserves standard properties.

**Example output**
```
input file: test_c.bary
Version:  100

File properties
  property 0: eGroups
  property 1: eTriangles
  property 2: eValues
  property 3: eTriangleMinMaxs
  property 4: eHistogramEntries
  property 5: eGroupHistograms
  property 6: eMeshDisplacementDirectionBounds

Validation
  property eGroups
  property eTriangles
  property eValues
  property eTriangleMinMaxs
  property eHistogramEntries
  property eGroupHistograms
  property eMeshDisplacementDirectionBounds
  all passed

Globals
  triangles                  6
  groups                     1

ValueInfo
  valueCount               384
  valueByteSize              1
  valueByteAlign           128
  valueFormat       eDispC1_r11_unorm_block
  valueLayout       eTriangleBirdCurve
  valueFrequency    ePerTriangle

Group 0:
  triangleCount              6
  triangleFirst              0
  valueCount               384
  valueFirst                 0
  minSubdivLevel             3
  maxSubdivLevel             3
  bias  {0.000000, 0.000000, 0.000000, 0.000000}
  scale {1.000000, 0.000000, 0.000000, 0.000000}

  computed primitive subdivlevel histogram:
    subdiv  0:         0
    subdiv  1:         0
    subdiv  2:         0
    subdiv  3:         6
  computed primitive blockformat histogram:
    blockformat  1:         6
    blockformat  2:         0
    blockformat  3:         0
  file property block format histogram:
    subdiv  3 blockformat  1:         6
```

