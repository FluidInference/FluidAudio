# FastCluster Wrapper

This directory contains a C wrapper around the [fastcluster](https://github.com/fastcluster/fastcluster) library, specifically exposing centroid linkage hierarchical clustering for Swift.

## Purpose

The FastCluster wrapper is required for accurate reimplementation of the **pyannote community-1 speaker diarization pipeline** in Swift. The pyannote pipeline uses agglomerative hierarchical clustering with centroid linkage to cluster speaker embeddings, and this wrapper provides an efficient C++ implementation via a C interface accessible from Swift.

## What's Included

- **`FastClusterWrapper.cpp`**: C wrapper implementation
- **`fastcluster_internal.hpp`**: Internal fastcluster algorithms (from upstream fastcluster)
- **`include/FastClusterWrapper.h`**: C API header
- **`include/module.modulemap`**: Swift module bridge

## Functionality

### `fastcluster_compute_centroid_linkage()`

```c
fastcluster_wrapper_status fastcluster_compute_centroid_linkage(
    const double *data,          // Feature vectors (row-major layout)
    size_t pointCount,           // Number of vectors
    size_t dimension,            // Feature dimension
    double *dendrogramOut,       // Output dendrogram (SciPy format)
    size_t dendrogramLength      // Output buffer size
);
```

Computes agglomerative hierarchical clustering using centroid linkage on the input feature vectors. Returns a dendrogram in SciPy format (4 columns: left node, right node, distance, sample count).

## Integration

Used by `Sources/FluidAudio/Diarizer/Offline/Clustering/AHCClustering.swift` to perform speaker embedding clustering, which is a core component of the diarization pipeline.

## Source

- **Original Repository**: https://github.com/fastcluster/fastcluster
- **Algorithm**: Centroid linkage hierarchical clustering
- **Reference**: Based on algorithms by Daniel Müllner and Google Inc.

## License

fastcluster is licensed under the BSD 2-Clause License. See `ThirdPartyLicenses/fastcluster-LICENSE.md` for details.

Copyright:
- Until package version 1.1.23: © 2011 Daniel Müllner
- All changes from version 1.1.24 on: © Google Inc.
