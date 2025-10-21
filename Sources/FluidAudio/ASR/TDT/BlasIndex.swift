import Accelerate

#if ACCELERATE_NEW_LAPACK
typealias BlasIndex = __LAPACK_int
#else
typealias BlasIndex = Int32
#endif

@inline(__always)
func makeBlasIndex(_ value: Int, label: String) throws -> BlasIndex {
    guard let cast = BlasIndex(exactly: value) else {
        throw ASRError.processingFailed("\(label) exceeds supported range")
    }
    return cast
}
