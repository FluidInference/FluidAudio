import Foundation

/// The files a compiled ML Program bundle needs to pass
/// `ModelCache.validateCompiledModelLayout`: `coremldata.bin`, a `model.mil`
/// that names its weights, and the weights (issue #1147). Filesystem shape
/// only; no model is created or loaded.
enum CompiledBundleFixture {
    /// A `model.mil` in the shape Core ML compiles: a program whose one constant
    /// is read from `weights/weight.bin` through `@model_path/`.
    static let programText = """
        program(1.0)
        {
            func main<ios17>(tensor<fp16, [1]> x) {
                tensor<fp16, [1]> w = const()[name = string("w"), val = tensor<fp16, [1]>(BLOBFILE(\
        path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
            } -> (w);
        }

        """

    /// Writes `model.mil`, naming `weights/weight.bin`, into `bundle`.
    static func writeProgram(in bundle: URL) throws {
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try Data(programText.utf8).write(to: bundle.appendingPathComponent("model.mil"))
    }

    /// Writes `coremldata.bin`, `model.mil` and `weights/weight.bin` into `bundle`.
    static func writeComplete(at bundle: URL) throws {
        let weights = bundle.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: bundle.appendingPathComponent("coremldata.bin"))
        try writeProgram(in: bundle)
        try Data("w".utf8).write(to: weights.appendingPathComponent("weight.bin"))
    }
}
