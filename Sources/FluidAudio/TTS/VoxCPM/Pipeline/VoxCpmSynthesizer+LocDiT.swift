@preconcurrency import CoreML
import Foundation

extension VoxCpmSynthesizer {

    /// Run LocDiT diffusion to generate a latent patch.
    ///
    /// Implements 10-step backward Euler solver (t: 1.0 → 0.001) with
    /// classifier-free guidance (batch=2: conditioned + unconditioned).
    ///
    /// - Parameters:
    ///   - mu: [1, 1024] combined dit_hidden (lm_to_dit_proj(fsq) + res_to_dit_proj(res))
    ///   - cond: [1, 64, 4] prefix conditioning features (last generated patch)
    ///   - model: locdit_step MLModel
    /// - Returns: [1, 64, 4] generated latent patch
    static func runLocDiT(
        mu: MLMultiArray,
        cond: MLMultiArray,
        model: MLModel
    ) throws -> MLMultiArray {
        let patchSize = VoxCpmConstants.patchSize
        let featDim = VoxCpmConstants.featDim
        let nTimesteps = VoxCpmConstants.defaultTimesteps
        let cfgValue = VoxCpmConstants.defaultCfgValue

        // Initialize random noise [1, 64, 4]
        let noiseShape: [NSNumber] = [1, NSNumber(value: featDim), NSNumber(value: patchSize)]
        let noise = try MLMultiArray(shape: noiseShape, dataType: .float32)
        let noisePtr = noise.dataPointer.bindMemory(to: Float.self, capacity: featDim * patchSize)
        for i in 0..<(featDim * patchSize) {
            noisePtr[i] = gaussianRandom()
        }

        // Build batch=2 for CFG [conditioned, unconditioned]
        let batchShape: [NSNumber] = [2, NSNumber(value: featDim), NSNumber(value: patchSize)]
        let xBatch = try MLMultiArray(shape: batchShape, dataType: .float32)
        let muBatchShape: [NSNumber] = [2, NSNumber(value: VoxCpmConstants.hiddenSize)]
        let muBatch = try MLMultiArray(shape: muBatchShape, dataType: .float32)
        let condBatch = try MLMultiArray(shape: batchShape, dataType: .float32)

        let xPtr = xBatch.dataPointer.bindMemory(
            to: Float.self, capacity: 2 * featDim * patchSize)
        let muPtr = muBatch.dataPointer.bindMemory(
            to: Float.self, capacity: 2 * VoxCpmConstants.hiddenSize)
        let condPtr = condBatch.dataPointer.bindMemory(
            to: Float.self, capacity: 2 * featDim * patchSize)

        let muSrc = mu.dataPointer.bindMemory(to: Float.self, capacity: VoxCpmConstants.hiddenSize)
        let condSrc = cond.dataPointer.bindMemory(to: Float.self, capacity: featDim * patchSize)

        // Copy noise to both batch items
        for i in 0..<(featDim * patchSize) {
            xPtr[i] = noisePtr[i]
            xPtr[featDim * patchSize + i] = noisePtr[i]
        }
        // mu: [conditioned, zeros]
        for i in 0..<VoxCpmConstants.hiddenSize {
            muPtr[i] = muSrc[i]
            muPtr[VoxCpmConstants.hiddenSize + i] = 0
        }
        // cond: [cond, cond]
        for i in 0..<(featDim * patchSize) {
            condPtr[i] = condSrc[i]
            condPtr[featDim * patchSize + i] = condSrc[i]
        }

        // Time schedule: backward from 1.0 to 0.001
        var tSpan = [Float](repeating: 0, count: nTimesteps + 1)
        for i in 0...nTimesteps {
            tSpan[i] = 1.0 - Float(i) * (1.0 - 1e-3) / Float(nTimesteps)
        }

        let tArray = try MLMultiArray(shape: [2], dataType: .float32)
        let dtArray = try MLMultiArray(shape: [2], dataType: .float32)
        let tPtr = tArray.dataPointer.bindMemory(to: Float.self, capacity: 2)
        let dtPtr = dtArray.dataPointer.bindMemory(to: Float.self, capacity: 2)
        dtPtr[0] = 0
        dtPtr[1] = 0

        // Euler solver loop
        for step in 0..<nTimesteps {
            let tVal = tSpan[step]
            let stepDt = tSpan[step] - tSpan[step + 1]
            tPtr[0] = tVal
            tPtr[1] = tVal

            let inputDict = try MLDictionaryFeatureProvider(dictionary: [
                LocDiTKeys.x: MLFeatureValue(multiArray: xBatch),
                LocDiTKeys.mu: MLFeatureValue(multiArray: muBatch),
                LocDiTKeys.t: MLFeatureValue(multiArray: tArray),
                LocDiTKeys.cond: MLFeatureValue(multiArray: condBatch),
                LocDiTKeys.dt: MLFeatureValue(multiArray: dtArray),
            ])

            let pred = try model.prediction(from: inputDict)
            let velocity = try ensureFloat32(
                pred.featureValue(for: LocDiTKeys.velocity)!.multiArrayValue!)
            let velPtr = velocity.dataPointer.bindMemory(
                to: Float.self, capacity: 2 * featDim * patchSize)

            // CFG: v = v_uncond + cfg * (v_cond - v_uncond)
            // Then apply backward Euler: x = x - dt * v
            let elemCount = featDim * patchSize
            for i in 0..<elemCount {
                let vCond = velPtr[i]
                let vUncond = velPtr[elemCount + i]
                let v = vUncond + cfgValue * (vCond - vUncond)
                let xNew = xPtr[i] - stepDt * v
                xPtr[i] = xNew
                xPtr[elemCount + i] = xNew
            }
        }

        // Extract result [1, 64, 4] from batch item 0
        let result = try MLMultiArray(shape: noiseShape, dataType: .float32)
        let resPtr = result.dataPointer.bindMemory(to: Float.self, capacity: featDim * patchSize)
        for i in 0..<(featDim * patchSize) {
            resPtr[i] = xPtr[i]
        }

        return result
    }

    /// Box-Muller transform for Gaussian random number generation.
    private static func gaussianRandom() -> Float {
        let u1 = Float.random(in: Float.leastNormalMagnitude...1.0)
        let u2 = Float.random(in: 0.0...1.0)
        return sqrtf(-2.0 * logf(u1)) * cosf(2.0 * .pi * u2)
    }
}
