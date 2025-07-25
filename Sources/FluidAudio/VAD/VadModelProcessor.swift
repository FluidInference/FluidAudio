import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
internal struct VadModelProcessor {
    
    private let logger = Logger(subsystem: "com.fluidinfluence.vad", category: "ModelProcessor")
    private let config: VadConfig
    
    init(config: VadConfig) {
        self.config = config
    }
    
    func processSTFT(_ audioChunk: [Float], stftModel: MLModel) throws -> MLMultiArray {
        let audioArray = try MLMultiArray(shape: [1, NSNumber(value: config.chunkSize)], dataType: .float32)
        
        for i in 0..<audioChunk.count {
            audioArray[i] = NSNumber(value: audioChunk[i])
        }
        
        let input = try MLDictionaryFeatureProvider(dictionary: ["audio_input": audioArray])
        let output = try stftModel.prediction(from: input)
        
        let preferredOutputNames = ["stft_features", "features", "output"]
        
        for outputName in preferredOutputNames {
            if let stftOutput = output.featureValue(for: outputName)?.multiArrayValue {
                if config.debugMode {
                    let shape = stftOutput.shape.map { $0.intValue }
                    logger.debug("STFT output '\(outputName)' shape: \(shape)")
                }
                return stftOutput
            }
        }
        
        guard let outputName = output.featureNames.first,
              let stftOutput = output.featureValue(for: outputName)?.multiArrayValue else {
            throw VadError.modelProcessingFailed("No STFT output found")
        }
        
        if config.debugMode {
            let shape = stftOutput.shape.map { $0.intValue }
            logger.debug("STFT fallback output '\(outputName)' shape: \(shape)")
        }
        
        return stftOutput
    }
    
    func processEncoder(_ concatenatedFeatures: MLMultiArray, encoderModel: MLModel) throws -> MLMultiArray {
        if config.debugMode {
            let shape = concatenatedFeatures.shape.map { $0.intValue }
            logger.debug("Encoder input shape: \(shape)")
        }
        
        let input = try MLDictionaryFeatureProvider(dictionary: ["stft_features": concatenatedFeatures])
        let output = try encoderModel.prediction(from: input)
        
        let preferredOutputNames = ["encoder_output", "encoded_features", "output"]
        
        for outputName in preferredOutputNames {
            if let encoderOutput = output.featureValue(for: outputName)?.multiArrayValue {
                if config.debugMode {
                    let shape = encoderOutput.shape.map { $0.intValue }
                    logger.debug("Encoder output '\(outputName)' shape: \(shape)")
                }
                return encoderOutput
            }
        }
        
        if config.debugMode {
            let availableOutputs = output.featureNames.joined(separator: ", ")
            logger.warning("None of preferred output names found. Available outputs: [\(availableOutputs)]. Using first available.")
        }
        
        guard let outputName = output.featureNames.first,
              let encoderOutput = output.featureValue(for: outputName)?.multiArrayValue else {
            throw VadError.modelProcessingFailed("No encoder output found")
        }
        
        if config.debugMode {
            let shape = encoderOutput.shape.map { $0.intValue }
            logger.debug("Encoder fallback output '\(outputName)' shape: \(shape)")
        }
        
        return encoderOutput
    }
    
    func processRNN(
        _ processedFeatures: MLMultiArray,
        hState: MLMultiArray,
        cState: MLMultiArray,
        rnnModel: MLModel
    ) throws -> (rnnFeatures: MLMultiArray, hState: MLMultiArray, cState: MLMultiArray) {
        
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_features": processedFeatures,
            "h_in": hState,
            "c_in": cState
        ])
        
        let output = try rnnModel.prediction(from: input)
        
        if config.debugMode {
            logger.info("🔍 RNN Model Output Investigation:")
            for featureName in output.featureNames.sorted() {
                if let featureValue = output.featureValue(for: featureName)?.multiArrayValue {
                    let shape = featureValue.shape.map { $0.intValue }
                    logger.info("  Output '\(featureName)': shape \(shape)")
                } else {
                    logger.info("  Output '\(featureName)': non-MLMultiArray type")
                }
            }
        }
        
        var rnnFeatures: MLMultiArray?
        var newHState: MLMultiArray?
        var newCState: MLMultiArray?
        
        rnnFeatures = output.featureValue(for: "rnn_features")?.multiArrayValue
        newHState = output.featureValue(for: "h_out")?.multiArrayValue
        newCState = output.featureValue(for: "c_out")?.multiArrayValue
        
        if rnnFeatures == nil || newHState == nil || newCState == nil {
            if config.debugMode {
                logger.info("⚠️ Expected output names not found, discovering actual names...")
            }
            
            for featureName in output.featureNames {
                if let featureValue = output.featureValue(for: featureName)?.multiArrayValue {
                    let shape = featureValue.shape.map { $0.intValue }
                    
                    if rnnFeatures == nil && shape.count == 3 && shape[1] > 1 {
                        rnnFeatures = featureValue
                        if config.debugMode {
                            logger.info("✓ Found sequence output: '\(featureName)' shape: \(shape)")
                        }
                    } else if shape == [1, 1, 128] {
                        let nameLower = featureName.lowercased()
                        if newHState == nil && (nameLower.contains("h") || nameLower.contains("hidden")) {
                            newHState = featureValue
                            if config.debugMode {
                                logger.info("✓ Found h_state output: '\(featureName)' shape: \(shape)")
                            }
                        } else if newCState == nil && (nameLower.contains("c") || nameLower.contains("cell")) {
                            newCState = featureValue
                            if config.debugMode {
                                logger.info("✓ Found c_state output: '\(featureName)' shape: \(shape)")
                            }
                        } else if newHState == nil {
                            newHState = featureValue
                            if config.debugMode {
                                logger.info("✓ Found h_state output (fallback): '\(featureName)' shape: \(shape)")
                            }
                        } else if newCState == nil {
                            newCState = featureValue
                            if config.debugMode {
                                logger.info("✓ Found c_state output (fallback): '\(featureName)' shape: \(shape)")
                            }
                        }
                    }
                }
            }
        }
        
        guard let finalRnnFeatures = rnnFeatures else {
            throw VadError.modelProcessingFailed("No RNN sequence output found")
        }
        guard let finalHState = newHState else {
            throw VadError.modelProcessingFailed("No h_state output found")
        }
        guard let finalCState = newCState else {
            throw VadError.modelProcessingFailed("No c_state output found")
        }
        
        return (finalRnnFeatures, finalHState, finalCState)
    }
    
    func prepareEncoderFeaturesForRNN(_ encoderFeatures: MLMultiArray) throws -> MLMultiArray {
        let shape = encoderFeatures.shape.map { $0.intValue }
        
        if shape.count == 3 && shape[0] == 1 && shape[1] == 4 && shape[2] == 128 {
            return encoderFeatures
        }
        
        if config.debugMode {
            logger.debug("Encoder features have unexpected shape: \(shape), reshaping to (1, 4, 128)")
        }
        
        let targetArray = try MLMultiArray(shape: [1, 4, 128], dataType: .float32)
        
        if shape.count == 2 && shape[0] == 1 {
            let featureSize = min(shape[1], 128)
            for timeStep in 0..<4 {
                for feature in 0..<featureSize {
                    let sourceIndex = feature
                    let targetIndex = timeStep * 128 + feature
                    if sourceIndex < encoderFeatures.count && targetIndex < targetArray.count {
                        targetArray[targetIndex] = encoderFeatures[sourceIndex]
                    }
                }
            }
        } else {
            logger.warning("Unexpected encoder shape \(shape), using linear copy fallback")
            let sourceElements = min(encoderFeatures.count, targetArray.count)
            for i in 0..<sourceElements {
                targetArray[i] = encoderFeatures[i]
            }
        }
        
        return targetArray
    }
    
    func concatenateTemporalFeatures(_ featureBuffer: [MLMultiArray]) throws -> MLMultiArray {
        guard featureBuffer.count == 4 else {
            throw VadError.modelProcessingFailed("Feature buffer must contain exactly 4 frames")
        }
        
        let singleShape = featureBuffer[0].shape.map { $0.intValue }
        guard singleShape.count >= 2 else {
            throw VadError.modelProcessingFailed("Invalid feature shape")
        }
        
        let batchSize = singleShape[0]
        let featureSize = singleShape[1]
        let temporalFrames = 4
        
        let concatenatedArray = try MLMultiArray(
            shape: [NSNumber(value: batchSize), NSNumber(value: featureSize), NSNumber(value: temporalFrames)],
            dataType: .float32
        )
        
        for frameIndex in 0..<temporalFrames {
            let frameFeatures = featureBuffer[frameIndex]
            
            for i in 0..<batchSize {
                for j in 0..<featureSize {
                    let sourceIndex = i * featureSize + j
                    let targetIndex = i * (featureSize * temporalFrames) + j * temporalFrames + frameIndex
                    concatenatedArray[targetIndex] = frameFeatures[sourceIndex]
                }
            }
        }
        
        return concatenatedArray
    }
}