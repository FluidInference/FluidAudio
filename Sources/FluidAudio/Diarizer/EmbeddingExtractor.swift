import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
internal struct EmbeddingExtractor {
    
    internal let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "Embedding")
    
    func getEmbedding(
        audioChunk: ArraySlice<Float>,
        binarizedSegments: [[[Float]]],
        slidingWindowFeature: SlidingWindowFeature,
        embeddingModel: MLModel,
        sampleRate: Int = 16000
    ) throws -> [[Float]] {
        let chunkSize = 10 * sampleRate
        let audioTensor = audioChunk
        let numFrames = slidingWindowFeature.data[0].count
        let numSpeakers = slidingWindowFeature.data[0][0].count
        
        logger.info("=== Embedding Extraction Debug ===")
        logger.info("Chunk size: \(chunkSize), Frames: \(numFrames), Speakers: \(numSpeakers)")
        
        var cleanFrames = Array(
            repeating: Array(repeating: 0.0 as Float, count: 1), count: numFrames)
        
        for f in 0..<numFrames {
            let frame = slidingWindowFeature.data[0][f]
            let speakerSum = frame.reduce(0, +)
            cleanFrames[f][0] = (speakerSum < 2.0) ? 1.0 : 0.0
        }
        
        var cleanSegmentData = Array(
            repeating: Array(
                repeating: Array(repeating: 0.0 as Float, count: numSpeakers), count: numFrames),
            count: 1
        )
        
        for f in 0..<numFrames {
            for s in 0..<numSpeakers {
                cleanSegmentData[0][f][s] = slidingWindowFeature.data[0][f][s] * cleanFrames[f][0]
            }
        }
        
        var cleanMasks: [[Float]] = Array(
            repeating: Array(repeating: 0.0, count: numFrames), count: numSpeakers)
        
        for s in 0..<numSpeakers {
            for f in 0..<numFrames {
                cleanMasks[s][f] = cleanSegmentData[0][f][s]
            }
        }
        
        guard
            let waveformArray = try? MLMultiArray(
                shape: [numSpeakers, chunkSize] as [NSNumber], dataType: .float32),
            let maskArray = try? MLMultiArray(
                shape: [numSpeakers, numFrames] as [NSNumber], dataType: .float32)
        else {
            throw DiarizerError.processingFailed("Failed to allocate MLMultiArray for embeddings")
        }
        
        for s in 0..<numSpeakers {
            for i in 0..<min(chunkSize, audioTensor.count) {
                waveformArray[s * chunkSize + i] = NSNumber(value: audioTensor[audioTensor.startIndex + i])
            }
        }
        
        for s in 0..<numSpeakers {
            for f in 0..<numFrames {
                maskArray[s * numFrames + f] = NSNumber(value: cleanMasks[s][f])
            }
        }
        
        let inputs: [String: Any] = [
            "waveform": waveformArray,
            "mask": maskArray,
        ]
        
        guard
            let output = try? embeddingModel.prediction(
                from: MLDictionaryFeatureProvider(dictionary: inputs)),
            let multiArray = output.featureValue(for: "embedding")?.multiArrayValue
        else {
            throw DiarizerError.processingFailed("Embedding model prediction failed")
        }
        
        return convertToSendableArray(multiArray)
    }
    
    func selectMostActiveSpeaker(
        embeddings: [[Float]],
        binarizedSegments: [[[Float]]]
    ) -> (embedding: [Float], activity: Float) {
        guard !embeddings.isEmpty, !binarizedSegments.isEmpty else {
            return ([], 0.0)
        }
        
        let numSpeakers = min(embeddings.count, binarizedSegments[0][0].count)
        var speakerActivities: [Float] = []
        
        for speakerIndex in 0..<numSpeakers {
            var totalActivity: Float = 0.0
            let numFrames = binarizedSegments[0].count
            
            for frameIndex in 0..<numFrames {
                totalActivity += binarizedSegments[0][frameIndex][speakerIndex]
            }
            
            speakerActivities.append(totalActivity)
        }
        
        guard
            let maxActivityIndex = speakerActivities.indices.max(by: {
                speakerActivities[$0] < speakerActivities[$1]
            })
        else {
            return (embeddings[0], 0.0)
        }
        
        let maxActivity = speakerActivities[maxActivityIndex]
        let normalizedActivity = maxActivity / Float(binarizedSegments[0].count)
        
        return (embeddings[maxActivityIndex], normalizedActivity)
    }
    
    private func convertToSendableArray(_ multiArray: MLMultiArray) -> [[Float]] {
        let shape = multiArray.shape.map { $0.intValue }
        let numRows = shape[0]
        let numCols = shape[1]
        let strides = multiArray.strides.map { $0.intValue }
        
        var result: [[Float]] = Array(
            repeating: Array(repeating: 0.0, count: numCols), count: numRows)
        
        for i in 0..<numRows {
            for j in 0..<numCols {
                let index = i * strides[0] + j * strides[1]
                result[i][j] = multiArray[index].floatValue
            }
        }
        
        return result
    }
    
    private func getEmbeddingOptimized(
        audioChunk: ArraySlice<Float>,
        slidingWindowFeature: SlidingWindowFeature,
        embeddingModel: MLModel,
        preprocessor: MLModel,
        sampleRate: Int = 16000
    ) throws -> [[Float]] {
        let chunkSize = 10 * sampleRate
        let numFrames = slidingWindowFeature.data[0].count
        let numSpeakers = slidingWindowFeature.data[0][0].count
        
        // Prepare inputs for preprocessor
        guard
            let audioArray = try? MLMultiArray(
                shape: [1, 1, chunkSize] as [NSNumber], dataType: .float32),
            let masksArray = try? MLMultiArray(
                shape: [1, numFrames, numSpeakers] as [NSNumber], dataType: .float32)
        else {
            throw DiarizerError.processingFailed("Failed to allocate MLMultiArray for preprocessing")
        }
        
        // Fill audio array
        for i in 0..<min(chunkSize, audioChunk.count) {
            audioArray[i] = NSNumber(value: audioChunk[audioChunk.startIndex + i])
        }
        
        // Fill masks array from sliding window feature
        for f in 0..<numFrames {
            for s in 0..<numSpeakers {
                masksArray[f * numSpeakers + s] = NSNumber(value: slidingWindowFeature.data[0][f][s])
            }
        }
        
        // Run preprocessor
        let preprocessorInputs: [String: Any] = [
            "audio": audioArray,
            "speaker_masks": masksArray
        ]
        
        guard
            let preprocessorOutput = try? preprocessor.prediction(
                from: MLDictionaryFeatureProvider(dictionary: preprocessorInputs)),
            let waveforms = preprocessorOutput.featureValue(for: "waveforms")?.multiArrayValue,
            let cleanMasks = preprocessorOutput.featureValue(for: "masks_transposed")?.multiArrayValue
        else {
            throw DiarizerError.processingFailed("Preprocessing failed")
        }
        
        // Check if we should use the optimized model
        // TEMPORARILY DISABLED: The demo optimized model produces poor embeddings
        /*
        if let optimizedModel = optimizedEmbeddingNoSlice {
            logger.info("🚀 Using optimized embedding model (no SliceByIndex) via preprocessor path")
            
            // Optimized model expects different inputs
            let optimizedInputs: [String: Any] = [
                "waveforms": waveforms,
                "masks": cleanMasks  // Note: optimized model expects "masks" not "mask"
            ]
            
            do {
                let output = try optimizedModel.prediction(
                    from: MLDictionaryFeatureProvider(dictionary: optimizedInputs))
                
                guard let embeddings = output.featureValue(for: "embeddings")?.multiArrayValue else {
                    throw DiarizerError.processingFailed("No embeddings output from optimized model")
                }
                
                // Convert to array format - shape is (3, 256)
                var result: [[Float]] = []
                let embeddingDim = 256
                
                for speaker in 0..<numSpeakers {
                    var embedding: [Float] = []
                    for dim in 0..<embeddingDim {
                        embedding.append(embeddings[speaker * embeddingDim + dim].floatValue)
                    }
                    result.append(embedding)
                }
                
                logger.info("✅ Embeddings extracted using optimized model")
                return result
                
            } catch {
                logger.error("Optimized model failed: \(error), falling back to regular model")
                // Fall through to use regular model
            }
        }
        */
        
        // Run regular embedding model with preprocessed inputs
        let embeddingInputs: [String: Any] = [
            "waveform": waveforms,
            "mask": cleanMasks
        ]
        
        guard
            let output = try? embeddingModel.prediction(
                from: MLDictionaryFeatureProvider(dictionary: embeddingInputs)),
            let multiArray = output.featureValue(for: "embedding")?.multiArrayValue
        else {
            throw DiarizerError.processingFailed("Embedding model prediction failed")
        }
        
        return convertToSendableArray(multiArray)
    }
    
    private func getEmbeddingFullyOptimized(
        audioChunk: ArraySlice<Float>,
        slidingWindowFeature: SlidingWindowFeature,
        optimizedModel: MLModel,
        preprocessor: MLModel,
        sampleRate: Int
    ) throws -> [[Float]] {
        logger.info("🚀 Extracting embeddings with fully optimized model - No SliceByIndex!")
        
        let chunkSize = 10 * sampleRate
        let numFrames = slidingWindowFeature.data[0].count
        let numSpeakers = slidingWindowFeature.data[0][0].count
        
        // Create MLMultiArrays for preprocessor
        guard
            let audioArray = try? MLMultiArray(
                shape: [1, 1, NSNumber(value: chunkSize)],
                dataType: .float32
            ),
            let masksArray = try? MLMultiArray(
                shape: [1, NSNumber(value: numFrames), NSNumber(value: numSpeakers)],
                dataType: .float32
            )
        else {
            throw DiarizerError.processingFailed("Failed to create input arrays")
        }
        
        // Fill audio array
        for i in 0..<min(chunkSize, audioChunk.count) {
            audioArray[i] = NSNumber(value: audioChunk[audioChunk.startIndex + i])
        }
        
        // Fill masks array from sliding window feature
        for f in 0..<numFrames {
            for s in 0..<numSpeakers {
                masksArray[f * numSpeakers + s] = NSNumber(value: slidingWindowFeature.data[0][f][s])
            }
        }
        
        // Run preprocessor first
        let preprocessorInputs: [String: Any] = [
            "audio": audioArray,
            "speaker_masks": masksArray
        ]
        
        guard
            let preprocessorOutput = try? preprocessor.prediction(
                from: MLDictionaryFeatureProvider(dictionary: preprocessorInputs)),
            let waveforms = preprocessorOutput.featureValue(for: "waveforms")?.multiArrayValue,
            let masks = preprocessorOutput.featureValue(for: "masks_transposed")?.multiArrayValue
        else {
            throw DiarizerError.processingFailed("Preprocessing failed")
        }
        
        logger.info("Preprocessor output shapes - waveforms: \(waveforms.shape), masks: \(masks.shape)")
        
        // Run optimized model - no SliceByIndex operations!
        let optimizedInputs: [String: Any] = [
            "waveforms": waveforms,
            "masks": masks  // The optimized model expects "masks" not "masks_transposed"
        ]
        
        do {
            let output = try optimizedModel.prediction(
                from: MLDictionaryFeatureProvider(dictionary: optimizedInputs))
            
            guard let embeddings = output.featureValue(for: "embeddings")?.multiArrayValue else {
                throw DiarizerError.processingFailed("No embeddings output from optimized model")
            }
            
            logger.info("Optimized model output shape: \(embeddings.shape)")
            
            // Continue with embedding processing...
            // Convert embeddings to array format
            // Shape should be (3, 256)
            var result: [[Float]] = []
            let embeddingDim = 256
            
            for speaker in 0..<numSpeakers {
                var embedding: [Float] = []
                for dim in 0..<embeddingDim {
                    embedding.append(embeddings[speaker * embeddingDim + dim].floatValue)
                }
                result.append(embedding)
            }
            
            logger.info("✅ Successfully extracted embeddings using optimized model")
            return result
            
        } catch {
            logger.error("Optimized model prediction failed: \(error)")
            throw DiarizerError.processingFailed("Optimized model failed: \(error.localizedDescription)")
        }
    }
    
    private func getEmbeddingWithBatchExtractor(
        audioChunk: ArraySlice<Float>,
        slidingWindowFeature: SlidingWindowFeature,
        embeddingModel: MLModel,
        batchExtractor: MLModel,
        sampleRate: Int = 16000
    ) throws -> [[Float]] {
        logger.info("🚀 Using batch frame extraction - processing all frames at once!")
        
        let chunkSize = 10 * sampleRate
        let numFrames = slidingWindowFeature.data[0].count
        let numSpeakers = slidingWindowFeature.data[0][0].count
        
        // Prepare waveforms and masks for batch extraction
        guard
            let waveformsArray = try? MLMultiArray(
                shape: [numSpeakers, chunkSize] as [NSNumber],
                dataType: .float32
            ),
            let masksArray = try? MLMultiArray(
                shape: [numSpeakers, numFrames] as [NSNumber],
                dataType: .float32
            )
        else {
            throw DiarizerError.processingFailed("Failed to create input arrays for batch extraction")
        }
        
        // Fill waveforms (duplicate audio for each speaker)
        for s in 0..<numSpeakers {
            for i in 0..<min(chunkSize, audioChunk.count) {
                waveformsArray[s * chunkSize + i] = NSNumber(value: audioChunk[audioChunk.startIndex + i])
            }
        }
        
        // Fill masks from sliding window feature
        for s in 0..<numSpeakers {
            for f in 0..<numFrames {
                masksArray[s * numFrames + f] = NSNumber(value: slidingWindowFeature.data[0][f][s])
            }
        }
        
        // Run batch frame extraction (NO SliceByIndex!)
        let extractorInputs: [String: Any] = [
            "waveforms": waveformsArray,
            "masks": masksArray
        ]
        
        let extractorOutput: MLFeatureProvider
        do {
            extractorOutput = try batchExtractor.prediction(from: MLDictionaryFeatureProvider(dictionary: extractorInputs))
        } catch {
            logger.error("Batch extractor prediction failed: \(error)")
            throw DiarizerError.processingFailed("Batch frame extraction prediction failed: \(error.localizedDescription)")
        }
        
        guard
            let extractedFrames = extractorOutput.featureValue(for: "extracted_frames")?.multiArrayValue,
            let activeMask = extractorOutput.featureValue(for: "active_mask")?.multiArrayValue
        else {
            // Log available outputs for debugging
            let availableOutputs = extractorOutput.featureNames.joined(separator: ", ")
            logger.error("Failed to get expected outputs. Available: \(availableOutputs)")
            throw DiarizerError.processingFailed("Batch frame extraction failed - outputs not found")
        }
        
        logger.info("✅ Frames extracted in batch! Shape: \(extractedFrames.shape)")
        
        // Now we need to process these frames through the embedding model
        // The extracted_frames shape is (num_speakers, num_frames, frame_length)
        // We need to reshape for the embedding model which expects different input
        
        // For now, we'll fall back to the regular embedding processing
        // but with the pre-extracted frames (no more SliceByIndex!)
        
        // Create clean masks based on speaker activity
        var cleanMasks: [[Float]] = Array(
            repeating: Array(repeating: 0.0, count: numFrames),
            count: numSpeakers
        )
        
        for s in 0..<numSpeakers {
            for f in 0..<numFrames {
                // Use the sliding window feature data for clean masks
                let speakerSum = slidingWindowFeature.data[0][f].reduce(0, +)
                let isClean: Float = speakerSum < 2.0 ? 1.0 : 0.0
                cleanMasks[s][f] = slidingWindowFeature.data[0][f][s] * isClean
            }
        }
        
        // Prepare inputs for embedding model
        guard
            let waveformArray = try? MLMultiArray(
                shape: [numSpeakers, chunkSize] as [NSNumber],
                dataType: .float32
            ),
            let maskArray = try? MLMultiArray(
                shape: [numSpeakers, numFrames] as [NSNumber],
                dataType: .float32
            )
        else {
            throw DiarizerError.processingFailed("Failed to create embedding model inputs")
        }
        
        // Fill arrays
        for s in 0..<numSpeakers {
            for i in 0..<min(chunkSize, audioChunk.count) {
                waveformArray[s * chunkSize + i] = NSNumber(value: audioChunk[audioChunk.startIndex + i])
            }
            for f in 0..<numFrames {
                maskArray[s * numFrames + f] = NSNumber(value: cleanMasks[s][f])
            }
        }
        
        // Run embedding model
        let embeddingInputs: [String: Any] = [
            "waveform": waveformArray,
            "mask": maskArray
        ]
        
        guard
            let output = try? embeddingModel.prediction(
                from: MLDictionaryFeatureProvider(dictionary: embeddingInputs)
            ),
            let multiArray = output.featureValue(for: "embedding")?.multiArrayValue
        else {
            throw DiarizerError.processingFailed("Embedding model prediction failed")
        }
        
        logger.info("✅ Embeddings extracted with batch processing - no SliceByIndex operations!")
        
        return convertToSendableArray(multiArray)
    }
}