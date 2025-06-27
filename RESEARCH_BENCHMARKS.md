# FluidAudioSwift Research Benchmarks

## 🎯 Overview

This benchmark system enables **research-standard evaluation** of your speaker diarization system using **real datasets** from academic literature. The dataset downloading and caching is **fully integrated into Swift tests** - no external scripts or Python dependencies required!

### ✅ What's Implemented

**Standard Research Datasets:**
- ✅ **AMI IHM** (Individual Headset Mics) - Clean, close-talking conditions
- ✅ **AMI SDM** (Single Distant Mic) - Realistic far-field conditions
- 🔄 **VoxConverse** (planned) - Modern "in the wild" YouTube speech
- 🔄 **CALLHOME** (planned) - Telephone conversations (if purchased)

**Research Metrics:**
- ✅ **DER** (Diarization Error Rate) - Industry standard
- ✅ **JER** (Jaccard Error Rate) - Overlap accuracy
- ✅ **Miss/False Alarm/Speaker Error** rates breakdown
- ✅ Frame-level accuracy metrics
- ✅ **EER** (Equal Error Rate) for speaker verification
- ✅ **AUC** (Area Under Curve) for ROC analysis

**Integration Features:**
- ✅ **Automatic dataset downloading** from Hugging Face
- ✅ **Smart caching** - downloads once, reuses forever
- ✅ **Native Swift** - no Python dependencies
- ✅ **Real audio files** - actual AMI Meeting Corpus segments
- ✅ **Ground truth annotations** - proper speaker labels and timestamps

**Benchmark Tests:**
- ✅ `testAMI_IHM_SegmentationBenchmark()` - Clean conditions
- ✅ `testAMI_SDM_SegmentationBenchmark()` - Far-field conditions
- ✅ `testAMI_IHM_vs_SDM_Comparison()` - Difficulty validation
- ✅ Automatic dataset download and caching
- ✅ Research baseline comparisons

## 🚀 Quick Start

### **Option 1: Real Research Benchmarks (Recommended)**
```bash
# Real AMI Meeting Corpus data - downloads automatically
swift test --filter BenchmarkTests

# Specific tests:
swift test --filter testAMI_IHM_SegmentationBenchmark  # Clean conditions
swift test --filter testAMI_SDM_SegmentationBenchmark  # Far-field conditions
swift test --filter testAMI_IHM_vs_SDM_Comparison     # Compare difficulty
```

### **Option 2: Basic Functionality Tests**
```bash
# Simple synthetic audio tests (no downloads needed)
swift test --filter SyntheticBenchmarkTests

# Just check if your system works:
swift test --filter testBasicSegmentationWithSyntheticAudio
swift test --filter testBasicEmbeddingWithSyntheticAudio
```

### **Option 3: Research-Standard Metrics**
```bash
# Advanced research metrics and evaluation
swift test --filter ResearchBenchmarkTests
```

**First run output:**
```
⬇️ Downloading AMI IHM dataset from Hugging Face...
  ✅ Downloaded sample_000.wav (180.5s, 4 speakers)
  ✅ Downloaded sample_001.wav (95.2s, 3 speakers)
  ✅ Downloaded sample_002.wav (210.8s, 4 speakers)
🎉 AMI IHM dataset ready: 3 samples, 486.5s total
```

**Subsequent runs:**
```
📁 Loading cached AMI IHM dataset
```

## 📊 **What You Get**

### **Real AMI Dataset Audio:**
- **AMI IHM**: 3 samples, ~8 minutes total, 3-4 speakers each
- **AMI SDM**: 3 samples, ~8 minutes total, same meetings but far-field
- **16kHz WAV files** saved to `./datasets/ami_ihm/` and `./datasets/ami_sdm/`
- **Ground truth annotations** with precise speaker timestamps

### **No Dependencies Required:**
- ❌ **No Python** installation needed
- ❌ **No pip packages** to install
- ❌ **No shell scripts** to run
- ✅ **Pure Swift** implementation
- ✅ **URLSession** for downloads
- ✅ **Native WAV file** creation

### 📊 Expected Results

| Test | Research Baseline | Your Target | What It Measures |
|------|------------------|-------------|------------------|
| AMI IHM | 15-25% DER | <40% DER | Clean close-talking performance |
| AMI SDM | 25-35% DER | <60% DER | Realistic far-field performance |

**Note:** Your system uses general CoreML models, so expect higher error rates than specialized research systems initially.

---

## Detailed Documentation

FluidAudioSwift includes a comprehensive benchmark system designed to evaluate segmentation and embedding performance against standard metrics used in speaker diarization research papers. This system implements evaluation metrics and test scenarios based on recent research, particularly the **"Powerset multi-class cross entropy loss for neural speaker diarization"** paper and other standard benchmarks.

## Current Implementation Features

### 1. Segmentation Benchmarks
Your CoreML implementation uses a **powerset classification approach** with 7 classes:
- `{}` (silence/empty)
- `{0}`, `{1}`, `{2}` (single speakers)
- `{0,1}`, `{0,2}`, `{1,2}` (speaker pairs)

This aligns with the methodology described in the powerset paper.

### 2. Standard Research Metrics

#### Diarization Error Rate (DER)
```swift
// DER = (False Alarm + Missed Detection + Speaker Error) / Total Speech Time
let der = calculateDiarizationErrorRate(predicted: segments, groundTruth: gtSegments)
```

#### Jaccard Error Rate (JER)
```swift
// JER = 1 - (Intersection / Union) for each speaker
let jer = calculateJaccardErrorRate(predicted: segments, groundTruth: gtSegments)
```

#### Coverage and Purity
```swift
let coverage = calculateCoverage(predicted: segments, groundTruth: gtSegments)
let purity = calculatePurity(predicted: segments, groundTruth: gtSegments)
```

### 3. Embedding Quality Metrics

#### Equal Error Rate (EER)
```swift
let eer = calculateEqualErrorRate(similarities: similarities, labels: isMatches)
```

#### Area Under Curve (AUC)
```swift
let auc = verificationResults.calculateAUC()
```

## Using the Benchmark System

### Basic Usage

```swift
import XCTest
@testable import FluidAudioSwift

// Initialize the diarization system
let config = DiarizerConfig(backend: .coreML, debugMode: true)
let manager = DiarizerFactory.createManager(config: config)

// Initialize the system (downloads models if needed)
try await manager.initialize()

// Run segmentation benchmark
let testAudio = loadAudioFile("path/to/test.wav")
let segments = try await manager.performSegmentation(testAudio, sampleRate: 16000)

// Evaluate against ground truth
let metrics = calculateResearchMetrics(
    predicted: segments,
    groundTruth: groundTruthSegments,
    datasetName: "MyDataset"
)

print("DER: \(metrics.diarizationErrorRate)%")
print("JER: \(metrics.jaccardErrorRate)%")
```

### Powerset Classification Evaluation

```swift
// Test specific powerset scenarios
let powersetTests = [
    (audio: silenceAudio, expectedClass: PowersetClass.empty),
    (audio: singleSpeakerAudio, expectedClass: PowersetClass.speaker0),
    (audio: twoSpeakerAudio, expectedClass: PowersetClass.speakers01)
]

let confusionMatrix = PowersetConfusionMatrix()

for test in powersetTests {
    let segments = try await manager.performSegmentation(test.audio, sampleRate: 16000)
    let predictedClass = determinePowersetClass(from: segments)
    confusionMatrix.addPrediction(actual: test.expectedClass, predicted: predictedClass)
}

let accuracy = confusionMatrix.calculateAccuracy()
print("Powerset Classification Accuracy: \(accuracy)%")
```

## Integrating with Real Research Datasets

### Dataset Integration Examples

#### 1. AMI Meeting Corpus
```swift
func evaluateOnAMI() async throws {
    let amiFiles = loadAMIDataset() // Your implementation
    var totalDER: Float = 0.0

    for amiFile in amiFiles {
        let audio = loadAudio(amiFile.audioPath)
        let groundTruth = loadRTTM(amiFile.rttmPath) // Load ground truth annotations

        let predictions = try await manager.performSegmentation(audio, sampleRate: 16000)
        let der = calculateDiarizationErrorRate(predicted: predictions, groundTruth: groundTruth)

        totalDER += der
        print("AMI \(amiFile.name): DER = \(der)%")
    }

    print("Average AMI DER: \(totalDER / Float(amiFiles.count))%")
}
```

#### 2. DIHARD Challenge
```swift
func evaluateOnDIHARD() async throws {
    let dihardFiles = loadDIHARDDataset()

    for file in dihardFiles {
        let metrics = try await evaluateFile(
            audioPath: file.audioPath,
            rttmPath: file.rttmPath,
            domain: file.domain // telephone, meeting, etc.
        )

        print("DIHARD \(file.domain): DER=\(metrics.der)%, JER=\(metrics.jer)%")
    }
}
```

### Custom Dataset Integration

```swift
// Example: Loading your own dataset
struct CustomDataset {
    let audioFiles: [URL]
    let annotations: [URL] // RTTM format
}

func evaluateCustomDataset(_ dataset: CustomDataset) async throws {
    var results: [String: ResearchMetrics] = [:]

    for (audioFile, annotationFile) in zip(dataset.audioFiles, dataset.annotations) {
        // Load audio
        let audio = try loadAudioFile(audioFile)

        // Load ground truth from RTTM or custom format
        let groundTruth = try parseAnnotations(annotationFile)

        // Run prediction
        let predictions = try await manager.performSegmentation(audio, sampleRate: 16000)

        // Calculate metrics
        let metrics = calculateResearchMetrics(
            predicted: predictions,
            groundTruth: groundTruth,
            datasetName: audioFile.lastPathComponent
        )

        results[audioFile.lastPathComponent] = metrics
    }

    // Report aggregate results
    reportResults(results)
}
```

## Standard Research Datasets Integration

The benchmark system supports integration with standard research datasets used in speaker diarization literature:

### Supported Datasets

#### Free Datasets (Recommended Start)
- **AMI Meeting Corpus** - 100 hours of meeting recordings
  - **IHM (Individual Headset)** - Clean close-talking mics (easiest)
  - **SDM (Single Distant Mic)** - Far-field single channel (realistic)
  - **MDM (Multiple Distant Mics)** - Microphone arrays (most challenging)
- **VoxConverse** - 64 hours of YouTube conversations (modern benchmark)
- **CHiME-5** - Multi-channel dinner party recordings (very challenging)
- **LibriSpeech** - Clean read speech (baseline comparisons)

#### Commercial Datasets (LDC)
- **CALLHOME** - $500, 17 hours telephone conversations
- **DIHARD II** - $300, 46 hours multi-domain recordings

### Quick Start with Free Data

```swift
// Start with AMI IHM (easiest)
func downloadAMIDataset() async throws {
    let amiURL = "https://huggingface.co/datasets/diarizers-community/ami"

    // Download preprocessed AMI data
    let dataset = try await HuggingFaceDataset.load(
        "diarizers-community/ami",
        subset: "ihm"  // Individual headset mics (cleanest)
    )

    return dataset
}

// Alternative: VoxConverse (modern benchmark)
func downloadVoxConverse() async throws {
    let voxURL = "https://github.com/joonson/voxconverse"
    // Download VoxConverse dataset
}
```

### Local AMI Dataset Setup

To use real AMI data instead of synthetic audio:

#### Option 1: Quick Test Setup (Recommended)
```bash
# 1. Install Python dependencies
pip install datasets librosa soundfile

# 2. Download a small subset for testing
python3 -c "
from datasets import load_dataset
import soundfile as sf
import os

# Create datasets directory
os.makedirs('./datasets/ami_ihm/test', exist_ok=True)

# Download AMI IHM test set (small subset)
dataset = load_dataset('diarizers-community/ami', 'ihm')
test_data = dataset['test']

print(f'Downloaded {len(test_data)} test samples')

# Save first 3 samples for quick testing
for i, sample in enumerate(test_data.select(range(3))):
    audio = sample['audio']['array']
    sf.write(f'./datasets/ami_ihm/test/sample_{i:03d}.wav', audio, 16000)
    print(f'Saved sample {i}')
"

# 3. Run your benchmarks
swift test --filter testAMI_IHM_SegmentationBenchmark
```

#### Option 2: Full Dataset Setup
```bash
# Download complete AMI datasets
# IHM (clean, close-talking mics)
python3 -c "
from datasets import load_dataset
dataset = load_dataset('diarizers-community/ami', 'ihm')
# Process and save locally...
"

# SDM (far-field, single distant mic)
python3 -c "
from datasets import load_dataset
dataset = load_dataset('diarizers-community/ami', 'sdm')
# Process and save locally...
"
```

#### Expected Performance Baselines

Based on research literature:

| Dataset | Variant | Research Baseline DER | Your Target |
|---------|---------|----------------------|-------------|
| AMI     | IHM     | 15-25%              | < 40%       |
| AMI     | SDM     | 25-35%              | < 60%       |

**Note:** Your system should perform worse than research baselines initially since those use specialized diarization models, while you're using general CoreML models.

### Dataset Integration Examples

```swift
// Download and prepare AMI corpus
func setupAMIDataset() async throws {
    let amiDownloader = AMICorpusDownloader()
    let amiData = try await amiDownloader.download(to: "datasets/ami/")

    // Convert AMI annotations to benchmark format
    let converter = AMIAnnotationConverter()
    let benchmarkData = try converter.convertToBenchmarkFormat(amiData)

    return benchmarkData
}

// Run benchmarks on CALLHOME (if available)
func testCALLHOMEBenchmark() async throws {
    guard let callhomeData = try? loadCALLHOMEDataset() else {
        print("⚠️ CALLHOME dataset not available - using synthetic data")
        return
    }

    let results = try await runDiarizationBenchmark(
        dataset: callhomeData,
        metrics: [.DER, .JER, .coverage, .purity]
    )

    // Compare with published results
    assertPerformanceComparison(results, publishedBaselines: .callhome2023)
}
```

### Automatic Dataset Download

```swift
class ResearchDatasetManager {
    func downloadFreeDatasets() async throws {
        // AMI Corpus
        try await downloadAMI()

        // VoxConverse
        try await downloadVoxConverse()

        // LibriSpeech samples
        try await downloadLibriSpeechSamples()
    }

    private func downloadAMI() async throws {
        let url = "https://groups.inf.ed.ac.uk/ami/corpus/"
        // Implementation for AMI download and setup
    }
}
```

## Performance Benchmarking

### Real-Time Factor (RTF) Testing
```swift
func benchmarkProcessingSpeed() async throws {
    let testFiles = [
        (duration: 10.0, name: "short_audio"),
        (duration: 60.0, name: "medium_audio"),
        (duration: 300.0, name: "long_audio")
    ]

    for test in testFiles {
        let audio = generateTestAudio(durationSeconds: test.duration)
        let startTime = CFAbsoluteTimeGetCurrent()

        let segments = try await manager.performSegmentation(audio, sampleRate: 16000)

        let processingTime = CFAbsoluteTimeGetCurrent() - startTime
        let rtf = processingTime / Double(test.duration)

        print("\(test.name): RTF = \(rtf)x")
        assert(rtf < 2.0, "Processing should be < 2x real-time")
    }
}
```

### Memory Usage Monitoring
```swift
func benchmarkMemoryUsage() async throws {
    let initialMemory = getMemoryUsage()

    // Process various audio lengths
    for duration in [10.0, 30.0, 60.0, 120.0] {
        let audio = generateTestAudio(durationSeconds: duration)
        let _ = try await manager.performSegmentation(audio, sampleRate: 16000)

        let currentMemory = getMemoryUsage()
        let memoryIncrease = currentMemory - initialMemory

        print("Duration: \(duration)s, Memory increase: \(memoryIncrease)MB")
    }
}
```

## Embedding Quality Evaluation

### Speaker Verification Testing
```swift
func evaluateEmbeddingQuality() async throws {
    let speakerPairs = createSpeakerVerificationDataset()
    var results: [(similarity: Float, isMatch: Bool)] = []

    for pair in speakerPairs {
        let similarity = try await manager.compareSpeakers(
            audio1: pair.audio1,
            audio2: pair.audio2
        )

        results.append((similarity: similarity, isMatch: pair.isMatch))
    }

    // Calculate EER
    let eer = calculateEqualErrorRate(
        similarities: results.map { $0.similarity },
        labels: results.map { $0.isMatch }
    )

    print("Speaker Verification EER: \(eer)%")
    assert(eer < 15.0, "EER should be < 15% for good embedding quality")
}
```

## Research Paper Comparisons

### Powerset Cross-Entropy Loss Paper Metrics
The current implementation can be directly compared against results from the powerset paper:

```swift
// Expected benchmark results from the paper on standard datasets:
let expectedResults = [
    "AMI": (der: 25.2, jer: 45.8),
    "DIHARD": (der: 32.1, jer: 52.3),
    "CALLHOME": (der: 20.8, jer: 38.5)
]

// Your results comparison
func compareAgainstPaperResults() async throws {
    for (dataset, expected) in expectedResults {
        let ourResult = try await evaluateOnDataset(dataset)

        print("\(dataset):")
        print("  Paper DER: \(expected.der)% | Our DER: \(ourResult.der)%")
        print("  Paper JER: \(expected.jer)% | Our JER: \(ourResult.jer)%")

        let derImprovement = expected.der - ourResult.der
        print("  DER Improvement: \(derImprovement)%")
    }
}
```

## Advanced Usage

### Ablation Studies
```swift
// Test different configuration parameters
func performAblationStudy() async throws {
    let configurations = [
        DiarizerConfig(clusteringThreshold: 0.5),
        DiarizerConfig(clusteringThreshold: 0.7),
        DiarizerConfig(clusteringThreshold: 0.9)
    ]

    for config in configurations {
        let manager = DiarizerFactory.createManager(config: config)
        try await manager.initialize()

        let metrics = try await evaluateConfiguration(manager, config)
        print("Threshold \(config.clusteringThreshold): DER = \(metrics.der)%")
    }
}
```

### Cross-Domain Evaluation
```swift
func evaluateAcrossDomains() async throws {
    let domains = ["meeting", "telephone", "broadcast", "interview"]

    for domain in domains {
        let testFiles = loadDomainFiles(domain)
        let avgMetrics = try await evaluateFiles(testFiles)

        print("\(domain.capitalized) Domain:")
        print("  Average DER: \(avgMetrics.der)%")
        print("  Average JER: \(avgMetrics.jer)%")
    }
}
```

## Integration with CI/CD

### Automated Benchmarking
```swift
// Add to your CI pipeline
func runAutomatedBenchmarks() async throws {
    let benchmarkSuite = BenchmarkSuite()

    // Add test cases
    benchmarkSuite.add(.segmentationAccuracy)
    benchmarkSuite.add(.embeddingQuality)
    benchmarkSuite.add(.processingSpeed)

    let results = try await benchmarkSuite.runAll()

    // Generate report
    let report = BenchmarkReport(results: results)
    try report.saveTo("benchmark_results.json")

    // Assert performance thresholds
    assert(results.averageDER < 30.0, "DER regression detected!")
    assert(results.averageRTF < 1.5, "Processing too slow!")
}
```

## Extending the Benchmark System

### Adding New Metrics
```swift
extension ResearchMetrics {
    func calculateFalseAlarmRate() -> Float {
        // Your implementation
    }

    func calculateMissedDetectionRate() -> Float {
        // Your implementation
    }
}
```

### Custom Test Scenarios
```swift
struct CustomBenchmarkScenario {
    let name: String
    let audioGenerator: () -> [Float]
    let groundTruthGenerator: () -> [SpeakerSegment]
    let expectedMetrics: (der: Float, jer: Float)
}

func addCustomScenario(_ scenario: CustomBenchmarkScenario) {
    // Add to benchmark suite
}
```

## Conclusion

This benchmark system provides comprehensive evaluation capabilities for your FluidAudioSwift implementation. It enables direct comparison with research papers and helps track performance improvements over time. The modular design allows easy extension for new metrics and test scenarios as the field evolves.

### Key Benefits:
1. **Research Alignment**: Direct comparison with published papers
2. **Regression Testing**: Catch performance degradations
3. **Configuration Optimization**: Find best parameters for your use case
4. **Quality Assurance**: Ensure consistent performance across updates
5. **Publication Ready**: Generate metrics suitable for research papers

For questions or contributions to the benchmark system, please refer to the main FluidAudioSwift documentation.
