Pod::Spec.new do |spec|
  spec.name         = "FluidAudio"
  spec.version      = "0.6.1"
  spec.summary      = "Speaker diarization, voice-activity-detection and transcription with CoreML"
  spec.description  = <<-DESC
                       Fluid Audio is a Swift SDK for fully local, low-latency audio AI on Apple devices,
                       with inference offloaded to the Apple Neural Engine (ANE). The SDK includes
                       state-of-the-art speaker diarization, transcription, and voice activity detection
                       via open-source models that can be integrated with just a few lines of code.
                       DESC
  
  spec.homepage     = "https://github.com/FluidInference/FluidAudio"
  spec.license      = { :type => "MIT", :file => "LICENSE" }
  spec.author       = { "FluidInference" => "info@fluidinference.com" }
  
  spec.ios.deployment_target = "17.0"
  spec.osx.deployment_target = "14.0"
  
  spec.source       = { :git => "https://github.com/FluidInference/FluidAudio.git", :tag => "v#{spec.version}" }
  spec.source_files = "Sources/FluidAudio/**/*.swift"

  # iOS Configuration
  # Exclude TTS module from iOS builds to avoid ESpeakNG xcframework linking issues.
  # CocoaPods has known limitations with vendored xcframeworks during pod lib lint on iOS:
  # the framework symbols aren't properly linked in the temporary build environment,
  # causing "Undefined symbols" linker errors even though the binary is valid.
  # iOS builds include: ASR (speech recognition), Diarization, and VAD (voice activity detection).
  spec.ios.exclude_files = "Sources/FluidAudio/TextToSpeech/**/*"
  spec.ios.frameworks = "CoreML", "AVFoundation", "Accelerate", "UIKit"

  # macOS Configuration
  # ESpeakNG framework is only vendored for macOS in the podspec (not a framework limitation).
  # The xcframework supports iOS, but CocoaPods fails to link it during iOS validation.
  # This enables TTS (text-to-speech) functionality with G2P (grapheme-to-phoneme) conversion.
  # macOS builds include: ASR, Diarization, VAD, and TTS with ESpeakNG support.
  spec.osx.vendored_frameworks = "Sources/FluidAudio/Frameworks/ESpeakNG.xcframework"
  spec.osx.frameworks = "CoreML", "AVFoundation", "Accelerate", "Cocoa"

  spec.swift_versions = ["5.10"]

  # Enable module definition for proper framework imports
  spec.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES'
  }
end
