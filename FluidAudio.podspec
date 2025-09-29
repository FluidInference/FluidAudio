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
  
  spec.swift_versions = ["5.10"]
  spec.frameworks = "CoreML", "AVFoundation", "Accelerate"
end
