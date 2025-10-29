Pod::Spec.new do |spec|
  spec.name         = "FluidAudio"
  spec.version      = "0.7.6"
  spec.summary      = "Speaker diarization, voice-activity-detection and transcription with CoreML"
  spec.description  = <<-DESC
                       Fluid Audio is a Swift SDK for fully local, low-latency audio AI on Apple devices,
                       with inference offloaded to the Apple Neural Engine (ANE). The SDK includes
                       state-of-the-art speaker diarization, transcription, and voice activity detection
                       via open-source models that can be integrated with just a few lines of code.
                       DESC

  spec.homepage     = "https://github.com/FluidInference/FluidAudio"
  spec.license      = { :type => "Apache 2.0", :file => "LICENSE" }
  spec.author       = { "FluidInference" => "info@fluidinference.com" }

  spec.ios.deployment_target = "17.0"
  spec.osx.deployment_target = "14.0"

  spec.source       = { :git => "https://github.com/FluidInference/FluidAudio.git", :tag => "v#{spec.version}" }
  spec.swift_versions = ["5.10"]

  spec.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'ARCHS[sdk=macosx*]' => 'arm64',
    'EXCLUDED_ARCHS[sdk=macosx*]' => 'x86_64',
    'ARCHS[sdk=iphonesimulator*]' => 'arm64',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386 x86_64',
    'ARCHS[sdk=iphoneos*]' => 'arm64'
  }

  spec.user_target_xcconfig = {
    'ARCHS[sdk=macosx*]' => 'arm64',
    'EXCLUDED_ARCHS[sdk=macosx*]' => 'x86_64',
    'ARCHS[sdk=iphonesimulator*]' => 'arm64',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386 x86_64',
    'ARCHS[sdk=iphoneos*]' => 'arm64'
  }

  spec.subspec "FastClusterWrapper" do |wrapper|
    wrapper.requires_arc = false
    wrapper.source_files = "Sources/FastClusterWrapper/**/*.{cpp,h,hpp}"
    wrapper.public_header_files = "Sources/FastClusterWrapper/include/FastClusterWrapper.h"
    wrapper.private_header_files = "Sources/FastClusterWrapper/fastcluster_internal.hpp"
    wrapper.header_mappings_dir = "Sources/FastClusterWrapper"
    wrapper.pod_target_xcconfig = {
      'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17'
    }
  end

  spec.subspec "Core" do |core|
    core.dependency "#{spec.name}/FastClusterWrapper"
    core.source_files = "Sources/FluidAudio/**/*.swift"

    # iOS Configuration
    # Exclude TTS module from iOS builds to avoid ESpeakNG xcframework linking issues.
    # CocoaPods has known limitations with vendored xcframeworks during pod lib lint on iOS:
    # the framework symbols aren't properly linked in the temporary build environment,
    # causing "Undefined symbols" linker errors even though the binary is valid.
    # iOS builds include: ASR (speech recognition), Diarization, and VAD (voice activity detection).
    core.ios.exclude_files = "Sources/FluidAudio/TextToSpeech/**/*"
    core.ios.frameworks = "CoreML", "AVFoundation", "Accelerate", "UIKit"

    # macOS Configuration
    # ESpeakNG framework is only vendored for macOS in the podspec (not a framework limitation).
    # The xcframework supports iOS, but CocoaPods fails to link it during iOS validation.
    # This enables TTS (text-to-speech) functionality with G2P (grapheme-to-phoneme) conversion.
    # macOS builds include: ASR, Diarization, VAD, and TTS with ESpeakNG support.
    core.osx.vendored_frameworks = "Sources/FluidAudio/Frameworks/ESpeakNG.xcframework"
    core.osx.frameworks = "CoreML", "AVFoundation", "Accelerate", "Cocoa"
  end

  spec.default_subspecs = ["Core"]
end
