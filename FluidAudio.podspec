Pod::Spec.new do |spec|
  spec.name         = "FluidAudio"
  spec.version      = "0.7.8"
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
  # CocoaPods sets SWIFT_VERSION based on this list; use values Xcode recognizes.
  spec.swift_versions = ["6.0"]

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

  spec.subspec "MachTaskSelfWrapper" do |mach|
    mach.source_files = "Sources/MachTaskSelfWrapper/**/*.{c,h}"
    mach.public_header_files = "Sources/MachTaskSelfWrapper/include/MachTaskSelf.h"
    mach.header_mappings_dir = "Sources/MachTaskSelfWrapper"
  end

  spec.subspec "Core" do |core|
    core.dependency "#{spec.name}/FastClusterWrapper"
    core.dependency "#{spec.name}/MachTaskSelfWrapper"
    core.source_files = "Sources/FluidAudio/**/*.swift"

    # iOS Configuration
    # TTS sources are moved under `Sources/FluidAudioTTS` and are not part of the Core subspec.
    # iOS builds include ASR, Diarization, and VAD.
    core.ios.frameworks = "CoreML", "AVFoundation", "Accelerate", "UIKit"

    # macOS Configuration
    core.osx.frameworks = "CoreML", "AVFoundation", "Accelerate", "Cocoa"
  end

  spec.default_subspecs = ["Core"]
end
