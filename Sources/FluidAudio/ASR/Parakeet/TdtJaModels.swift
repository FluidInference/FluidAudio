@preconcurrency import CoreML
import Foundation

/// Configuration for Japanese TDT models
public enum TdtJaConfig: ParakeetLanguageModelConfig {
    public static let blankId: Int = 3072
    public static let repository: Repo = .parakeetTdtJa
    public static let languageLabel: String = "TDT ja (Japanese)"
    public static let loggerCategory: String = "TdtJaModels"

    public static let preprocessorFile: String = ModelNames.TDTJa.preprocessorFile
    public static let encoderFile: String = ModelNames.TDTJa.encoderFile
    public static let decoderFile: String = ModelNames.TDTJa.decoderFile
    public static let vocabularyFile: String = ModelNames.TDTJa.vocabularyFile
    public static let jointFile: String? = ModelNames.TDTJa.jointFile

    public static let supportsInt8Encoder: Bool = false
    public static let encoderFp32File: String? = nil
}

/// Container for Parakeet TDT ja (Japanese) CoreML models (full TDT pipeline)
public typealias TdtJaModels = ParakeetLanguageModels<TdtJaConfig>
