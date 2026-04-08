@preconcurrency import CoreML
import Foundation

/// Configuration for Chinese (zh-CN) CTC models
public enum CtcZhCnConfig: ParakeetLanguageModelConfig {
    public static let blankId: Int = 7000
    public static let repository: Repo = .parakeetCtcZhCn
    public static let languageLabel: String = "CTC zh-CN"
    public static let loggerCategory: String = "CtcZhCnModels"

    public static let preprocessorFile: String = ModelNames.CTCZhCn.preprocessorFile
    public static let encoderFile: String = ModelNames.CTCZhCn.encoderFile
    public static let decoderFile: String = ModelNames.CTCZhCn.decoderFile
    public static let vocabularyFile: String = ModelNames.CTCZhCn.vocabularyFile
    public static let jointFile: String? = nil

    public static let supportsInt8Encoder: Bool = true
    public static let encoderFp32File: String? = ModelNames.CTCZhCn.encoderFp32File
}

/// Container for Parakeet CTC zh-CN CoreML models (full pipeline)
public typealias CtcZhCnModels = ParakeetLanguageModels<CtcZhCnConfig>
