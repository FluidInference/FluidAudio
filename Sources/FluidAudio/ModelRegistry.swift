import Foundation
import OSLog

/// Model registry configuration for downloading models and datasets
/// Handles both registry URL and proxy configuration
public enum ModelRegistry {
    private static let logger = AppLogger(category: "ModelRegistry")

    // MARK: - Error Types

    /// Error thrown when URL construction fails
    public enum Error: Swift.Error, CustomStringConvertible {
        case invalidURL(String)

        public var description: String {
            switch self {
            case .invalidURL(let urlString):
                return "Invalid URL construction: \(urlString)"
            }
        }
    }

    // MARK: - Registry URL Configuration

    // Mutable static for runtime configuration. nonisolated(unsafe) because it's
    // typically set once at startup before any concurrent access.
    nonisolated(unsafe) private static var _customBaseURL: String?

    /// Base registry URL (default: HuggingFace)
    /// Can be overridden programmatically to use a different model registry or mirror.
    /// Priority: programmatic override → REGISTRY_URL env var → MODEL_REGISTRY_URL env var → default
    public static var baseURL: String {
        get {
            _customBaseURL
                ?? ProcessInfo.processInfo.environment["REGISTRY_URL"]
                ?? ProcessInfo.processInfo.environment["MODEL_REGISTRY_URL"]
                ?? "https://huggingface.co"
        }
        set {
            _customBaseURL = newValue
        }
    }

    // MARK: - URL Construction

    /// Construct API URL for listing model repository contents
    public static func apiModels(_ repoPath: String, _ apiPath: String) throws -> URL {
        let urlString = "\(baseURL)/api/models/\(repoPath)/\(apiPath)"
        guard let url = URL(string: urlString) else {
            throw Error.invalidURL(urlString)
        }
        return url
    }

    /// Construct download URL for a model file
    public static func resolveModel(_ repoPath: String, _ filePath: String) throws -> URL {
        let urlString = "\(baseURL)/\(repoPath)/resolve/main/\(filePath)"
        guard let url = URL(string: urlString) else {
            throw Error.invalidURL(urlString)
        }
        return url
    }

    /// Construct API URL for listing dataset contents
    public static func apiDatasets(_ dataset: String, _ apiPath: String) throws -> URL {
        let urlString = "\(baseURL)/api/datasets/\(dataset)/\(apiPath)"
        guard let url = URL(string: urlString) else {
            throw Error.invalidURL(urlString)
        }
        return url
    }

    /// Construct download URL for a dataset file
    public static func resolveDataset(_ dataset: String, _ filePath: String) throws -> URL {
        let urlString = "\(baseURL)/datasets/\(dataset)/resolve/main/\(filePath)"
        guard let url = URL(string: urlString) else {
            throw Error.invalidURL(urlString)
        }
        return url
    }

    /// Construct base URL for dataset directory (without trailing slash)
    public static func resolveDatasetBase(_ dataset: String) -> String {
        "\(baseURL)/datasets/\(dataset)/resolve/main"
    }

    // MARK: - Session Configuration

    /// Create a URLSession configured with registry URL and proxy settings
    static func configuredSession() -> URLSession {
        let configuration = URLSessionConfiguration.default

        // Configure proxy settings if environment variables are set
        if let proxyConfig = configureProxySettings() {
            configuration.connectionProxyDictionary = proxyConfig
        }

        return URLSession(configuration: configuration)
    }

    // MARK: - Proxy Configuration (macOS only)

    private static func configureProxySettings() -> [String: Any]? {
        #if os(macOS)
        var proxyConfig: [String: Any] = [:]

        // Configure HTTPS proxy
        if let httpsProxy = ProcessInfo.processInfo.environment["https_proxy"],
            let proxySettings = parseProxyURL(httpsProxy, type: "HTTPS")
        {
            proxyConfig.merge(proxySettings) { _, new in new }
        }

        // Configure HTTP proxy
        if let httpProxy = ProcessInfo.processInfo.environment["http_proxy"],
            let proxySettings = parseProxyURL(httpProxy, type: "HTTP")
        {
            proxyConfig.merge(proxySettings) { _, new in new }
        }

        return proxyConfig.isEmpty ? nil : proxyConfig
        #else
        // Proxy configuration not available on iOS
        return nil
        #endif
    }

    private static func parseProxyURL(_ proxyURLString: String, type: String) -> [String: Any]? {
        #if os(macOS)
        guard let proxyURL = URL(string: proxyURLString),
            let host = proxyURL.host,
            let port = proxyURL.port
        else {
            logger.warning("Invalid \(type) proxy URL: \(proxyURLString)")
            return nil
        }

        let proxyKey: String
        let enableKey: String
        let portKey: String

        switch type {
        case "HTTPS":
            enableKey = kCFNetworkProxiesHTTPSEnable as String
            proxyKey = kCFNetworkProxiesHTTPSProxy as String
            portKey = kCFNetworkProxiesHTTPSPort as String
        case "HTTP":
            enableKey = kCFNetworkProxiesHTTPEnable as String
            proxyKey = kCFNetworkProxiesHTTPProxy as String
            portKey = kCFNetworkProxiesHTTPPort as String
        default:
            return nil
        }

        logger.info("Configured \(type) proxy: \(host):\(port)")
        return [
            enableKey: true,
            proxyKey: host,
            portKey: port,
        ]
        #else
        // Proxy configuration not available on iOS
        return nil
        #endif
    }
}
