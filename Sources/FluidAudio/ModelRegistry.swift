import Foundation
import OSLog

/// Model registry configuration for downloading models and datasets
/// Handles both registry URL and proxy configuration
public enum ModelRegistry {
    private static let logger = AppLogger(category: "ModelRegistry")

    // MARK: - Registry URL Configuration

    private static var _customBaseURL: String?

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
    public static func apiModels(_ repoPath: String, _ apiPath: String) -> URL {
        URL(string: "\(baseURL)/api/models/\(repoPath)/\(apiPath)")!
    }

    /// Construct download URL for a model file
    public static func resolveModel(_ repoPath: String, _ filePath: String) -> URL {
        URL(string: "\(baseURL)/\(repoPath)/resolve/main/\(filePath)")!
    }

    /// Construct API URL for listing dataset contents
    public static func apiDatasets(_ dataset: String, _ apiPath: String) -> URL {
        URL(string: "\(baseURL)/api/datasets/\(dataset)/\(apiPath)")!
    }

    /// Construct download URL for a dataset file
    public static func resolveDataset(_ dataset: String, _ filePath: String) -> URL {
        URL(string: "\(baseURL)/datasets/\(dataset)/resolve/main/\(filePath)")!
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
        var hasProxyConfig = false

        // Configure HTTPS proxy
        if let httpsProxy = ProcessInfo.processInfo.environment["https_proxy"],
            let proxySettings = parseProxyURL(httpsProxy, type: "HTTPS")
        {
            proxyConfig.merge(proxySettings) { _, new in new }
            hasProxyConfig = true
        }

        // Configure HTTP proxy
        if let httpProxy = ProcessInfo.processInfo.environment["http_proxy"],
            let proxySettings = parseProxyURL(httpProxy, type: "HTTP")
        {
            proxyConfig.merge(proxySettings) { _, new in new }
            hasProxyConfig = true
        }

        return hasProxyConfig ? proxyConfig : nil
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

        let config: [String: Any]
        switch type {
        case "HTTPS":
            config = [
                kCFNetworkProxiesHTTPSEnable as String: true,
                kCFNetworkProxiesHTTPSProxy as String: host,
                kCFNetworkProxiesHTTPSPort as String: port,
            ]
        case "HTTP":
            config = [
                kCFNetworkProxiesHTTPEnable as String: true,
                kCFNetworkProxiesHTTPProxy as String: host,
                kCFNetworkProxiesHTTPPort as String: port,
            ]
        default:
            return nil
        }

        logger.info("Configured \(type) proxy: \(host):\(port)")
        return config
        #else
        // Proxy configuration not available on iOS
        return nil
        #endif
    }
}
