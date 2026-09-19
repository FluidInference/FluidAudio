import CuaDemoCore
import Foundation
import WebKit

/// A bounded DOM driver for the demo's own WKWebView. It cannot navigate external sites.
@MainActor
public final class BrowserDriver: NSObject, WKNavigationDelegate {
    public let webView: WKWebView
    private var loadContinuation: CheckedContinuation<Void, Error>?

    public override init() {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        webView = WKWebView(frame: .zero, configuration: configuration)
        super.init()
        webView.navigationDelegate = self
    }

    /// Load a fresh local form, waiting for WebKit to finish its navigation.
    public func load(_ scenario: DemoScenario) async throws {
        guard loadContinuation == nil else { throw DemoError("A browser form is already loading.") }
        try await withCheckedThrowingContinuation { continuation in
            loadContinuation = continuation
            webView.loadHTMLString(BrowserPage.html(scenario), baseURL: nil)
        }
    }

    public func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        loadContinuation?.resume()
        loadContinuation = nil
    }

    public func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        failLoad(error)
    }

    public func webView(
        _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error
    ) {
        failLoad(error)
    }

    public func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        failLoad(DemoError("The browser process terminated. Reload the form."))
    }

    private func failLoad(_ error: Error) {
        loadContinuation?.resume(throwing: error)
        loadContinuation = nil
    }

    public func webView(
        _ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void
    ) {
        decisionHandler(navigationAction.request.url?.absoluteString == "about:blank" ? .allow : .cancel)
    }

    private static let observationScript = """
        function observe(e) {
          const role=e.tagName==='BUTTON'?'Button':e.type==='checkbox'?'CheckBox':'Edit';
          return {id:e.id,role,title:(e.labels?.[0]?.textContent || e.textContent).trim(),
                  value:e.tagName==='BUTTON'?'':e.value,checked:role==='CheckBox' && e.checked,
                  form:document.querySelector('h1').textContent.trim()};
        }
        """

    /// Observe the actual current DOM, without consulting expected labels or source entities.
    public func controls() async throws -> [BrowserControl] {
        try JSONDecoder().decode(
            [BrowserControl].self,
            from: Data(
                try await script(
                    Self.observationScript
                        + "return JSON.stringify(Array.from(document.querySelectorAll('input,button')).map(observe));"
                ).utf8))
    }

    /// Show which live control is being considered; focus also scrolls the HTML viewport.
    public func highlight(_ id: String) async throws {
        _ = try await script(
            """
            document.querySelectorAll('.active').forEach(e=>e.classList.remove('active'));
            const e=document.getElementById(id); if(!e)throw Error('Control disappeared');
            e.parentElement.classList.add('active');e.scrollIntoView({block:'nearest'});
            return JSON.stringify(true);
            """, arguments: ["id": id])
    }

    /// Reject stale observations, execute compatible effects, dispatch input/change, then read back.
    public func apply(_ action: BrowserAction, observed: BrowserControl) async throws {
        let encoder = JSONEncoder()
        let expected = String(decoding: try encoder.encode(observed), as: UTF8.self)
        let command = String(decoding: try encoder.encode(action), as: UTF8.self)
        _ = try await script(
            Self.observationScript + """
                const old=JSON.parse(expected),a=JSON.parse(command),e=document.getElementById(old.id);
                if(!e)throw Error('Control disappeared');
                const now=observe(e);
                if(Object.keys(old).some(k=>old[k]!==now[k]))throw Error('Control changed while the model was scoring');
                if(a.kind==='fill') {
                  if(now.role!=='Edit')throw Error('Expected text field');
                  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,a.value);
                  e.dispatchEvent(new Event('input',{bubbles:true})); e.dispatchEvent(new Event('change',{bubbles:true}));
                } else if(a.kind==='check') {
                  if(now.role!=='CheckBox')throw Error('Expected checkbox');
                  if(!e.checked)e.click();
                } else if(a.kind!=='skip' && a.kind!=='review')throw Error('Unsupported action');
                e.parentElement.querySelector('.effect').textContent={fill:'Filled ✓',check:'Checked ✓',skip:'Skipped',review:'Review click'}[a.kind];
                return JSON.stringify(true);
                """, arguments: ["expected": expected, "command": command])
        guard let actual = try await controls().first(where: { $0.id == observed.id }),
            actual.value == action.value, actual.checked == action.checked,
            actual.title == observed.title, actual.role == observed.role, actual.form == observed.form
        else { throw DemoError("The browser did not retain the predicted action.") }
    }

    /// Return event/submission counts for real browser integration verification.
    public func counters() async throws -> [String: Int] {
        try JSONDecoder().decode(
            [String: Int].self,
            from: Data(
                try await script(
                    "return JSON.stringify({...window.eventCounts,submit:window.submitCount});"
                ).utf8))
    }

    /// Execute local integration checks through the same isolated page world as the driver.
    public func script(_ body: String, arguments: [String: Any] = [:]) async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            webView.callAsyncJavaScript(body, arguments: arguments, in: nil, in: .page) { result in
                switch result {
                case .success(let value):
                    guard let string = value as? String else {
                        continuation.resume(throwing: DemoError("Browser returned an invalid observation."))
                        return
                    }
                    continuation.resume(returning: string)
                case .failure(let error): continuation.resume(throwing: error)
                }
            }
        }
    }
}
