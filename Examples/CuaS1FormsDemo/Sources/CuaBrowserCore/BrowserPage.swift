import CuaDemoCore
import Foundation

/// HTML fixtures contain controls and labels, never answer keys or source values.
public enum BrowserPage {
    /// Render the original control catalog as an independent, editable HTML form.
    public static func html(_ scenario: DemoScenario) -> String {
        let controls = scenario.controls.map { control in
            let label = escape(control.title)
            let id = "control-\(control.id)"
            let input: String
            switch control.role {
            case "Edit": input = "<input id='\(id)' type='text' autocomplete='off' placeholder='Empty'>"
            case "CheckBox": input = "<input id='\(id)' type='checkbox'>"
            default: input = "<button id='\(id)' type='button'>\(label)</button>"
            }
            return """
                <div class='control \(control.role == "Edit" ? "field" : "wide")'>
                <label for='\(id)'>\(label)</label>\(input)<span class='effect'></span></div>
                """
        }.joined(separator: "\n")
        return """
            <!doctype html><html><head><meta charset='utf-8'>
            <meta http-equiv='Content-Security-Policy' content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
            <style>
            *{box-sizing:border-box}body{margin:0;padding:22px;background:#fff;color:#1b3032;font:13px -apple-system,sans-serif}
            .eyebrow{font-size:10px;letter-spacing:1.6px;color:#687d76;margin-bottom:8px}
            h1{font-size:21px;line-height:1.25;margin:0 0 8px} .hint{color:#687d76;font-size:11px;margin-bottom:24px}
            form{display:grid;grid-template-columns:1fr 1fr;gap:16px 12px}
            .control{position:relative;min-width:0;padding:5px;border-radius:8px;transition:background .15s}
            .control.active{background:#e0f5e9;outline:2px solid #228660;outline-offset:3px}
            label{display:block;font-size:11px;font-weight:600;margin-bottom:7px;padding-right:32px}
            input[type=text]{width:100%;min-width:0;border:1px solid #d8e2dc;border-radius:7px;height:35px;padding:8px;color:#18352b;background:#fcfdfc;font:12px -apple-system}
            input:focus{outline:2px solid #258c62}input[type=checkbox]{accent-color:#228660}
            .wide{grid-column:1/-1}.wide label{display:inline;font-weight:400}.wide input{float:left;margin-right:8px}
            button{border:0;border-radius:6px;background:#e6eee8;padding:8px 14px;color:#1b4031;cursor:pointer}
            .wide:has(button) label{display:none}.effect{font-size:9px;color:#19754f;position:absolute;right:5px;top:5px}
            #receipt{display:none;padding:12px;margin-top:16px;background:#e0f5e9;border-radius:8px}
            </style></head><body>
            <div class='eyebrow'>LOCAL BROWSER · EDITABLE FORM</div><h1>\(escape(scenario.title))</h1>
            <p class='hint'>Watch the agent inspect, choose, fill, and check each control.</p>
            <form onsubmit='event.preventDefault()'>\(controls)</form>
            <div id='receipt'>Local demo receipt. No data was sent.</div>
            <script>
            window.eventCounts={input:0,change:0}; window.submitCount=0;
            document.addEventListener('input',()=>eventCounts.input++);
            document.addEventListener('change',()=>eventCounts.change++);
            document.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
              window.submitCount++;document.getElementById('receipt').style.display='block';
            }));
            </script></body></html>
            """
    }

    /// Escape values before including fixture strings in HTML text or attributes.
    public static func escape(_ value: String) -> String {
        value.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;").replacingOccurrences(of: "'", with: "&#39;")
    }
}

/// State obtained from the live DOM, independent of the original fixture's current state.
public struct BrowserControl: Codable, Equatable, Sendable {
    public let id: String
    public let role: String
    public let title: String
    public let value: String
    public let checked: Bool
    public let form: String

    /// Build the model input using the actual DOM label, role, form title, and current state.
    public func context(task: String) -> String {
        let state = role == "CheckBox" ? (checked ? "checked" : "unchecked") : "value=\"\(value)\""
        return "\(task)\nFORM \(form)\nELEMENT \(role) \"\(title)\" \(state)"
    }

    /// Derive an allowed action from a model-selected candidate. Clicking stays a review proposal.
    public func action(_ option: String) throws -> BrowserAction {
        if option == "skip" { return BrowserAction(kind: "skip", value: value, checked: checked) }
        if option == "click", role == "Button" {
            return BrowserAction(kind: "review", value: value, checked: checked)
        }
        if option == "check", role == "CheckBox" {
            return BrowserAction(kind: "check", value: value, checked: true)
        }
        if let fill = DemoCatalog.fillParts(option), role == "Edit" {
            return BrowserAction(kind: "fill", value: fill.value, checked: checked)
        }
        throw DemoError("The predicted action is incompatible with the observed browser control.")
    }
}

/// A validated action and expected state used for independent DOM readback.
public struct BrowserAction: Codable, Sendable {
    public let kind: String
    public let value: String
    public let checked: Bool
}
