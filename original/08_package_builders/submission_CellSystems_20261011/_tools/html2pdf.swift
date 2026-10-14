// Render an HTML file (Quick Look / OfficeImport preview of a .docx) to a paginated A4 PDF with WebKit.
import Cocoa
import WebKit

let args = CommandLine.arguments
let inURL = URL(fileURLWithPath: args[1])
let outURL = URL(fileURLWithPath: args[2])

class Delegate: NSObject, WKNavigationDelegate {
    var done = false
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            let pi = NSPrintInfo()
            pi.paperSize = NSSize(width: 595.3, height: 841.9)
            pi.topMargin = 0; pi.bottomMargin = 0; pi.leftMargin = 0; pi.rightMargin = 0
            pi.horizontalPagination = .fit
            pi.verticalPagination = .automatic
            pi.jobDisposition = .save
            pi.dictionary()[NSPrintInfo.AttributeKey.jobSavingURL] = outURL
            let op = webView.printOperation(with: pi)
            op.showsPrintPanel = false
            op.showsProgressPanel = false
            op.view?.frame = webView.bounds
            op.runModal(for: self.window!, delegate: self, didRun: #selector(self.printed(_:success:contextInfo:)), contextInfo: nil)
        }
    }
    var window: NSWindow?
    @objc func printed(_ op: NSPrintOperation, success: Bool, contextInfo: UnsafeMutableRawPointer?) {
        print(success ? "ok" : "fail")
        exit(success ? 0 : 1)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.prohibited)
let d = Delegate()
let wv = WKWebView(frame: NSRect(x: 0, y: 0, width: 620, height: 842))
let win = NSWindow(contentRect: wv.frame, styleMask: [.borderless], backing: .buffered, defer: false)
win.contentView = wv
d.window = win
wv.navigationDelegate = d
wv.loadFileURL(inURL, allowingReadAccessTo: inURL.deletingLastPathComponent())
DispatchQueue.main.asyncAfter(deadline: .now() + 120) { print("timeout"); exit(2) }
app.run()
