import Foundation
import Vision
import AppKit

struct Box: Codable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}
struct TextResult: Codable {
    let text: String
    let confidence: Float
    let box: Box
}
struct ImageResult: Codable {
    let path: String
    let results: [TextResult]
    let error: String?
}

if CommandLine.arguments.count < 2 {
    FileHandle.standardError.write(Data("用法：vocr 图片 [图片 ...]\n".utf8))
    exit(2)
}
var output: [ImageResult] = []
for path in CommandLine.arguments.dropFirst() {
    autoreleasepool {
        guard let image = NSImage(contentsOfFile: path),
              let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            output.append(ImageResult(path: path, results: [], error: "无法读取图片"))
            return
        }
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US"]
        request.usesLanguageCorrection = false
        do {
            try VNImageRequestHandler(cgImage: cg).perform([request])
            let results = (request.results ?? []).compactMap { observation -> TextResult? in
                guard let candidate = observation.topCandidates(1).first else { return nil }
                let b = observation.boundingBox
                // Vision 使用左下原点；输出统一使用左上原点。
                return TextResult(text: candidate.string, confidence: candidate.confidence,
                                  box: Box(x: max(0, b.minX), y: max(0, 1 - b.maxY),
                                           width: min(1, b.maxX) - max(0, b.minX),
                                           height: min(1, b.maxY) - max(0, b.minY)))
            }
            output.append(ImageResult(path: path, results: results, error: nil))
        } catch {
            output.append(ImageResult(path: path, results: [], error: error.localizedDescription))
        }
    }
}
let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys]
do {
    let data = try encoder.encode(output)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
} catch {
    FileHandle.standardError.write(Data("JSON 编码失败\n".utf8))
    exit(1)
}
