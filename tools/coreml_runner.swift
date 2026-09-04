import CoreML
import Darwin.Mach
import Foundation

enum RunnerError: Error, CustomStringConvertible {
    case usage(String)
    case invalidComputeUnits(String)
    case invalidInput(String)
    case missingModelInput
    case missingModelOutput

    var description: String {
        switch self {
        case .usage(let message), .invalidComputeUnits(let message), .invalidInput(let message):
            return message
        case .missingModelInput:
            return "Core ML model has no input"
        case .missingModelOutput:
            return "Core ML prediction has no multi-array output"
        }
    }
}

struct Arguments {
    let modelPath: String
    let inputPath: String
    let outputPath: String
    let computeUnits: MLComputeUnits
    let sequenceLength: Int
    let width: Int
    let warmup: Int
    let iterations: Int

    init(_ values: [String]) throws {
        guard values.count == 9 else {
            throw RunnerError.usage(
                "usage: coreml_runner MODEL INPUT.raw OUTPUT.raw "
                    + "{all|cpu_only|cpu_gpu|cpu_ne} SEQUENCE WIDTH WARMUP ITERATIONS"
            )
        }
        modelPath = values[1]
        inputPath = values[2]
        outputPath = values[3]
        switch values[4] {
        case "all": computeUnits = .all
        case "cpu_only": computeUnits = .cpuOnly
        case "cpu_gpu": computeUnits = .cpuAndGPU
        case "cpu_ne": computeUnits = .cpuAndNeuralEngine
        default: throw RunnerError.invalidComputeUnits("unknown compute units: \(values[4])")
        }
        guard
            let parsedSequence = Int(values[5]), parsedSequence > 0,
            let parsedWidth = Int(values[6]), parsedWidth > 0,
            let parsedWarmup = Int(values[7]), parsedWarmup >= 0,
            let parsedIterations = Int(values[8]), parsedIterations > 0
        else {
            throw RunnerError.usage("sequence, width, warmup, and iterations must be valid integers")
        }
        sequenceLength = parsedSequence
        width = parsedWidth
        warmup = parsedWarmup
        iterations = parsedIterations
    }
}

func elapsedSeconds(_ start: UInt64, _ end: UInt64) -> Double {
    Double(end - start) / 1_000_000_000.0
}

func percentile(_ sorted: [Double], _ fraction: Double) -> Double {
    let index = Int((Double(sorted.count - 1) * fraction).rounded(.up))
    return sorted[index]
}

func residentBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
        }
    }
    return status == KERN_SUCCESS ? info.resident_size : 0
}

func deviceName(_ device: MLComputeDevice) -> String {
    switch device {
    case .cpu: return "cpu"
    case .gpu: return "gpu"
    case .neuralEngine: return "neural_engine"
    @unknown default: return "unknown"
    }
}

func loadFloat16Array(path: String, shape: [NSNumber]) throws -> MLMultiArray {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let elementCount = shape.reduce(1) { $0 * $1.intValue }
    guard data.count == elementCount * MemoryLayout<UInt16>.size else {
        throw RunnerError.invalidInput(
            "input has \(data.count) bytes; expected \(elementCount * 2) for shape \(shape)"
        )
    }
    let array = try MLMultiArray(shape: shape, dataType: .float16)
    data.withUnsafeBytes { rawBuffer in
        let values = rawBuffer.bindMemory(to: UInt16.self)
        for index in 0..<elementCount {
            array[index] = NSNumber(value: Float(Float16(bitPattern: UInt16(littleEndian: values[index]))))
        }
    }
    return array
}

func writeFloat16Array(_ array: MLMultiArray, path: String) throws {
    var values = [UInt16]()
    values.reserveCapacity(array.count)
    for index in 0..<array.count {
        values.append(Float16(array[index].floatValue).bitPattern.littleEndian)
    }
    let data = values.withUnsafeBytes { Data($0) }
    try data.write(to: URL(fileURLWithPath: path), options: .atomic)
}

@main
struct CoreMLRunner {
    static func main() async throws {
        let arguments = try Arguments(CommandLine.arguments)
        let modelURL = URL(fileURLWithPath: arguments.modelPath)
        let initialResidentBytes = residentBytes()

        let compileStart = DispatchTime.now().uptimeNanoseconds
        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
        } else {
            compiledURL = try await MLModel.compileModel(at: modelURL)
        }
        let compileEnd = DispatchTime.now().uptimeNanoseconds

        let configuration = MLModelConfiguration()
        configuration.computeUnits = arguments.computeUnits
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        let loadEnd = DispatchTime.now().uptimeNanoseconds
        let loadedResidentBytes = residentBytes()

        guard let inputName = model.modelDescription.inputDescriptionsByName.keys.sorted().first else {
            throw RunnerError.missingModelInput
        }
        let input = try loadFloat16Array(
            path: arguments.inputPath,
            shape: [1, NSNumber(value: arguments.sequenceLength), NSNumber(value: arguments.width)]
        )
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(multiArray: input)]
        )

        var finalOutput: MLMultiArray?
        for _ in 0..<arguments.warmup {
            let prediction = try await model.prediction(from: provider)
            finalOutput = prediction.featureValue(for: prediction.featureNames.sorted().first!)?.multiArrayValue
        }

        var latencies = [Double]()
        latencies.reserveCapacity(arguments.iterations)
        for _ in 0..<arguments.iterations {
            let start = DispatchTime.now().uptimeNanoseconds
            let prediction = try await model.prediction(from: provider)
            let end = DispatchTime.now().uptimeNanoseconds
            latencies.append(elapsedSeconds(start, end))
            finalOutput = prediction.featureValue(for: prediction.featureNames.sorted().first!)?.multiArrayValue
        }
        guard let output = finalOutput else {
            throw RunnerError.missingModelOutput
        }
        try writeFloat16Array(output, path: arguments.outputPath)

        let sorted = latencies.sorted()
        var planOperations = [[String: Any]]()
        var planError: String?
        do {
            let plan = try await MLComputePlan.load(
                contentsOf: compiledURL,
                configuration: configuration
            )
            if case .program(let program) = plan.modelStructure {
                func visit(_ block: MLModelStructure.Program.Block) {
                    for operation in block.operations {
                        var record: [String: Any] = ["operator": operation.operatorName]
                        if let usage = plan.deviceUsage(for: operation) {
                            record["preferred_device"] = deviceName(usage.preferred)
                            record["supported_devices"] = usage.supported.map(deviceName)
                        }
                        if let cost = plan.estimatedCost(of: operation) {
                            record["estimated_cost_weight"] = cost.weight
                        }
                        planOperations.append(record)
                        for nested in operation.blocks {
                            visit(nested)
                        }
                    }
                }
                for function in program.functions.keys.sorted() {
                    visit(program.functions[function]!.block)
                }
            }
        } catch {
            planError = String(describing: error)
        }

        var result: [String: Any] = [
            "compile_seconds": elapsedSeconds(compileStart, compileEnd),
            "load_seconds": elapsedSeconds(loadStart, loadEnd),
            "resident_bytes": [
                "initial": initialResidentBytes,
                "after_load": loadedResidentBytes,
                "after_benchmark": residentBytes(),
                "load_delta": max(loadedResidentBytes, initialResidentBytes) - initialResidentBytes,
            ],
            "compute_units": CommandLine.arguments[4],
            "input_name": inputName,
            "output_shape": output.shape.map { $0.intValue },
            "warmup": arguments.warmup,
            "iterations": arguments.iterations,
            "latency_seconds": [
                "mean": latencies.reduce(0, +) / Double(latencies.count),
                "p50": percentile(sorted, 0.50),
                "p95": percentile(sorted, 0.95),
                "min": sorted.first!,
                "max": sorted.last!,
            ],
            "compute_plan_operations": planOperations,
        ]
        if let error = planError {
            result["compute_plan_error"] = error
        }
        let json = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
        FileHandle.standardOutput.write(json)
        FileHandle.standardOutput.write(Data([0x0A]))
    }
}
