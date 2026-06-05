#!/usr/bin/env swift
// ANE Profiler — reports the CoreML compute-unit *plan* (ANE / GPU / CPU) for each
// .mlmodelc bundle passed as an argument. Uses MLComputePlan (macOS 14.4+ / iOS 17.4+).
//
// Usage:  swift Scripts/ane_profile.swift [--units ane|gpu|cpu|all] <path-to.mlmodelc> ...
//
// Notes:
//  - This is the scheduler's *plan* (preferredComputeDevice per op), not a runtime power trace.
//  - Counts are by operation (mlprogram) or layer (neuralnetwork). Pipelines recurse into submodels.

import CoreML
import Foundation

// Unbuffered stdout so partial results survive a hard CoreML crash (some bundles
// SIGSEGV/SIGABRT inside MLComputePlan.load — a Swift do/catch can't trap those).
setvbuf(stdout, nil, _IONBF, 0)

struct Counts { var ane = 0, gpu = 0, cpu = 0, other = 0
    var total: Int { ane + gpu + cpu + other }
    static func + (l: Counts, r: Counts) -> Counts {
        Counts(ane: l.ane + r.ane, gpu: l.gpu + r.gpu, cpu: l.cpu + r.cpu, other: l.other + r.other)
    }
}

func classify(_ device: MLComputeDevice, into c: inout Counts) {
    switch device {
    case .neuralEngine: c.ane += 1
    case .gpu: c.gpu += 1
    case .cpu: c.cpu += 1
    @unknown default: c.other += 1
    }
}

@available(macOS 14.4, *)
func walk(_ ops: [MLModelStructure.Program.Operation], _ plan: MLComputePlan, _ c: inout Counts) {
    for op in ops {
        // const/dummy ops have no device; skip those without a usage entry.
        if let usage = plan.deviceUsage(for: op) {
            classify(usage.preferred, into: &c)
        }
        for block in op.blocks { walk(block.operations, plan, &c) }
    }
}

@available(macOS 14.4, *)
func profile(_ structure: MLModelStructure, _ plan: MLComputePlan) -> Counts {
    var c = Counts()
    switch structure {
    case .program(let program):
        for (_, fn) in program.functions { walk(fn.block.operations, plan, &c) }
    case .neuralNetwork(let nn):
        for layer in nn.layers {
            if let usage = plan.deviceUsage(for: layer) {
                classify(usage.preferred, into: &c)
            }
        }
    case .pipeline(let pipeline):
        for sub in pipeline.subModels { c = c + profile(sub, plan) }
    case .unsupported:
        break
    @unknown default:
        break
    }
    return c
}

func pct(_ n: Int, _ total: Int) -> String {
    total == 0 ? "—" : String(format: "%2.0f%%", 100.0 * Double(n) / Double(total))
}

// ---- arg parsing ----
var units: MLComputeUnits = .cpuAndNeuralEngine
var paths: [String] = []
var it = CommandLine.arguments.dropFirst().makeIterator()
while let a = it.next() {
    if a == "--units", let v = it.next() {
        switch v.lowercased() {
        case "ane", "cpuandneuralengine": units = .cpuAndNeuralEngine
        case "gpu", "cpuandgpu": units = .cpuAndGPU
        case "cpu", "cpuonly": units = .cpuOnly
        case "all": units = .all
        default: FileHandle.standardError.write(Data("unknown --units \(v)\n".utf8))
        }
    } else {
        paths.append(a)
    }
}

guard !paths.isEmpty else {
    print("usage: swift Scripts/ane_profile.swift [--units ane|gpu|cpu|all] <model.mlmodelc> ...")
    exit(2)
}

guard #available(macOS 14.4, *) else {
    FileHandle.standardError.write(Data("requires macOS 14.4+\n".utf8))
    exit(1)
}

let config = MLModelConfiguration()
config.computeUnits = units

print(String(format: "%-46@  %5@ %5@ %5@   %@", "model" as NSString, "ANE" as NSString,
             "GPU" as NSString, "CPU" as NSString, "ops"))
print(String(repeating: "-", count: 78))

let sem = DispatchSemaphore(value: 0)
Task {
    var grand = Counts()
    for path in paths {
        let url = URL(fileURLWithPath: path)
        let name = url.deletingPathExtension().lastPathComponent
        do {
            let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
            let c = profile(plan.modelStructure, plan)
            grand = grand + c
            print(String(format: "%-46@  %5@ %5@ %5@   %d", name as NSString,
                         pct(c.ane, c.total) as NSString, pct(c.gpu, c.total) as NSString,
                         pct(c.cpu, c.total) as NSString, c.total))
        } catch {
            print(String(format: "%-46@  load failed: %@", name as NSString,
                         "\(error)" as NSString))
        }
    }
    if paths.count > 1 {
        print(String(repeating: "-", count: 78))
        print(String(format: "%-46@  %5@ %5@ %5@   %d", "TOTAL" as NSString,
                     pct(grand.ane, grand.total) as NSString, pct(grand.gpu, grand.total) as NSString,
                     pct(grand.cpu, grand.total) as NSString, grand.total))
    }
    sem.signal()
}
sem.wait()
