import Testing

@testable import FluidAudio

@Suite("KittenTTS Manager Tests")
struct KittenTtsManagerTests {

    @Test("Manager initializes with nano variant")
    func initNano() async {
        let manager = KittenTtsManager(variant: .nano)
        let available = await manager.isAvailable
        #expect(!available)
    }

    @Test("Manager initializes with mini variant")
    func initMini() async {
        let manager = KittenTtsManager(variant: .mini)
        let available = await manager.isAvailable
        #expect(!available)
    }

    @Test("Synthesize throws when not initialized")
    func synthesizeBeforeInit() async {
        let manager = KittenTtsManager(variant: .nano)
        do {
            _ = try await manager.synthesize(text: "test")
            Issue.record("Expected error but succeeded")
        } catch {
            // Expected
            #expect(error is KittenTTSError)
        }
    }

    @Test("Default voice is expr-voice-3-f")
    func defaultVoice() {
        #expect(KittenTtsConstants.defaultVoice == "expr-voice-3-f")
    }

    @Test("Available voices list has 8 entries")
    func availableVoices() {
        #expect(ModelNames.KittenTTS.availableVoices.count == 8)
    }

    @Test("KittenTtsVariant cases")
    func variantCases() {
        #expect(KittenTtsVariant.allCases.count == 2)
        #expect(KittenTtsVariant.nano.rawValue == "nano")
        #expect(KittenTtsVariant.mini.rawValue == "mini")
    }

    @Test("Model variant max tokens")
    func modelVariantMaxTokens() {
        #expect(ModelNames.KittenTTS.Variant.fiveSecond.maxTokens == 70)
        #expect(ModelNames.KittenTTS.Variant.tenSecond.maxTokens == 140)
    }

    @Test("Nano model filenames")
    func nanoFileNames() {
        let fiveS = ModelNames.KittenTTS.Variant.fiveSecond.nanoFileName()
        let tenS = ModelNames.KittenTTS.Variant.tenSecond.nanoFileName()
        #expect(fiveS == "kittentts_5s.mlmodelc")
        #expect(tenS == "kittentts_10s.mlmodelc")
    }

    @Test("Mini model filenames")
    func miniFileNames() {
        let fiveS = ModelNames.KittenTTS.Variant.fiveSecond.miniFileName()
        let tenS = ModelNames.KittenTTS.Variant.tenSecond.miniFileName()
        #expect(fiveS == "kittentts_mini_5s.mlmodelc")
        #expect(tenS == "kittentts_mini_10s.mlmodelc")
    }

    @Test("Repo configuration for nano")
    func repoNano() {
        let repo = Repo.kittenTtsNano
        #expect(repo.remotePath == "alexwengg/kittentts-coreml")
        #expect(repo.subPath == "nano")
        #expect(repo.folderName == "kittentts-coreml/nano")
    }

    @Test("Repo configuration for mini")
    func repoMini() {
        let repo = Repo.kittenTtsMini
        #expect(repo.remotePath == "alexwengg/kittentts-coreml")
        #expect(repo.subPath == "mini")
        #expect(repo.folderName == "kittentts-coreml/mini")
    }
}
