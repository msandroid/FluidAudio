@testable import FluidAudio
import Testing

struct KokoroAutomaticVariantTests {
    @Test
    func selectVariant_singleFifteenVariantUsesFifteenForShortTokenCounts() throws {
        let caps = KokoroSynthesizer.TokenCapacities(
            short: 249,
            long: 249,
            forcedAutomaticVariant: .fifteenSecond
        )
        let v = try KokoroSynthesizer.selectVariant(forTokenCount: 50, preference: nil, capacities: caps)
        #expect(v == .fifteenSecond)
    }

    @Test
    func selectVariant_singleFiveVariantUsesFiveWithinCapacity() throws {
        let caps = KokoroSynthesizer.TokenCapacities(
            short: 71,
            long: 71,
            forcedAutomaticVariant: .fiveSecond
        )
        let v = try KokoroSynthesizer.selectVariant(forTokenCount: 40, preference: nil, capacities: caps)
        #expect(v == .fiveSecond)
    }

    @Test
    func selectVariant_dualModeStillPicksFiveForSmallCounts() throws {
        let caps = KokoroSynthesizer.TokenCapacities(short: 71, long: 249, forcedAutomaticVariant: nil)
        let v = try KokoroSynthesizer.selectVariant(forTokenCount: 50, preference: nil, capacities: caps)
        #expect(v == .fiveSecond)
    }

    @Test
    func selectVariant_dualModePromotesToFifteenForLargeCounts() throws {
        let caps = KokoroSynthesizer.TokenCapacities(short: 71, long: 249, forcedAutomaticVariant: nil)
        let v = try KokoroSynthesizer.selectVariant(forTokenCount: 111, preference: nil, capacities: caps)
        #expect(v == .fifteenSecond)
    }
}
