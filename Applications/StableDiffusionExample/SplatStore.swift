// Copyright © 2024 Apple Inc.

import Foundation

@Observable @MainActor
final class SplatStore {
    var tileSplatURLs: [URL] = []
    var combinedSplatURL: URL?
    var numTiles: Int = 0
    var isBuilding: Bool = false
    var errorMessage: String?
}
