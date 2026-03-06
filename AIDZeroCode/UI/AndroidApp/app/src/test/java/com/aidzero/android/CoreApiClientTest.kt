package com.aidzero.android

import org.junit.Assert.assertEquals
import org.junit.Test

class CoreApiClientTest {
    @Test
    fun normalizeBaseUrlAddsHttpAndTrailingSlash() {
        assertEquals(
            "http://192.168.1.20:8765/",
            CoreApiClient.normalizeBaseUrl("192.168.1.20:8765"),
        )
    }

    @Test
    fun normalizeBaseUrlPreservesExistingScheme() {
        assertEquals(
            "http://10.0.0.5:8765/",
            CoreApiClient.normalizeBaseUrl("http://10.0.0.5:8765/"),
        )
    }
}
