package com.aidzero.android

import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

data class CoreHealth(
    val coreUrl: String,
    val status: String,
    val agent: String,
    val provider: String,
    val model: String,
)

class CoreApiClient(baseUrl: String) {
    val baseUrl: String = normalizeBaseUrl(baseUrl)

    fun health(): CoreHealth {
        val payload = request("GET", "/health")
        val data = payload.getJSONObject("data")
        return CoreHealth(
            coreUrl = baseUrl,
            status = data.optString("status", "ok"),
            agent = data.optString("active_profile", "unknown"),
            provider = data.optString("provider", ""),
            model = data.optString("model", ""),
        )
    }

    fun resetSession(): String {
        request(
            method = "POST",
            path = "/engine/session/reset",
            body = JSONObject(),
        )
        return "Started a new conversation."
    }

    fun addPrompt(prompt: String) {
        request(
            method = "POST",
            path = "/history/add",
            body = JSONObject().put("prompt", prompt),
        )
    }

    fun runPrompt(prompt: String): String {
        val event = JSONObject()
            .put("kind", "interactive")
            .put("source", "android")
            .put("prompt", prompt)
            .put(
                "metadata",
                JSONObject()
                    .put("trigger", "interactive")
                    .put("channel", "android")
                    .put("transport", "native"),
            )
        val payload = request(
            method = "POST",
            path = "/engine/run_event",
            body = JSONObject()
                .put("event", event)
                .put("max_rounds", 6),
        )
        val result = payload.getJSONObject("data").getJSONObject("result")
        return result.optString("response", "").ifBlank { "No response generated." }
    }

    private fun request(method: String, path: String, body: JSONObject? = null): JSONObject {
        val connection = (URL(baseUrl + path.removePrefix("/")).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 15_000
            readTimeout = 180_000
            setRequestProperty("Accept", "application/json")
            if (body != null) {
                doOutput = true
                setRequestProperty("Content-Type", "application/json; charset=utf-8")
            }
        }

        try {
            if (body != null) {
                OutputStreamWriter(connection.outputStream, Charsets.UTF_8).use { writer ->
                    writer.write(body.toString())
                }
            }

            val statusCode = connection.responseCode
            val stream = if (statusCode in 200..299) connection.inputStream else connection.errorStream
            val raw = BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).use { reader ->
                reader.readText()
            }
            val payload = JSONObject(raw)
            if (!payload.optBoolean("ok", false)) {
                throw IllegalStateException(payload.optString("error", "Unknown core error."))
            }
            return payload
        } finally {
            connection.disconnect()
        }
    }

    companion object {
        fun normalizeBaseUrl(raw: String): String {
            val trimmed = raw.trim()
            require(trimmed.isNotEmpty()) { "Core URL cannot be empty." }
            val withScheme = if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
                trimmed
            } else {
                "http://$trimmed"
            }
            return withScheme.trimEnd('/') + "/"
        }
    }
}
