package com.vectordb;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * OllamaClient — HTTP client for the Ollama local LLM server.
 *
 * <p>Provides two main capabilities:</p>
 * <ul>
 *   <li><b>Embedding</b> — converts text into a dense float vector via the
 *       {@code /api/embeddings} endpoint.</li>
 *   <li><b>Generation</b> — produces a text response from a prompt via the
 *       {@code /api/generate} endpoint (non-streaming).</li>
 * </ul>
 *
 * <p>By default the client connects to {@code http://127.0.0.1:11434} (the
 * standard Ollama port). Both the embedding model and the generation model
 * can be changed by setting the public fields {@link #embedModel} and
 * {@link #genModel}.</p>
 */
public class OllamaClient {

    // ──────────────────────────────────────────────────────────────────────
    //  1. Fields
    // ──────────────────────────────────────────────────────────────────────

    /** Base URL of the Ollama server (no trailing slash). */
    private final String baseUrl;

    /** Model name used for embedding requests. */
    public String embedModel = "nomic-embed-text";

    /** Model name used for text-generation requests. */
    public String genModel = "llama3.2";

    /** Shared HTTP client with a 3-second connect timeout. */
    private final HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(3))
            .build();

    // ──────────────────────────────────────────────────────────────────────
    //  2. Constructors
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Creates a client pointing to the default Ollama URL
     * ({@code http://127.0.0.1:11434}).
     */
    public OllamaClient() {
        this("http://127.0.0.1:11434");
    }

    /**
     * Creates a client pointing to a custom Ollama URL.
     *
     * @param url the base URL (a trailing slash, if present, is stripped)
     */
    public OllamaClient(String url) {
        this.baseUrl = url.replaceAll("/$", "");
    }

    // ──────────────────────────────────────────────────────────────────────
    //  3. Internal Helpers
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Convenience delegate to {@link JsonUtil#escape(String)} for building
     * JSON payloads safely.
     *
     * @param s the raw string
     * @return the JSON-escaped string
     */
    private static String esc(String s) {
        return JsonUtil.escape(s);
    }

    /**
     * Sends a POST request with a JSON body to the given path on the Ollama
     * server.
     *
     * @param path    the API path (e.g. {@code /api/embeddings})
     * @param body    the JSON request body
     * @param timeout request timeout in seconds
     * @return the response body on HTTP 200, or {@code null} on any error
     */
    private String post(String path, String body, int timeout) {
        try {
            HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + path))
                    .timeout(Duration.ofSeconds(timeout))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

            return response.statusCode() == 200 ? response.body() : null;
        } catch (Exception e) {
            return null;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  4. Health Check
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Checks whether the Ollama server is reachable by hitting the
     * {@code /api/tags} endpoint with a short timeout.
     *
     * @return {@code true} if the server responds with HTTP 200;
     *         {@code false} otherwise
     */
    public boolean isAvailable() {
        try {
            HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/api/tags"))
                    .timeout(Duration.ofSeconds(2))
                    .GET()
                    .build();

            return client.send(request, HttpResponse.BodyHandlers.discarding()).statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  5. Embed
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Generates a dense embedding vector for the given text using the
     * configured {@link #embedModel}.
     *
     * <p>Calls Ollama's {@code /api/embeddings} endpoint with a 35-second
     * timeout and parses the {@code "embedding"} array from the response.</p>
     *
     * @param text the input text to embed
     * @return the embedding vector, or an empty array on failure
     */
    public float[] embed(String text) {
        String payload = "{\"model\":\"" + esc(embedModel) + "\","
                + "\"prompt\":\"" + esc(text) + "\"}";

        String body = post("/api/embeddings", payload, 35);

        if (body == null) {
            return new float[0];
        }

        return JsonUtil.extractFloatArray(body, "embedding");
    }

    // ──────────────────────────────────────────────────────────────────────
    //  6. Generate
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Generates a text response from the given prompt using the configured
     * {@link #genModel}.
     *
     * <p>Calls Ollama's {@code /api/generate} endpoint in non-streaming mode
     * with a 180-second (3-minute) timeout and extracts the {@code "response"}
     * string from the result.</p>
     *
     * @param prompt the user prompt
     * @return the generated text, or an error message if Ollama is unavailable
     */
    public String generate(String prompt) {
        String payload = "{\"model\":\"" + esc(genModel) + "\","
                + "\"prompt\":\"" + esc(prompt) + "\","
                + "\"stream\":false}";

        String body = post("/api/generate", payload, 180);

        if (body == null) {
            return "ERROR: Ollama unavailable. Run: ollama serve";
        }

        return JsonUtil.extractString(body, "response");
    }
}
