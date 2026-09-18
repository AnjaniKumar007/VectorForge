package com.vectordb;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * TextChunker — Splits a block of text into overlapping, fixed-size word chunks.
 *
 * <p>This is useful for preparing long documents before embedding: each chunk
 * is small enough to fit within a model's context window, and the overlap
 * ensures that no information is lost at chunk boundaries.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 *   // 50-word chunks with a 10-word overlap
 *   List<String> chunks = TextChunker.chunk(longText, 50, 10);
 * }</pre>
 */
public final class TextChunker {

    /** Utility class — prevent instantiation. */
    private TextChunker() {}

    // ──────────────────────────────────────────────────────────────────────
    //  1. Chunk
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Splits {@code text} into a list of word-level chunks with a sliding
     * window.
     *
     * <p><b>Algorithm:</b></p>
     * <ol>
     *   <li>Tokenise the input by whitespace.</li>
     *   <li>Slide a window of {@code chunkWords} words forward by
     *       {@code chunkWords − overlapWords} words on each step.</li>
     *   <li>Join the words in each window back into a single string.</li>
     * </ol>
     *
     * <p>If the text contains fewer words than {@code chunkWords}, the
     * original text is returned as a single-element list.</p>
     *
     * @param text         the input text (may be {@code null} or blank)
     * @param chunkWords   number of words per chunk
     * @param overlapWords number of words shared between consecutive chunks
     * @return an unmodifiable list with the original text if it fits in one
     *         chunk, or a mutable list of overlapping chunks otherwise;
     *         empty list if the input is {@code null}/blank
     */
    public static List<String> chunk(String text, int chunkWords, int overlapWords) {

        // ── Guard: null / blank input ──
        String trimmed = (text == null) ? "" : text.trim();
        if (trimmed.isEmpty()) {
            return List.of();
        }

        // ── Tokenise by whitespace ──
        String[] words = trimmed.split("\\s+");

        // If the text already fits in a single chunk, return it as-is
        if (words.length <= chunkWords) {
            return List.of(text);
        }

        // ── Sliding-window chunking ──
        List<String> out = new ArrayList<>();
        int step = chunkWords - overlapWords;

        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);

            // Join the current window back into a single string
            out.add(String.join(" ", Arrays.copyOfRange(words, i, end)));

            // Stop if we've reached the end of the word array
            if (end == words.length) {
                break;
            }
        }

        return out;
    }
}
