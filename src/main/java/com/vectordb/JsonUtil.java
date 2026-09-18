package com.vectordb;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * JsonUtil — Lightweight, dependency-free JSON helper utilities.
 *
 * <p>Provides methods for:</p>
 * <ul>
 *   <li>Escaping and quoting strings for safe JSON output</li>
 *   <li>Serialising float arrays and integer lists as JSON arrays</li>
 *   <li>Extracting typed values (string, int, float[]) from raw JSON bodies
 *       without a full parser</li>
 * </ul>
 *
 * <p>This is intentionally minimal — no external JSON library is required.</p>
 */
public final class JsonUtil {

    /** Utility class — prevent instantiation. */
    private JsonUtil() {}

    // ──────────────────────────────────────────────────────────────────────
    //  1. String Escaping & Quoting
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Escapes a string for safe inclusion inside a JSON string literal.
     *
     * <p>Handles: {@code "}, {@code \}, newline, carriage-return, and tab.</p>
     *
     * @param s the raw string (may be {@code null})
     * @return the escaped string (empty string if input is {@code null})
     */
    public static String escape(String s) {
        if (s == null) {
            return "";
        }

        StringBuilder out = new StringBuilder();
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"'  -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default   -> out.append(c);
            }
        }
        return out.toString();
    }

    /**
     * Wraps a string in double-quotes after escaping it.
     *
     * <p>Example: {@code q("hello")} → {@code "hello"}</p>
     *
     * @param s the raw string
     * @return a JSON-safe quoted string
     */
    public static String q(String s) {
        return "\"" + escape(s) + "\"";
    }

    // ──────────────────────────────────────────────────────────────────────
    //  2. Array Serialisation
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Serialises a {@code float[]} as a JSON array with 4-decimal precision.
     *
     * <p>Example: {@code arr(new float[]{1.5f, 2.0f})} → {@code [1.5000,2.0000]}</p>
     *
     * @param v the float array
     * @return a JSON array string
     */
    public static String arr(float[] v) {
        StringBuilder s = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) {
                s.append(',');
            }
            s.append(String.format(Locale.US, "%.4f", v[i]));
        }
        return s.append(']').toString();
    }

    /**
     * Serialises a {@code List<Integer>} as a JSON integer array.
     *
     * <p>Example: {@code arrInt(List.of(1,2,3))} → {@code [1,2,3]}</p>
     *
     * @param v the integer list
     * @return a JSON array string
     */
    public static String arrInt(List<Integer> v) {
        StringBuilder s = new StringBuilder("[");
        for (int i = 0; i < v.size(); i++) {
            if (i > 0) {
                s.append(',');
            }
            s.append(v.get(i));
        }
        return s.append(']').toString();
    }

    // ──────────────────────────────────────────────────────────────────────
    //  3. JSON Value Extraction — String
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Extracts a JSON string value for the given {@code key} from a raw JSON
     * body. Handles standard escape sequences inside the value.
     *
     * <p>Example: given {@code {"name":"John"}}, calling
     * {@code extractString(body, "name")} returns {@code "John"}.</p>
     *
     * @param body the raw JSON string
     * @param key  the JSON key to look up
     * @return the unescaped string value, or {@code ""} if not found
     */
    public static String extractString(String body, String key) {
        String needle = "\"" + key + "\"";
        int p = body.indexOf(needle);
        if (p < 0) {
            return "";
        }

        // Advance past the key and the colon separator
        p = body.indexOf(':', p + needle.length());
        if (p < 0) {
            return "";
        }
        p++;

        // Skip whitespace before the opening quote
        while (p < body.length() && Character.isWhitespace(body.charAt(p))) {
            p++;
        }
        if (p >= body.length() || body.charAt(p) != '"') {
            return "";
        }
        p++; // skip opening quote

        // Read characters, handling escape sequences
        StringBuilder out = new StringBuilder();
        boolean escaped = false;

        for (; p < body.length(); p++) {
            char c = body.charAt(p);

            if (escaped) {
                switch (c) {
                    case '"'  -> out.append('"');
                    case '\\' -> out.append('\\');
                    case 'n'  -> out.append('\n');
                    case 'r'  -> out.append('\r');
                    case 't'  -> out.append('\t');
                    default   -> out.append(c);
                }
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                break;  // closing quote
            } else {
                out.append(c);
            }
        }

        return out.toString();
    }

    // ──────────────────────────────────────────────────────────────────────
    //  4. JSON Value Extraction — Integer
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Extracts a JSON integer value for the given {@code key} from a raw JSON
     * body.
     *
     * <p>Example: given {@code {"count":42}}, calling
     * {@code extractInt(body, "count", 0)} returns {@code 42}.</p>
     *
     * @param body the raw JSON string
     * @param key  the JSON key to look up
     * @param def  default value returned when the key is absent or unparseable
     * @return the parsed integer, or {@code def} on failure
     */
    public static int extractInt(String body, String key, int def) {
        String needle = "\"" + key + "\"";
        int p = body.indexOf(needle);
        if (p < 0) {
            return def;
        }

        // Advance past the key and the colon separator
        p = body.indexOf(':', p + needle.length());
        if (p < 0) {
            return def;
        }

        try {
            p++;

            // Skip whitespace
            while (p < body.length() && Character.isWhitespace(body.charAt(p))) {
                p++;
            }

            // Find the end of the numeric token (digits and optional leading minus)
            int end = p;
            while (end < body.length()
                    && (Character.isDigit(body.charAt(end)) || body.charAt(end) == '-')) {
                end++;
            }

            return Integer.parseInt(body.substring(p, end));
        } catch (Exception e) {
            return def;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  5. JSON Value Extraction — Float Array
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Extracts a JSON float array value for the given {@code key} from a raw
     * JSON body. Supports nested brackets.
     *
     * <p>Example: given {@code {"vec":[1.0, 2.5, 3.0]}}, calling
     * {@code extractFloatArray(body, "vec")} returns {@code {1.0f, 2.5f, 3.0f}}.</p>
     *
     * @param body the raw JSON string
     * @param key  the JSON key to look up
     * @return the parsed float array, or an empty array on failure
     */
    public static float[] extractFloatArray(String body, String key) {
        String needle = "\"" + key + "\"";
        int p = body.indexOf(needle);
        if (p < 0) {
            return new float[0];
        }

        // Find the opening bracket
        p = body.indexOf('[', p + needle.length());
        if (p < 0) {
            return new float[0];
        }

        // Find the matching closing bracket (handles nested brackets)
        int end = p + 1;
        int depth = 1;
        while (end < body.length() && depth > 0) {
            char c = body.charAt(end++);
            if (c == '[') {
                depth++;
            } else if (c == ']') {
                depth--;
            }
        }
        if (depth != 0) {
            return new float[0];
        }

        // Parse the comma-separated numbers between the brackets
        String raw = body.substring(p + 1, end - 1).trim();
        if (raw.isEmpty()) {
            return new float[0];
        }

        String[] parts = raw.split(",");
        float[] result = new float[parts.length];
        try {
            for (int i = 0; i < parts.length; i++) {
                result[i] = Float.parseFloat(parts[i].trim());
            }
            return result;
        } catch (Exception ex) {
            return new float[0];
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  6. Comma-Separated Vector Parsing
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Parses a plain comma-separated string of numbers into a {@code float[]}.
     *
     * <p>Invalid tokens are silently skipped, so the returned array may be
     * shorter than the number of comma-separated segments.</p>
     *
     * <p>Example: {@code parseVector("1.0, 2.5, 3.0")} → {@code {1.0f, 2.5f, 3.0f}}</p>
     *
     * @param s the comma-separated number string (may be {@code null} or blank)
     * @return parsed float array, or empty array on {@code null}/blank input
     */
    public static float[] parseVector(String s) {
        if (s == null || s.isBlank()) {
            return new float[0];
        }

        String[] parts = s.split(",");
        float[] result = new float[parts.length];
        int count = 0;

        for (String token : parts) {
            try {
                result[count++] = Float.parseFloat(token.trim());
            } catch (Exception ignored) {
                // skip unparseable tokens
            }
        }

        return Arrays.copyOf(result, count);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  7. Alias — extractArrayField
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Convenience alias for {@link #extractFloatArray(String, String)}.
     *
     * @param body the raw JSON string
     * @param key  the JSON key to look up
     * @return the parsed float array, or an empty array on failure
     */
    public static float[] extractArrayField(String body, String key) {
        return extractFloatArray(body, key);
    }
}
