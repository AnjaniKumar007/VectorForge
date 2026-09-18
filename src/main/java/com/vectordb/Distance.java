package com.vectordb;

public final class Distance {
    private Distance() {}
    public interface Metric { float apply(float[] a, float[] b); }

    public static float euclidean(float[] a, float[] b) {
        float s = 0f;
        for (int i = 0; i < a.length; i++) { float d = a[i] - b[i]; s += d * d; }
        return (float)Math.sqrt(s);
    }
    public static float cosine(float[] a, float[] b) {
        float dot = 0f, na = 0f, nb = 0f;
        for (int i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / ((float)Math.sqrt(na) * (float)Math.sqrt(nb));
    }
    public static float manhattan(float[] a, float[] b) {
        float s = 0f;
        for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s;
    }
    public static Metric get(String name) {
        return switch (name == null ? "" : name) {
            case "cosine" -> Distance::cosine;
            case "manhattan" -> Distance::manhattan;
            default -> Distance::euclidean;
        };
    }
}
