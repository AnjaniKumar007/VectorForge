package com.vectordb;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * VectorDB — In-memory vector database with multiple index backends.
 *
 * <p>Stores vectors alongside metadata and category labels, and indexes them
 * in three independent structures:</p>
 * <ul>
 *   <li><b>BruteForce</b> — exact linear scan (baseline)</li>
 *   <li><b>KDTree</b> — axis-aligned spatial partitioning</li>
 *   <li><b>HNSW</b> — approximate nearest-neighbour graph index</li>
 * </ul>
 *
 * <p>Callers can choose which algorithm to use at query time, and a built-in
 * {@link #benchmark} method lets you compare all three on the same query.</p>
 *
 * <p>All public methods are {@code synchronized} to guarantee thread safety.</p>
 */
public class VectorDB {

    // ──────────────────────────────────────────────────────────────────────
    //  1. Fields
    // ──────────────────────────────────────────────────────────────────────

    /** Primary storage: vector ID → VectorItem. */
    private final Map<Integer, VectorItem> store = new HashMap<>();

    /** Brute-force (linear scan) index. */
    private final BruteForce bf = new BruteForce();

    /** KD-Tree spatial index. */
    private final KDTree kdt;

    /** HNSW approximate nearest-neighbour graph index. */
    private final HNSW hnsw = new HNSW(16, 200);

    /** Auto-incrementing ID counter for new vectors. */
    private int nextId = 1;

    /** Fixed embedding dimensionality for this database. */
    public final int dims;

    /** Whether a JVM warm-up pass has been run (avoids cold-start noise in benchmarks). */
    private boolean warmedUp = false;

    // ──────────────────────────────────────────────────────────────────────
    //  2. Constructor
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Creates a new vector database for embeddings of the given dimensionality.
     *
     * @param d the fixed number of dimensions for every vector
     */
    public VectorDB(int d) {
        this.dims = d;
        this.kdt = new KDTree(d);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  3. Insert
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Inserts a new vector into the store and all three indexes.
     *
     * @param meta short metadata / label for the vector
     * @param cat  category string
     * @param emb  the embedding vector (will be cloned internally)
     * @param dist the distance metric used for HNSW insertion ordering
     * @return the auto-generated vector ID
     */
    public synchronized int insert(String meta, String cat, float[] emb, Distance.Metric dist) {
        VectorItem v = new VectorItem(nextId++, meta, cat, emb.clone());
        store.put(v.id(), v);

        // Index in all three structures
        bf.insert(v);
        kdt.insert(v);
        hnsw.insert(v, dist);

        return v.id();
    }

    // ──────────────────────────────────────────────────────────────────────
    //  4. Remove
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Removes a vector by ID from the store and all three indexes.
     *
     * <p>The KD-Tree does not support single-node deletion, so it is rebuilt
     * from scratch after removal.</p>
     *
     * @param id the vector ID to remove
     * @return {@code true} if the vector existed and was removed;
     *         {@code false} if no vector with that ID was found
     */
    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) {
            return false;
        }

        store.remove(id);
        bf.remove(id);
        hnsw.remove(id);
        kdt.rebuild(store.values());

        return true;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  5. Result Records
    // ──────────────────────────────────────────────────────────────────────

    /**
     * A single search result.
     *
     * @param id        vector ID
     * @param metadata  metadata string
     * @param category  category label
     * @param embedding the raw embedding vector
     * @param distance  distance from the query vector
     */
    public record Hit(
            int id,
            String metadata,
            String category,
            float[] embedding,
            float distance
    ) {}

    /**
     * Wrapper for search results including timing and algorithm info.
     *
     * @param hits      list of matched {@link Hit}s
     * @param latencyUs best-of-5 latency in microseconds
     * @param algo      the algorithm used ({@code "hnsw"}, {@code "bruteforce"}, {@code "kdtree"})
     * @param metric    the distance metric name
     */
    public record SearchOut(
            List<Hit> hits,
            long latencyUs,
            String algo,
            String metric
    ) {}

    /**
     * Benchmark results comparing all three index backends.
     *
     * @param bfUs   best-of-5 brute-force latency in microseconds
     * @param kdUs   best-of-5 KD-Tree latency in microseconds
     * @param hnswUs best-of-5 HNSW latency in microseconds
     * @param n      number of vectors in the store at benchmark time
     */
    public record BenchOut(
            long bfUs,
            long kdUs,
            long hnswUs,
            int n
    ) {}

    // ──────────────────────────────────────────────────────────────────────
    //  6. Warm-Up (JVM JIT helper)
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Runs a short warm-up pass on all three indexes so that subsequent
     * latency measurements are not polluted by JIT compilation and class
     * loading.
     *
     * <p>Called once (lazily) before the first search or benchmark.</p>
     *
     * @param q the query vector to use for warm-up
     * @param d the distance metric
     */
    private void warmup(float[] q, Distance.Metric d) {
        if (warmedUp || store.isEmpty()) {
            return;
        }

        for (int w = 0; w < 3; w++) {
            bf.knn(q, 1, d);
            kdt.knn(q, 1, d);
            hnsw.knn(q, 1, Math.min(store.size(), 10), d);
        }

        warmedUp = true;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  7. Adaptive ef (HNSW search width)
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Determines an appropriate HNSW {@code ef} (search beam width) based on
     * the current collection size and the requested {@code k}.
     *
     * <ul>
     *   <li>≤ 50 vectors  → ef = max(k, min(n, 10))</li>
     *   <li>≤ 200 vectors → ef = max(2k, 20)</li>
     *   <li>&gt; 200 vectors → ef = max(3k, 50)</li>
     * </ul>
     *
     * @param k number of neighbours requested
     * @return the computed ef value
     */
    private int adaptiveEf(int k) {
        int n = store.size();

        if (n <= 50) {
            return Math.max(k, Math.min(n, 10));
        }
        if (n <= 200) {
            return Math.max(k * 2, 20);
        }
        return Math.max(k * 3, 50);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  8. Search
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Searches for the {@code k} nearest neighbours using the specified
     * algorithm and distance metric.
     *
     * <p>The query is run 5 times and the <b>best</b> (lowest) latency is
     * recorded, reducing noise from GC pauses or scheduling jitter.</p>
     *
     * @param q      query embedding vector
     * @param k      number of nearest neighbours to return
     * @param metric distance metric name ({@code "cosine"}, {@code "euclidean"}, {@code "manhattan"})
     * @param algo   algorithm name ({@code "hnsw"}, {@code "bruteforce"}, {@code "kdtree"})
     * @return a {@link SearchOut} containing hits, latency, and metadata
     */
    public synchronized SearchOut search(float[] q, int k, String metric, String algo) {
        Distance.Metric d = Distance.get(metric);
        warmup(q, d);

        int ef = adaptiveEf(k);

        // Run 5 iterations and keep the best latency
        List<Neighbor> raw = null;
        long bestUs = Long.MAX_VALUE;

        for (int r = 0; r < 5; r++) {
            long start = System.nanoTime();

            raw = switch (algo) {
                case "bruteforce" -> bf.knn(q, k, d);
                case "kdtree"     -> kdt.knn(q, k, d);
                default           -> hnsw.knn(q, k, ef, d);
            };

            long us = (System.nanoTime() - start) / 1000;
            if (us < bestUs) {
                bestUs = us;
            }
        }

        // Map raw Neighbor results to Hit records
        List<Hit> hits = new ArrayList<>();
        for (Neighbor n : raw) {
            VectorItem v = store.get(n.id());
            if (v != null) {
                hits.add(new Hit(v.id(), v.metadata(), v.category(), v.embedding(), n.distance()));
            }
        }

        return new SearchOut(hits, bestUs, algo, metric);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  9. Benchmark
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Benchmarks all three index backends on the same query and returns
     * their best-of-5 latencies.
     *
     * @param q      query embedding vector
     * @param k      number of neighbours to search for
     * @param metric distance metric name
     * @return a {@link BenchOut} with per-algorithm latencies and collection size
     */
    public synchronized BenchOut benchmark(float[] q, int k, String metric) {
        Distance.Metric d = Distance.get(metric);
        warmup(q, d);

        int ef = adaptiveEf(k);

        long bfBest = Long.MAX_VALUE;
        long kdBest = Long.MAX_VALUE;
        long hwBest = Long.MAX_VALUE;

        for (int r = 0; r < 5; r++) {
            long t;
            long elapsed;

            // Brute-force
            t = System.nanoTime();
            bf.knn(q, k, d);
            elapsed = (System.nanoTime() - t) / 1000;
            if (elapsed < bfBest) {
                bfBest = elapsed;
            }

            // KD-Tree
            t = System.nanoTime();
            kdt.knn(q, k, d);
            elapsed = (System.nanoTime() - t) / 1000;
            if (elapsed < kdBest) {
                kdBest = elapsed;
            }

            // HNSW
            t = System.nanoTime();
            hnsw.knn(q, k, ef, d);
            elapsed = (System.nanoTime() - t) / 1000;
            if (elapsed < hwBest) {
                hwBest = elapsed;
            }
        }

        return new BenchOut(bfBest, kdBest, hwBest, store.size());
    }

    // ──────────────────────────────────────────────────────────────────────
    //  10. Utility / Accessor Methods
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Returns a snapshot list of every vector currently in the store.
     *
     * @return a new {@code ArrayList} containing all {@link VectorItem}s
     */
    public synchronized List<VectorItem> all() {
        return new ArrayList<>(store.values());
    }

    /**
     * Returns the HNSW graph structure info for visualisation / debugging.
     *
     * @return a {@link HNSW.GraphInfo} snapshot
     */
    public synchronized HNSW.GraphInfo hnswInfo() {
        return hnsw.getInfo();
    }

    /**
     * Returns the number of vectors currently stored.
     *
     * @return vector count
     */
    public synchronized int size() {
        return store.size();
    }
}
