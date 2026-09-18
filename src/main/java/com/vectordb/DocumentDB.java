package com.vectordb;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DocumentDB — A lightweight in-memory document store backed by vector indexes.
 *
 * Each document consists of a title, body text, and a pre-computed embedding
 * vector. Documents are indexed in both an HNSW graph (for large collections)
 * and a brute-force index (used as a fallback for tiny collections), so that
 * approximate-nearest-neighbour searches can be performed efficiently.
 *
 * All public methods are {@code synchronized} to guarantee thread safety.
 */
public class DocumentDB {

    // ──────────────────────────────────────────────────────────────────────
    //  1. Inner Record — DocItem
    // ──────────────────────────────────────────────────────────────────────

    /**
     * An immutable record representing a single document stored in the DB.
     *
     * @param id        unique auto-incremented identifier
     * @param title     short human-readable title
     * @param text      full document body / content
     * @param embedding the dense vector representation of the document
     */
    public record DocItem(
            int id,
            String title,
            String text,
            float[] embedding
    ) {}

    // ──────────────────────────────────────────────────────────────────────
    //  2. Fields
    // ──────────────────────────────────────────────────────────────────────

    /** Primary storage: id → DocItem. */
    private final Map<Integer, DocItem> store = new HashMap<>();

    /** HNSW index for fast approximate nearest-neighbour queries. */
    private final HNSW hnsw = new HNSW(16, 200);

    /** Brute-force index used as fallback for very small collections. */
    private final BruteForce bf = new BruteForce();

    /** Auto-incrementing ID counter for new documents. */
    private int nextId = 1;

    /** Embedding dimensionality — set on the first insert, zero until then. */
    private int dims = 0;

    // ──────────────────────────────────────────────────────────────────────
    //  3. Insert
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Inserts a new document into the store and indexes its embedding.
     *
     * @param title short descriptive title
     * @param text  full document text
     * @param emb   dense embedding vector (will be cloned internally)
     * @return the auto-generated document ID
     */
    public synchronized int insert(String title, String text, float[] emb) {
        // Capture dimensionality from the very first embedding
        if (dims == 0) {
            dims = emb.length;
        }

        // Build the document item (clone the embedding to avoid aliasing)
        DocItem item = new DocItem(nextId++, title, text, emb.clone());
        store.put(item.id(), item);

        // Index in both vector structures
        VectorItem vi = new VectorItem(item.id(), title, "doc", emb);
        hnsw.insert(vi, Distance::cosine);
        bf.insert(vi);

        return item.id();
    }

    // ──────────────────────────────────────────────────────────────────────
    //  4. Search
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Searches for the {@code k} nearest documents to the given query vector,
     * filtering out any results whose distance exceeds {@code maxDist}.
     *
     * <p>For very small collections (&lt; 10 documents) the brute-force index
     * is used; otherwise the HNSW graph provides approximate results faster.</p>
     *
     * @param q       query embedding vector
     * @param k       maximum number of results to return
     * @param maxDist maximum allowable cosine distance (inclusive)
     * @return a list of (distance, DocItem) entries ordered by proximity
     */
    public synchronized List<Map.Entry<Float, DocItem>> search(float[] q, int k, float maxDist) {
        if (store.isEmpty()) {
            return List.of();
        }

        // Choose the index strategy based on collection size
        List<Neighbor> raw = store.size() < 10
                ? bf.knn(q, k, Distance::cosine)
                : hnsw.knn(q, k, 50, Distance::cosine);

        // Collect results that fall within the distance threshold
        List<Map.Entry<Float, DocItem>> out = new ArrayList<>();
        for (Neighbor n : raw) {
            DocItem d = store.get(n.id());
            if (d != null && n.distance() <= maxDist) {
                out.add(Map.entry(n.distance(), d));
            }
        }

        return out;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  5. Remove
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Removes a document by ID from both the store and the vector indexes.
     *
     * @param id the document ID to remove
     * @return {@code true} if the document existed and was removed;
     *         {@code false} if no document with that ID was found
     */
    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) {
            return false;
        }

        store.remove(id);
        hnsw.remove(id);
        bf.remove(id);

        return true;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  6. Utility / Accessor Methods
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Returns a snapshot list of every document currently in the store.
     *
     * @return a new {@code ArrayList} containing all {@link DocItem}s
     */
    public synchronized List<DocItem> all() {
        return new ArrayList<>(store.values());
    }

    /**
     * Returns the number of documents currently stored.
     *
     * @return document count
     */
    public synchronized int size() {
        return store.size();
    }

    /**
     * Returns the embedding dimensionality.
     * The value is {@code 0} until the first document is inserted.
     *
     * @return embedding vector length, or 0 if no documents exist yet
     */
    public synchronized int getDims() {
        return dims;
    }
}
