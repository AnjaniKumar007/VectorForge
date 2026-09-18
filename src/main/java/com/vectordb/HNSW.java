package com.vectordb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

/**
 * HNSW — Hierarchical Navigable Small World graph index.
 *
 * <p>Implements an approximate nearest-neighbour (ANN) index based on the HNSW
 * algorithm (Malkov &amp; Yashunin, 2018). Vectors are organised in a multi-layer
 * navigable graph where higher layers act as "express lanes" for faster
 * traversal and layer 0 contains every node.</p>
 *
 * <h3>Key parameters</h3>
 * <ul>
 *   <li><b>M</b> — max neighbours per node on layers 1+</li>
 *   <li><b>M0</b> — max neighbours per node on layer 0 (= 2 × M)</li>
 *   <li><b>efBuild</b> — search width during construction (controls quality)</li>
 *   <li><b>mL</b> — level generation multiplier: 1 / ln(M)</li>
 * </ul>
 */
public class HNSW {

    // ──────────────────────────────────────────────────────────────────────
    //  1. Inner Class — Node
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Internal graph node wrapping a {@link VectorItem} and its per-layer
     * neighbour lists.
     */
    private static class Node {

        /** The stored vector item. */
        final VectorItem item;

        /** The maximum layer this node was inserted into. */
        final int maxLayer;

        /**
         * Adjacency lists — one list of neighbour IDs per layer.
         * {@code nbrs.get(0)} is layer-0 neighbours, etc.
         */
        final List<List<Integer>> nbrs;

        /**
         * Creates a new node assigned to layers {@code 0 … maxLayer}.
         *
         * @param item     the vector item to store
         * @param maxLayer the highest layer for this node
         */
        Node(VectorItem item, int maxLayer) {
            this.item = item;
            this.maxLayer = maxLayer;
            this.nbrs = new ArrayList<>();
            for (int layer = 0; layer <= maxLayer; layer++) {
                nbrs.add(new ArrayList<>());
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  2. Fields
    // ──────────────────────────────────────────────────────────────────────

    /** All nodes in the graph, keyed by vector ID. */
    private final Map<Integer, Node> graph = new HashMap<>();

    /** Max neighbours per node on layers ≥ 1. */
    private final int M;

    /** Max neighbours per node on layer 0 (= 2 × M). */
    private final int M0;

    /** Search width used during index construction. */
    private final int efBuild;

    /** Level generation multiplier: 1 / ln(M). */
    private final float mL;

    /** Highest layer currently in the graph (-1 when empty). */
    private int topLayer = -1;

    /** Entry point node ID (-1 when the graph is empty). */
    private int entryPoint = -1;

    /** Deterministic RNG for reproducible layer assignment. */
    private final Random rng = new Random(42);

    // ──────────────────────────────────────────────────────────────────────
    //  3. Constructors
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Creates an HNSW index with default parameters (M = 16, efBuild = 200).
     */
    public HNSW() {
        this(16, 200);
    }

    /**
     * Creates an HNSW index with the given parameters.
     *
     * @param m       max neighbours per node on layers ≥ 1
     * @param efBuild search width during construction
     */
    public HNSW(int m, int efBuild) {
        this.M = m;
        this.M0 = 2 * m;
        this.efBuild = efBuild;
        this.mL = 1f / (float) Math.log(m);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  4. Random Level Generation
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Samples a random layer for a new node using an exponential distribution.
     * Most nodes land on layer 0; higher layers are exponentially rarer.
     *
     * @return the randomly chosen layer (≥ 0)
     */
    private int randLevel() {
        double u = rng.nextDouble();
        if (u == 0) {
            u = Double.MIN_VALUE;
        }
        return (int) Math.floor(-Math.log(u) * mL);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  5. Search Layer (core greedy beam-search)
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Performs a greedy beam-search on a single layer of the graph, returning
     * the {@code ef} closest neighbours found.
     *
     * <p>Uses two priority queues:</p>
     * <ul>
     *   <li><b>candidates</b> — min-heap of nodes still to explore</li>
     *   <li><b>found</b> — max-heap (by distance) of the best results so far,
     *       capped at size {@code ef}</li>
     * </ul>
     *
     * @param q     the query vector
     * @param ep    entry point node ID for this layer
     * @param ef    beam width / max result size
     * @param layer the specific graph layer to search
     * @param dist  the distance metric to use
     * @return sorted list of up to {@code ef} nearest {@link Neighbor}s
     */
    private List<Neighbor> searchLayer(float[] q, int ep, int ef, int layer, Distance.Metric dist) {

        Node epNode = graph.get(ep);
        if (epNode == null) {
            return List.of();
        }

        // Track visited node IDs to avoid re-processing
        HashSet<Integer> visited = new HashSet<>();

        // Min-heap: candidates still to explore (closest first)
        PriorityQueue<Neighbor> candidates = new PriorityQueue<>();

        // Max-heap: best results found so far (farthest first, for easy pruning)
        PriorityQueue<Neighbor> found = new PriorityQueue<>(Comparator.reverseOrder());

        // Seed with the entry point
        float d0 = dist.apply(q, epNode.item.embedding());
        Neighbor start = new Neighbor(d0, ep);
        visited.add(ep);
        candidates.add(start);
        found.add(start);

        // Greedy expansion
        while (!candidates.isEmpty()) {
            Neighbor cur = candidates.poll();

            // Early stop: if the closest candidate is farther than the worst
            // result in our found set (which is already full), we're done
            if (found.size() >= ef && cur.distance() > found.peek().distance()) {
                break;
            }

            Node node = graph.get(cur.id());
            if (node == null || layer >= node.nbrs.size()) {
                continue;
            }

            // Expand neighbours on this layer
            for (int nid : node.nbrs.get(layer)) {
                if (!visited.add(nid)) {
                    continue;   // already seen
                }

                Node nNode = graph.get(nid);
                if (nNode == null) {
                    continue;
                }

                float nd = dist.apply(q, nNode.item.embedding());
                Neighbor nn = new Neighbor(nd, nid);

                if (found.size() < ef || nd < found.peek().distance()) {
                    candidates.add(nn);
                    found.add(nn);

                    // Evict the farthest result if we exceeded the beam width
                    if (found.size() > ef) {
                        found.poll();
                    }
                }
            }
        }

        // Return results sorted by ascending distance
        List<Neighbor> result = new ArrayList<>(found);
        Collections.sort(result);
        return result;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  6. Neighbour Selection (simple truncation heuristic)
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Selects up to {@code maxM} closest neighbours from a sorted candidate
     * list. Uses a simple truncation strategy (no diversity / pruning
     * heuristic).
     *
     * @param candidates sorted list of candidate neighbours (closest first)
     * @param maxM       maximum number of neighbours to keep
     * @return list of selected neighbour IDs
     */
    private List<Integer> selectNbrs(List<Neighbor> candidates, int maxM) {
        List<Integer> result = new ArrayList<>();
        int limit = Math.min(candidates.size(), maxM);
        for (int i = 0; i < limit; i++) {
            result.add(candidates.get(i).id());
        }
        return result;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  7. Insert
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Inserts a vector item into the HNSW graph.
     *
     * <ol>
     *   <li>Assign a random layer level to the new node.</li>
     *   <li>From the current entry point, greedily descend through layers
     *       above the new node's level (using ef = 1).</li>
     *   <li>On layers from {@code min(topLayer, newLevel)} down to 0, run a
     *       full beam-search to find neighbours, connect them bidirectionally,
     *       and prune any neighbour lists that exceed their limit.</li>
     *   <li>If the new node is the highest layer yet, promote it to entry
     *       point.</li>
     * </ol>
     *
     * @param item the vector item to insert
     * @param dist the distance metric to use
     */
    public void insert(VectorItem item, Distance.Metric dist) {
        int id = item.id();
        int lvl = randLevel();
        graph.put(id, new Node(item, lvl));

        // ── First node in the graph? Just set it as entry point. ──
        if (entryPoint == -1) {
            entryPoint = id;
            topLayer = lvl;
            return;
        }

        int ep = entryPoint;

        // ── Phase 1: Greedy descent through layers above the new node ──
        for (int lc = topLayer; lc > lvl; lc--) {
            if (lc < graph.get(ep).nbrs.size()) {
                List<Neighbor> w = searchLayer(item.embedding(), ep, 1, lc, dist);
                if (!w.isEmpty()) {
                    ep = w.get(0).id();
                }
            }
        }

        // ── Phase 2: Insert and connect on layers min(topLayer, lvl) … 0 ──
        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {

            // Search for candidate neighbours on this layer
            List<Neighbor> w = searchLayer(item.embedding(), ep, efBuild, lc, dist);

            // Determine max connections for this layer
            int maxM = (lc == 0) ? M0 : M;

            // Select and assign this node's neighbours
            List<Integer> selected = selectNbrs(w, maxM);
            graph.get(id).nbrs.set(lc, selected);

            // Add reverse (bidirectional) connections and prune if needed
            for (int nid : selected) {
                Node other = graph.get(nid);
                if (other == null) {
                    continue;
                }

                // Ensure the neighbour has an adjacency list for this layer
                while (other.nbrs.size() <= lc) {
                    other.nbrs.add(new ArrayList<>());
                }

                List<Integer> conn = other.nbrs.get(lc);
                conn.add(id);

                // Prune the neighbour's list if it exceeds the limit
                if (conn.size() > maxM) {
                    List<Neighbor> distances = new ArrayList<>();
                    for (int c : conn) {
                        if (graph.containsKey(c)) {
                            float d = dist.apply(other.item.embedding(), graph.get(c).item.embedding());
                            distances.add(new Neighbor(d, c));
                        }
                    }
                    Collections.sort(distances);

                    conn.clear();
                    int keep = Math.min(maxM, distances.size());
                    for (int i = 0; i < keep; i++) {
                        conn.add(distances.get(i).id());
                    }
                }
            }

            // Use the closest result as entry point for the next layer down
            if (!w.isEmpty()) {
                ep = w.get(0).id();
            }
        }

        // ── Phase 3: Promote to entry point if this is the new highest layer ──
        if (lvl > topLayer) {
            topLayer = lvl;
            entryPoint = id;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  8. K-Nearest-Neighbour Query
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Queries the index for the {@code k} nearest neighbours of a query vector.
     *
     * <p>For very small graphs (≤ 64 nodes) a flat brute-force scan is used
     * instead of graph traversal, since the overhead of the multi-layer search
     * would outweigh any benefit.</p>
     *
     * @param q    query embedding vector
     * @param k    number of nearest neighbours to return
     * @param ef   search width (higher → more accurate, slower)
     * @param dist the distance metric to use
     * @return sorted list of up to {@code k} nearest {@link Neighbor}s
     */
    public List<Neighbor> knn(float[] q, int k, int ef, Distance.Metric dist) {
        if (entryPoint == -1) {
            return List.of();
        }

        int n = graph.size();

        // ── Fast path: flat scan for tiny graphs ──
        if (n <= 64) {
            List<Neighbor> all = new ArrayList<>(n);
            for (Node nd : graph.values()) {
                all.add(new Neighbor(dist.apply(q, nd.item.embedding()), nd.item.id()));
            }
            Collections.sort(all);
            return all.size() > k
                    ? new ArrayList<>(all.subList(0, k))
                    : all;
        }

        // ── Standard HNSW query: greedy descent + layer-0 beam-search ──
        int ep = entryPoint;

        // Descend through upper layers with ef = 1
        for (int lc = topLayer; lc > 0; lc--) {
            if (lc < graph.get(ep).nbrs.size()) {
                List<Neighbor> w = searchLayer(q, ep, 1, lc, dist);
                if (!w.isEmpty()) {
                    ep = w.get(0).id();
                }
            }
        }

        // Full beam-search on layer 0
        List<Neighbor> w = searchLayer(q, ep, Math.max(ef, k), 0, dist);
        return w.size() > k
                ? new ArrayList<>(w.subList(0, k))
                : w;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  9. Remove
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Removes a node from the graph by ID.
     *
     * <p>This performs a simple "tombstone" removal: the node is deleted and
     * all inbound edges pointing to it are pruned. If the removed node was the
     * entry point, a new entry point is elected from the remaining nodes.</p>
     *
     * @param id the vector ID to remove
     */
    public void remove(int id) {
        if (!graph.containsKey(id)) {
            return;
        }

        // Remove all inbound edges that reference this node
        for (Node nd : graph.values()) {
            for (List<Integer> layer : nd.nbrs) {
                layer.removeIf(x -> x == id);
            }
        }

        // Re-elect entry point if the removed node was the current one
        if (entryPoint == id) {
            entryPoint = -1;
            for (int nid : graph.keySet()) {
                if (nid != id) {
                    entryPoint = nid;
                    break;
                }
            }
        }

        graph.remove(id);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  10. Graph Introspection Records
    // ──────────────────────────────────────────────────────────────────────

    /**
     * View of a single node for graph visualisation / API responses.
     *
     * @param id       vector ID
     * @param metadata item metadata string
     * @param category item category
     * @param maxLayer the highest layer this node participates in
     */
    public record NodeView(int id, String metadata, String category, int maxLayer) {}

    /**
     * View of a single edge for graph visualisation / API responses.
     *
     * @param src   source node ID
     * @param dst   destination node ID
     * @param layer the graph layer this edge belongs to
     */
    public record EdgeView(int src, int dst, int layer) {}

    /**
     * Aggregated snapshot of the entire graph structure.
     *
     * @param topLayer      the highest occupied layer
     * @param nodeCount     total number of nodes
     * @param nodesPerLayer count of nodes present on each layer
     * @param edgesPerLayer count of (undirected) edges on each layer
     * @param nodes         list of all node views
     * @param edges         list of all edge views (de-duplicated: src &lt; dst)
     */
    public record GraphInfo(
            int topLayer,
            int nodeCount,
            List<Integer> nodesPerLayer,
            List<Integer> edgesPerLayer,
            List<NodeView> nodes,
            List<EdgeView> edges
    ) {}

    // ──────────────────────────────────────────────────────────────────────
    //  11. Get Graph Info
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Returns a full structural snapshot of the graph, including per-layer
     * node/edge counts and all node/edge views. Useful for visualization
     * endpoints or debugging.
     *
     * <p>Edges are de-duplicated so that each undirected edge appears once
     * (only the direction where {@code src < dst} is emitted).</p>
     *
     * @return a {@link GraphInfo} snapshot
     */
    public GraphInfo getInfo() {
        int maxL = Math.max(topLayer + 1, 1);

        // Per-layer counters, initialised to zero
        List<Integer> nodesPerLayer = new ArrayList<>(Collections.nCopies(maxL, 0));
        List<Integer> edgesPerLayer = new ArrayList<>(Collections.nCopies(maxL, 0));

        List<NodeView> nodes = new ArrayList<>();
        List<EdgeView> edges = new ArrayList<>();

        for (var entry : graph.entrySet()) {
            int id = entry.getKey();
            Node nd = entry.getValue();

            // Record the node view
            nodes.add(new NodeView(id, nd.item.metadata(), nd.item.category(), nd.maxLayer));

            // Walk each layer this node participates in
            for (int lc = 0; lc <= nd.maxLayer && lc < maxL; lc++) {
                nodesPerLayer.set(lc, nodesPerLayer.get(lc) + 1);

                // Count and record edges (only once per undirected pair)
                if (lc < nd.nbrs.size()) {
                    for (int nid : nd.nbrs.get(lc)) {
                        if (id < nid) {
                            edgesPerLayer.set(lc, edgesPerLayer.get(lc) + 1);
                            edges.add(new EdgeView(id, nid, lc));
                        }
                    }
                }
            }
        }

        return new GraphInfo(topLayer, graph.size(), nodesPerLayer, edgesPerLayer, nodes, edges);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  12. Size
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Returns the number of nodes currently in the graph.
     *
     * @return node count
     */
    public int size() {
        return graph.size();
    }
}
