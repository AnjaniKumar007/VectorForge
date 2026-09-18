package com.vectordb;

import java.util.*;

public class BruteForce {
    private final List<VectorItem> items = new ArrayList<>();
    public void insert(VectorItem v) { items.add(v); }
    public List<Neighbor> knn(float[] q, int k, Distance.Metric dist) {
        List<Neighbor> r = new ArrayList<>(items.size());
        for (VectorItem v : items) r.add(new Neighbor(dist.apply(q, v.embedding()), v.id()));
        Collections.sort(r);
        if (r.size() > k) return new ArrayList<>(r.subList(0, k));
        return r;
    }
    public void remove(int id) { items.removeIf(v -> v.id() == id); }
}
