package com.vectordb;

import java.util.*;

public class KDTree {
    private static class Node { VectorItem item; Node left, right; Node(VectorItem i){item=i;} }
    private Node root;
    private final int dims;
    public KDTree(int dims) { this.dims = dims; }
    private Node ins(Node n, VectorItem v, int d) {
        if (n == null) return new Node(v);
        int ax = d % dims;
        if (v.embedding()[ax] < n.item.embedding()[ax]) n.left = ins(n.left, v, d+1);
        else n.right = ins(n.right, v, d+1);
        return n;
    }
    public void insert(VectorItem v) { root = ins(root, v, 0); }
    private void knn(Node n, float[] q, int k, int d, Distance.Metric dist, PriorityQueue<Neighbor> heap) {
        if (n == null) return;
        float dn = dist.apply(q, n.item.embedding());
        if (heap.size() < k || dn < heap.peek().distance()) {
            heap.add(new Neighbor(dn, n.item.id()));
            if (heap.size() > k) heap.poll();
        }
        int ax = d % dims;
        float diff = q[ax] - n.item.embedding()[ax];
        Node closer = diff < 0 ? n.left : n.right;
        Node farther = diff < 0 ? n.right : n.left;
        knn(closer, q, k, d+1, dist, heap);
        if (heap.size() < k || Math.abs(diff) < heap.peek().distance()) knn(farther, q, k, d+1, dist, heap);
    }
    public List<Neighbor> knn(float[] q, int k, Distance.Metric dist) {
        if (k <= 0) return List.of();
        PriorityQueue<Neighbor> heap = new PriorityQueue<>(Comparator.reverseOrder());
        knn(root, q, k, 0, dist, heap);
        List<Neighbor> r = new ArrayList<>();
        while (!heap.isEmpty()) r.add(heap.poll());
        Collections.sort(r);
        return r;
    }
    public void rebuild(Collection<VectorItem> items) { root = null; for (VectorItem v : items) insert(v); }
}
