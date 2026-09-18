package com.vectordb;

public record Neighbor(float distance, int id) implements Comparable<Neighbor> {
    @Override public int compareTo(Neighbor o) {
        int c = Float.compare(distance, o.distance);
        return c != 0 ? c : Integer.compare(id, o.id);
    }
}
