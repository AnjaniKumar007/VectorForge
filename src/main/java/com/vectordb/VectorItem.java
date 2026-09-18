package com.vectordb;

public record VectorItem(int id, String metadata, String category, float[] embedding) {}
