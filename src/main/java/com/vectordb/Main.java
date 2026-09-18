package com.vectordb;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executors;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
public class Main {
    static final int DIMS=16;
    static final VectorDB db=new VectorDB(DIMS);
    static final DocumentDB docDB=new DocumentDB();
    static final OllamaClient ollama=new OllamaClient();
    public static void main(String[] args)throws Exception {
        DemoData.load(db);
        boolean up=ollama.isAvailable();
        System.out.println("=== VectorDB Java Engine ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size()+" demo vectors | "+DIMS+" dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: "+(up?"ONLINE":"OFFLINE (install from ollama.com)"));
        if(up)System.out.println("  embed model: "+ollama.embedModel+"  gen model: "+ollama.genModel);
        HttpServer server=HttpServer.create(new InetSocketAddress("0.0.0.0",8080),0);
        server.setExecutor(Executors.newFixedThreadPool(Math.max(8,Runtime.getRuntime().availableProcessors()-1)));
        route(server,"/", "GET", Main::home);
        route(server,"/search","GET",Main::search);
        route(server,"/insert","POST",Main::insert);
        route(server,"/items","GET",Main::items);
        route(server,"/benchmark","GET",Main::benchmark);
        route(server,"/hnsw-info","GET",Main::hnswInfo);
        route(server,"/stats","GET",Main::stats);
        route(server,"/status","GET",Main::status);
        route(server,"/doc/insert","POST",Main::docInsert);
        route(server,"/doc/delete/","DELETE",Main::docDelete);
        route(server,"/doc/list","GET",Main::docList);
        route(server,"/doc/search","POST",Main::docSearch);
        route(server,"/doc/ask","POST",Main::docAsk);
        route(server,"/delete/","DELETE",Main::delete);
        server.start();
        System.out.println("Server started on port 8080");
    }
    interface Handler {
        void run(HttpExchange e)throws Exception;
    }
    static void route(HttpServer s,String path,String method,Handler h) {
        s.createContext(path,e-> {
            cors(e); if(e.getRequestMethod().equals("OPTIONS")) {
                e.sendResponseHeaders(204,-1); return;
            }
            if(!e.getRequestMethod().equals(method)) {
                send(e,405,"{\"error\":\"method not allowed\"}","application/json"); return;
            }
            try {
                h.run(e);
            }catch(Exception ex) {
                ex.printStackTrace(); send(e,500,"{\"error\":\"server error\"}","application/json");
            }
        }
        );
    }
    static void cors(HttpExchange e) {
        Headers h=e.getResponseHeaders();
        h.set("Access-Control-Allow-Origin","*");
        h.set("Access-Control-Allow-Methods","GET, POST, DELETE, OPTIONS");
        h.set("Access-Control-Allow-Headers","Content-Type");
    }
    static String body(HttpExchange e)throws IOException {
        return new String(e.getRequestBody().readAllBytes(),StandardCharsets.UTF_8);
    }
    static Map<String,String> query(HttpExchange e) {
        Map<String,String>m=new HashMap<>();
        String q=e.getRequestURI().getRawQuery();
        if(q==null)return m;
        for(String p:q.split("&")) {
            String[]x=p.split("=",2);
            try {
                m.put(URLDecoder.decode(x[0],StandardCharsets.UTF_8),x.length>1?URLDecoder.decode(x[1],StandardCharsets.UTF_8):"");
            }catch(Exception ignored) {
            }
        }
        return m;
    }
    static void send(HttpExchange e,int status,String text,String type)throws IOException {
        byte[]b=text.getBytes(StandardCharsets.UTF_8);
        e.getResponseHeaders().set("Content-Type",type+"; charset=utf-8");
        e.sendResponseHeaders(status,b.length);
        try(OutputStream o=e.getResponseBody()) {
            o.write(b);
        }
    }
    static void home(HttpExchange e)throws Exception {
        try(InputStream in=Main.class.getResourceAsStream("/static/index.html")) {
            if(in==null) {
                send(e,404,"index.html not found","text/plain");
                return;
            }
            send(e,200,new String(in.readAllBytes(),StandardCharsets.UTF_8),"text/html");
        }
    }
    static void search(HttpExchange e)throws Exception {
        Map<String,String>q=query(e);
        float[]v=JsonUtil.parseVector(q.get("v"));
        if(v.length!=DIMS) {
            send(e,400,"{\"error\":\"need 16D vector\"}","application/json");
            return;
        }
        int k=parseInt(q.get("k"),5);
        String metric=q.getOrDefault("metric","cosine"),algo=q.getOrDefault("algo","hnsw");
        var o=db.search(v,k,metric,algo);
        StringBuilder s=new StringBuilder("{\"results\":[");
        for(int i=0; i<o.hits().size(); i++) {
            if(i>0)s.append(',');
            var h=o.hits().get(i);
            s.append("{\"id\":").append(h.id()).append(",\"metadata\":").append(JsonUtil.q(h.metadata())).append(",\"category\":").append(JsonUtil.q(h.category())).append(",\"distance\":").append(String.format(Locale.US,"%.6f",h.distance())).append(",\"embedding\":").append(JsonUtil.arr(h.embedding())).append('}');
        }
        s.append("],\"latencyUs\":").append(o.latencyUs()).append(",\"algo\":").append(JsonUtil.q(o.algo())).append(",\"metric\":").append(JsonUtil.q(o.metric())).append('}');
        send(e,200,s.toString(),"application/json");
    }
    static void insert(HttpExchange e)throws Exception {
        String b=body(e),meta=JsonUtil.extractString(b,"metadata"),cat=JsonUtil.extractString(b,"category");
        float[]emb=JsonUtil.extractArrayField(b,"embedding");
        if(meta.isEmpty()||emb.length!=DIMS) {
            send(e,400,"{\"error\":\"invalid body\"}","application/json");
            return;
        }
        int id=db.insert(meta,cat,emb,Distance::cosine);
        send(e,200,"{\"id\":"+id+"}","application/json");
    }
    static void delete(HttpExchange e)throws Exception {
        int id=tailId(e,"/delete/");
        send(e,200,"{\"ok\":"+db.remove(id)+"}","application/json");
    }
    static void items(HttpExchange e)throws Exception {
        StringBuilder s=new StringBuilder("[");
        var items=db.all();
        for(int i=0; i<items.size(); i++) {
            if(i>0)s.append(',');
            var v=items.get(i);
            s.append("{\"id\":").append(v.id()).append(",\"metadata\":").append(JsonUtil.q(v.metadata())).append(",\"category\":").append(JsonUtil.q(v.category())).append(",\"embedding\":").append(JsonUtil.arr(v.embedding())).append('}');
        }
        send(e,200,s.append(']').toString(),"application/json");
    }
    static void benchmark(HttpExchange e)throws Exception {
        Map<String,String>q=query(e);
        float[]v=JsonUtil.parseVector(q.get("v"));
        if(v.length!=DIMS) {
            send(e,400,"{\"error\":\"need 16D vector\"}","application/json");
            return;
        }
        var b=db.benchmark(v,parseInt(q.get("k"),5),q.getOrDefault("metric","cosine"));
        send(e,200,"{\"bruteforceUs\":"+b.bfUs()+",\"kdtreeUs\":"+b.kdUs()+",\"hnswUs\":"+b.hnswUs()+",\"itemCount\":"+b.n()+"}","application/json");
    }
    static void hnswInfo(HttpExchange e)throws Exception {
        var g=db.hnswInfo();
        StringBuilder s=new StringBuilder("{\"topLayer\":").append(g.topLayer()).append(",\"nodeCount\":").append(g.nodeCount()).append(",\"nodesPerLayer\":").append(JsonUtil.arrInt(g.nodesPerLayer())).append(",\"edgesPerLayer\":").append(JsonUtil.arrInt(g.edgesPerLayer())).append(",\"nodes\":[");
        for(int i=0; i<g.nodes().size(); i++) {
            if(i>0)s.append(',');
            var n=g.nodes().get(i);
            s.append("{\"id\":").append(n.id()).append(",\"metadata\":").append(JsonUtil.q(n.metadata())).append(",\"category\":").append(JsonUtil.q(n.category())).append(",\"maxLyr\":").append(n.maxLayer()).append('}');
        }
        s.append("],\"edges\":[");
        for(int i=0; i<g.edges().size(); i++) {
            if(i>0)s.append(',');
            var x=g.edges().get(i);
            s.append("{\"src\":").append(x.src()).append(",\"dst\":").append(x.dst()).append(",\"lyr\":").append(x.layer()).append('}');
        }
        send(e,200,s.append("]}").toString(),"application/json");
    }
    static void stats(HttpExchange e)throws Exception {
        send(e,200,"{\"count\":"+db.size()+",\"dims\":16,\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"],\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}","application/json");
    }
    static void status(HttpExchange e)throws Exception {
        boolean up=ollama.isAvailable();
        send(e,200,"{\"ollamaAvailable\":"+up+",\"embedModel\":"+JsonUtil.q(ollama.embedModel)+",\"genModel\":"+JsonUtil.q(ollama.genModel)+",\"docCount\":"+docDB.size()+",\"docDims\":"+docDB.getDims()+",\"demoDims\":16,\"demoCount\":"+db.size()+"}","application/json");
    }
    static void docInsert(HttpExchange e)throws Exception {
        String b=body(e),title=JsonUtil.extractString(b,"title"),text=JsonUtil.extractString(b,"text");
        if(title.isEmpty()||text.isEmpty()) {
            send(e,400,"{\"error\":\"need title and text\"}","application/json");
            return;
        }
        List<String>chunks=TextChunker.chunk(text,250,30);
        List<Integer>ids=new ArrayList<>();
        for(int i=0; i<chunks.size(); i++) {
            float[]emb=ollama.embed(chunks.get(i));
            if(emb.length==0) {
                send(e,503,"{\"error\":\"Ollama unavailable. Install Ollama and pull nomic-embed-text and llama3.2\"}","application/json");
                return;
            }
            String ct=chunks.size()>1?title+" ["+(i+1)+"/"+chunks.size()+"]":title;
            ids.add(docDB.insert(ct,chunks.get(i),emb));
        }
        send(e,200,"{\"ids\":"+ids.toString()+",\"chunks\":"+chunks.size()+",\"dims\":"+docDB.getDims()+"}","application/json");
    }
    static void docDelete(HttpExchange e)throws Exception {
        int id=tailId(e,"/doc/delete/");
        send(e,200,"{\"ok\":"+docDB.remove(id)+"}","application/json");
    }
    static void docList(HttpExchange e)throws Exception {
        StringBuilder s=new StringBuilder("[");
        var docs=docDB.all();
        for(int i=0; i<docs.size(); i++) {
            if(i>0)s.append(',');
            var d=docs.get(i);
            String p=d.text().substring(0,Math.min(120,d.text().length()))+(d.text().length()>120?"…":"");
            int words=d.text().isBlank()?0:d.text().trim().split("\\s+").length;
            s.append("{\"id\":").append(d.id()).append(",\"title\":").append(JsonUtil.q(d.title())).append(",\"preview\":").append(JsonUtil.q(p)).append(",\"words\":").append(words).append('}');
        }
        send(e,200,s.append(']').toString(),"application/json");
    }
    static void docSearch(HttpExchange e)throws Exception {
        String b=body(e),question=JsonUtil.extractString(b,"question");
        int k=JsonUtil.extractInt(b,"k",3);
        if(question.isEmpty()) {
            send(e,400,"{\"error\":\"need question\"}","application/json");
            return;
        }
        float[]emb=ollama.embed(question);
        if(emb.length==0) {
            send(e,503,"{\"error\":\"Ollama unavailable\"}","application/json");
            return;
        }
        var hits=docDB.search(emb,k,0.7f);
        StringBuilder s=new StringBuilder("{\"contexts\":[");
        for(int i=0; i<hits.size(); i++) {
            if(i>0)s.append(',');
            var h=hits.get(i);
            s.append("{\"id\":").append(h.getValue().id()).append(",\"title\":").append(JsonUtil.q(h.getValue().title())).append(",\"distance\":").append(String.format(Locale.US,"%.4f",h.getKey())).append('}');
        }
        send(e,200,s.append("]}").toString(),"application/json");
    }
    static void docAsk(HttpExchange e)throws Exception {
        String b=body(e),question=JsonUtil.extractString(b,"question");
        int k=JsonUtil.extractInt(b,"k",3);
        if(question.isEmpty()) {
            send(e,400,"{\"error\":\"need question\"}","application/json");
            return;
        }
        float[]q=ollama.embed(question);
        if(q.length==0) {
            send(e,503,"{\"error\":\"Ollama unavailable\"}","application/json");
            return;
        }
        var hits=docDB.search(q,k,0.7f);
        StringBuilder ctx=new StringBuilder();
        for(int i=0; i<hits.size(); i++)ctx.append('[').append(i+1).append("] ").append(hits.get(i).getValue().title()).append(":\\n").append(hits.get(i).getValue().text()).append("\\n\\n");
        String prompt="You are a helpful assistant. Answer the user's question directly. Use the provided context if it contains relevant information. If it doesn't, just use your own general knowledge. IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. Just answer the question naturally.\\n\\nContext:\\n"+ctx+"Question: "+question+"\\n\\nAnswer:";
        String answer=ollama.generate(prompt);
        StringBuilder s=new StringBuilder("{\"answer\":").append(JsonUtil.q(answer)).append(",\"model\":").append(JsonUtil.q(ollama.genModel)).append(",\"contexts\":[");
        for(int i=0; i<hits.size(); i++) {
            if(i>0)s.append(',');
            var h=hits.get(i);
            s.append("{\"id\":").append(h.getValue().id()).append(",\"title\":").append(JsonUtil.q(h.getValue().title())).append(",\"text\":").append(JsonUtil.q(h.getValue().text())).append(",\"distance\":").append(String.format(Locale.US,"%.4f",h.getKey())).append('}');
        }
        send(e,200,s.append("],\"docCount\":").append(docDB.size()).append('}').toString(),"application/json");
    }
    static int tailId(HttpExchange e,String prefix) {
        String p=e.getRequestURI().getPath();
        String x=p.substring(prefix.length());
        try {
            return Integer.parseInt(x);
        }catch(Exception ex) {
            return -1;
        }
    }
    static int parseInt(String s,int d) {
        try {
            return Integer.parseInt(s);
        }catch(Exception e) {
            return d;
        }
    }
}
