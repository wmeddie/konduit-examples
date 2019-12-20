package ai.xpress.dcx;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.RequestBuilder;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

public class Client {
    public static void main(String[] args) throws URISyntaxException, IOException {
        var file = new File("1902_airplane.png");
        try (var httpClient = HttpClients.createDefault()) {
            var uri = new URI("http://localhost:1337/json/image");
            var entity = MultipartEntityBuilder.create()
                    .addBinaryBody("default", file, ContentType.APPLICATION_OCTET_STREAM, "default.png")
                    .build();
            var requestCount = 1000;

            var stopWatch = StopWatch.createStarted();
            for (int i = 0; i < requestCount; i++) {
                doRequest(httpClient, uri, entity);
            }
            var seconds = stopWatch.getTime(TimeUnit.MILLISECONDS) / 1000.0;

            System.out.println("Took " + seconds + " for " + requestCount + " requets (" + (requestCount / seconds) + " RPS)");
        }
    }

    private static void doRequest(CloseableHttpClient httpClient, URI uri, HttpEntity entity) throws IOException {
        var req = RequestBuilder.post(uri)
                .setEntity(entity)
                .build();

        var response = httpClient.execute(req);
        var body = response.getEntity().getContent().readAllBytes();
        System.out.println(response.getStatusLine().toString() + " " + body.length);

        var json = new String(body, StandardCharsets.UTF_8);
        System.out.println(json);
    }
}
