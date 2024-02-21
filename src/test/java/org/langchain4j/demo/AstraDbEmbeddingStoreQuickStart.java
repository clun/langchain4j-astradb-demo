package org.langchain4j.demo;

import com.dtsx.astra.sdk.AstraDB;
import com.dtsx.astra.sdk.AstraDBAdmin;
import com.dtsx.astra.sdk.AstraDBCollection;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModelName;
import dev.langchain4j.store.embedding.astradb.AstraDbEmbeddingStore;
import io.stargate.sdk.data.domain.SimilarityMetric;
import org.junit.jupiter.api.Test;

public class AstraDbEmbeddingStoreQuickStart {

    // Assuming you have a running DB on Astra

    String astraToken = "<change_me>";
    String databaseEndpoint = "<change_me>";

    @Test
    public void testAstraDbEmbeddingStore() {

        // Connecting to your Database
        AstraDBAdmin astraDBAdminClient = new AstraDBAdmin(astraToken);
        AstraDB sampleDb = astraDBAdminClient.database("cultzyme_local_env_vector_db");
        sampleDb.changeKeyspace("default_keyspace"); // optional

        // Listing Collections
        sampleDb.findAllCollections().forEach(collection -> {
            System.out.println("Collection: " + collection.getName() + " dimension:" + collection.getOptions().getVector().getDimension());
        });

        // Creating a collection to store the vector embeddings
        AstraDBCollection collection = sampleDb.createCollection("test", 1536, SimilarityMetric.cosine);
        System.out.println("Your collection is created.");

        // Initializing the Store with the collection
        AstraDbEmbeddingStore embeddingStore = new AstraDbEmbeddingStore(collection);

        // Use a model like openAI
        EmbeddingModel embeddingModel = OpenAiEmbeddingModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName(OpenAiEmbeddingModelName.TEXT_EMBEDDING_3_SMALL)
                .build();

    }
}
