package org.langchain4j.demo;

import com.dtsx.astra.sdk.AstraDB;
import com.dtsx.astra.sdk.AstraDBCollection;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModelName;
import dev.langchain4j.model.openai.OpenAiTokenizer;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.astradb.AstraDbEmbeddingStore;
import io.stargate.sdk.data.domain.SimilarityMetric;
import io.stargate.sdk.data.domain.query.DeleteQuery;
import io.stargate.sdk.data.domain.query.DeleteResult;
import io.stargate.sdk.data.domain.query.Filter;
import io.stargate.sdk.http.domain.FilterOperator;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static dev.langchain4j.model.openai.OpenAiModelName.GPT_3_5_TURBO;

@Slf4j
public class AstraDbEmbeddingStoreRagTest {

    // -- Those are the Astra Credentials you got after creating your DB

    String astraToken = "AstraCS:<change_me>";
    String apiEndpoint = "https://<change_me>>.apps.astra.datastax.com/api/json";
    String keyspace = "default_keyspace";
    String collectionName = "demo_collection";
    int vectorDimension = 1536;

    @Test
    public void testAstraDbEmbeddingStore() throws FileNotFoundException {

        /*
         * ------------------------------------
         *    Accessing your Astra Database
         * ------------------------------------
         * To know more about what you can do:
         * @see https://awesome-astra.github.io/docs/pages/develop/sdk/astra-db-client-java/#working-with-collections
         */
        AstraDB sampleDb = new AstraDB(astraToken, apiEndpoint, keyspace);
        log.info("You are connected to Astra on keyspace '{}'", sampleDb.getCurrentKeyspace());

        /*
         * ------------------------------------
         *   Create a retrieve a collection
         * ------------------------------------
         *  AstraDBCollection collection = sampleDb.collection("demo_collection");
         *  boolean collection = sampleDb.isCollectionExists("demo_collection")
         */
        AstraDBCollection collection = sampleDb.createCollection(collectionName, vectorDimension, SimilarityMetric.cosine);
        log.info("Your collection '{}' has been created (if needed) ", collectionName);

        /*
         * Flushing the Collection Before Starting
         * To delete the collection itself => sampleDb.deleteCollection("demo_collection");
         */
        collection.deleteAll();
        log.info("Your collection '{}' has been flushed", collectionName);

        // Langchain4j Embedding Store
        AstraDbEmbeddingStore embeddingStore = new AstraDbEmbeddingStore(collection);
        log.info("EmbeddingStore is ready.", collectionName);

        // OpenAI model
        EmbeddingModel embeddingModel = OpenAiEmbeddingModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName(OpenAiEmbeddingModelName.TEXT_EMBEDDING_3_SMALL)
                .build();

        // Ingesting Documents with in ID (if not provided a id is created for you
        UUID doc1Id = UUID.randomUUID();
        UUID doc2Id = UUID.randomUUID();
        ingestDocument(doc1Id, "johnny.txt", embeddingModel, embeddingStore);
        ingestDocument(doc2Id, "story-about-happy-carrot.txt", embeddingModel, embeddingStore);

        // -- Semantic Search

        // Specify the question you want to ask the model
        String question = "Who is Johnny ?";

        // Embed the question
        Response<Embedding> questionEmbedding = embeddingModel.embed(question);

        /*
         * ------------------------------------
         * Add a MedataFilter (search by doc ID)
         * To know more about filters:
         * @see https://awesome-astra.github.io/docs/pages/develop/sdk/astra-db-client-java/#find-one
         * new Filter().where("document_id", FilterOperator.EQUALS_TO, "johnny.txt");
         * ------------------------------------
         */
        Filter filterByDocumentId = new Filter()
                .where("document_id", FilterOperator.EQUALS_TO, doc1Id.toString());

        // Limit result to keep relevant informations
        int maxResults = 5;

        // You can directly Filter on the collection
        log.info("RAG Search with the Collection");
        List<String> ragChunks = ragSearchWithTheCollection(collection, questionEmbedding.content().vector(), filterByDocumentId, maxResults);
        ragChunks.forEach(chunk -> log.info(" + chunk: {}", chunk));

        log.info("RAG Search with the Store");
        List<String> ragChunks2 =  ragSearchWithStore(embeddingStore, questionEmbedding.content(), filterByDocumentId, maxResults, 0.3);
        ragChunks2.forEach(chunk -> log.info(" + chunk: {}", chunk));

        // Prompting with Rag context
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Answer the following question to the best of your ability:\n"
                        + "Question:\n"
                        + "{{question}}\n"
                        + "Base your answer on the following information:\n"
                        + "{{information}}");
        Map<String, Object> variables = new HashMap<>();
        variables.put("question", question);
        variables.put("information", String.join(",", ragChunks2));

        Prompt prompt = promptTemplate.apply(variables);
        log.info("Final Prompt {}", prompt.text());

        // Send the prompt to the OpenAI chat model
        ChatLanguageModel chatModel = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName(GPT_3_5_TURBO)
                .temperature(0.7)
                .timeout(Duration.ofSeconds(15))
                .maxRetries(3)
                .logResponses(true)
                .logRequests(true)
                .build();

        Response<AiMessage> aiMessage = chatModel.generate(prompt.toUserMessage());

        // See an answer from the model
        String answer = aiMessage.content().text();
        log.info("Answer from the model: {}", answer);


        // Clean up
        // deleteDocumentById(collection, doc1Id);
        // deleteDocumentById(collection, doc2Id);

    }

    private List<String> ragSearchWithStore(AstraDbEmbeddingStore embeddingStore, Embedding embeddings, Filter metadataFilter, int maxResults, double minScore) {
        return embeddingStore
                .findRelevant(embeddings, metadataFilter,  maxResults, minScore)
                .stream()
                .map(emb -> emb.embedded().text()).toList();
    }

    private DeleteResult deleteDocumentById(AstraDBCollection collection, UUID documentId) {
        Filter deleteByid = new Filter().where("document_id", FilterOperator.EQUALS_TO, documentId.toString());
        return collection.deleteMany(DeleteQuery.builder().filter(deleteByid).build());
    }

    private List<String> ragSearchWithTheCollection(AstraDBCollection collection, float[] embeddings, Filter metadataFilter, int maxResults) {
        return collection
                .findVector(embeddings, metadataFilter, maxResults)
                .map(res -> (String) res.getData().get("body_blob"))
                .toList();
    }

    private void ingestDocument(UUID docId, String documentName, EmbeddingModel embeddingModel, AstraDbEmbeddingStore embeddingStore) {

        // Splitter is important, not the whole document fit a single embedding
        DocumentSplitter splitter = DocumentSplitters
                .recursive(100, 10, new OpenAiTokenizer(GPT_3_5_TURBO));

        // Different loader per document extensions
        DocumentParser documentParser = new TextDocumentParser();

        // Load document binary content as inputstream
        File myFile = new File(getClass().getResource("/" + documentName).getFile());
        Document myDocument = FileSystemDocumentLoader.loadDocument(myFile.toPath(), documentParser);

        // Langchain4j Ingestor
        EmbeddingStoreIngestor.builder()
                .documentSplitter(splitter)
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .textSegmentTransformer(ts -> {
                    // '_id' is the technical identifier
                    ts.metadata().add("document_id", docId.toString());
                    ts.metadata().add("document_format", "text");
                    return ts;
                }).build().ingest(myDocument);

        log.info("Document '{}' has been ingested with id {}", documentName, docId);
    }
}
