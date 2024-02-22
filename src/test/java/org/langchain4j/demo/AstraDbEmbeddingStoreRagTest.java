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
import static io.stargate.sdk.data.domain.SimilarityMetric.cosine;
import static io.stargate.sdk.http.domain.FilterOperator.EQUALS_TO;

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
         * Accessing Astra Database
         *
         * Operations at DB level:
         * @see https://awesome-astra.github.io/docs/pages/develop/sdk/astra-db-client-java/#working-with-collections
         */
        AstraDB sampleDb = new AstraDB(astraToken, apiEndpoint, keyspace);
        log.info("You are connected to Astra on keyspace '{}'", sampleDb.getCurrentKeyspace());

        /*
         * Accessing a collection (create if not exists)
         *
         * Operations at Collection level:
         * @see https://awesome-astra.github.io/docs/pages/develop/sdk/astra-db-client-java/#working-with-documents
         */
        AstraDBCollection collection = sampleDb.createCollection(collectionName, vectorDimension, cosine);
        log.info("Your collection '{}' has been created (if needed) ", collectionName);

        // Flushing Collection
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

        // in that sample it was ask to create a metadata document_id for filters
        UUID documentId1 = UUID.randomUUID();
        UUID documentId2 = UUID.randomUUID();
        ingestDocument(documentId1, "johnny.txt", embeddingModel, embeddingStore);
        ingestDocument(documentId2, "story-about-happy-carrot.txt", embeddingModel, embeddingStore);

        // ---------------------
        // ------- RAG ---------
        // ---------------------

        // Specify the question you want to ask the model
        String question = "Who is Johnny ?";

        // Embed the question
        Response<Embedding> questionEmbedding = embeddingModel.embed(question);

        /*
         * ------------------------------------
         * Search with metadata filtering
         *
         * @see https://awesome-astra.github.io/docs/pages/develop/sdk/astra-db-client-java/#find-one
         * ------------------------------------
         */
        Filter filterByDocumentId = new Filter()
                .where("document_id", EQUALS_TO, documentId1.toString());

        // Limit result to keep relevant informations
        int maxResults = 5;

        // Minimum score to keep the result
        double minScore = 0.3;

        log.info("RAG Search with the Store");
        String ragContext = String.join(",", embeddingStore
            .findRelevant(questionEmbedding.content(), filterByDocumentId,  maxResults, minScore)
            .stream()
            .map(emb -> emb.embedded().text())
            .toList());

        // Prompting with Rag context
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Answer the following question to the best of your ability:\n"
                        + "Question:\n"
                        + "{{question}}\n"
                        + "Base your answer on the following information:\n"
                        + "{{information}}");
        Map<String, Object> variables = new HashMap<>();
        variables.put("question", question);
        variables.put("information", ragContext);

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
        // deleteDocumentById(collection, documentId1);
        // deleteDocumentById(collection, documentId2);
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

    private DeleteResult deleteDocumentById(AstraDBCollection collection, UUID documentId) {
        Filter deleteByid = new Filter().where("document_id", EQUALS_TO, documentId.toString());
        return collection.deleteMany(DeleteQuery.builder().filter(deleteByid).build());
    }


}
