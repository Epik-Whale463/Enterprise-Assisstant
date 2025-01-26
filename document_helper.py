import logging
import os
from collections import Counter
from typing import List
from typing import Optional, Dict, Any

import nltk
import spacy
from langchain.chains import ConversationalRetrievalChain, load_summarize_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from nltk.corpus import stopwords
from qdrant_client import QdrantClient, models
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    text = ' '.join(text.split())
    text = ''.join(char for char in text if char.isprintable())
    return text.strip()

def validate_inputs(content: str, summarization_type: str, summary_length: str) -> tuple[bool, Optional[str]]:
    if not content or not content.strip():
        return False, "Content is empty. Please provide valid text for summarization."

    valid_types = {"extractive", "abstractive"}
    if summarization_type not in valid_types:
        return False, f"Invalid summarization type. Please select from: {', '.join(valid_types)}"

    valid_lengths = {"short", "medium", "long"}
    if summary_length not in valid_lengths:
        return False, f"Invalid summary length. Please select from: {', '.join(valid_lengths)}"

    return True, None

def get_abstractive_summary(content: str, ratio: float, summary_length: str) -> str:
    """
    Generates an abstractive summary using LangChain and Google Generative AI,
    structured with headings and bullet points for markdown rendering.
    """
    try:
        llm = GoogleGenerativeAI(
            google_api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-pro", temperature=0.7
        )

        # Enhanced prompt template for structured output
        prompt_template = """
        Analyze the following text and provide a meaningful coherent summary.
        The summary should be {summary_length} length and meaningful. Aim for a reduction ratio of around {ratio}.

        TEXT: {text}

        SUMMARY (Structured in plain text without any tags):
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"],
            partial_variables={"summary_length": summary_length, "ratio": ratio}
        )

        # Load the summarization chain
        summarize_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

        # Split the content into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200
        )
        texts = text_splitter.split_text(content)

        # Create Document objects from text chunks
        docs = [Document(page_content=t) for t in texts]

        # Generate the summary
        summary = summarize_chain.run(docs)

        return summary

    except Exception as e:
        logger.error(f"Abstractive summarization error: {str(e)}")
        raise RuntimeError(f"Summary generation failed: {str(e)}")


def get_extractive_summary(content: str, ratio: float) -> str:
    try:
        parser = PlaintextParser.from_string(content, Tokenizer("english"))
        summarizer = LexRankSummarizer()

        sentence_count = max(1, int(len(list(parser.document.sentences)) * ratio))
        summary_sentences = summarizer(parser.document, sentence_count)
        summary = " ".join([str(sent) for sent in summary_sentences])

        if not summary.strip():
            raise ValueError("Generated summary is empty")

        return summary

    except Exception as e:
        logger.error(f"Extractive summarization error: {str(e)}")
        raise RuntimeError(f"Summary generation failed: {str(e)}")

def generate_summary(
        content: str,
        summarization_type: str = "extractive",
        summary_length: str = "medium"
) -> Dict[str, Any]:
    try:
        is_valid, error_message = validate_inputs(content, summarization_type, summary_length)
        if not is_valid:
            return {"success": False, "error": error_message}

        ratio_map = {
            'short': 0.15,
            'medium': 0.30,
            'long': 0.50
        }
        ratio = ratio_map[summary_length]

        summary = (
            get_extractive_summary(content, ratio)
            if summarization_type == "extractive"
            else get_abstractive_summary(content, ratio, summary_length)
        )

        return {
            "success": True,
            "summary": summary,
            "metadata": {
                "original_length": len(content.split()),
                "summary_length": len(summary.split()),
                "reduction_ratio": 1 - (len(summary.split()) / len(content.split())),
                "summarization_type": summarization_type,
                "requested_length": summary_length
            }
        }

    except Exception as e:
        logger.error(f"Summary generation error: {str(e)}")
        return {
            "success": False,
            "error": f"Summarization failed: {str(e)}",
            "metadata": {
                "summarization_type": summarization_type,
                "requested_length": summary_length
            }
        }

def extract_keywords(content: str, options: List[str]) -> Dict[str, Any]:
    if not content.strip():
        return {"error": "Content is empty. Provide valid text for extraction."}

    stop_words = set(stopwords.words('english'))
    doc = nlp(content)
    results = {}

    if "technical" in options:
        technical_terms = [
            chunk.text for chunk in doc.noun_chunks
            if chunk.text.lower() not in stop_words
        ]
        results["Technical Terms"] = Counter(technical_terms).most_common(10)

    if "keyphrases" in options:
        key_phrases = [
            chunk.text for chunk in doc.noun_chunks
            if len(chunk.text.split()) > 1 and
               all(word.lower() not in stop_words for word in chunk.text.split())
        ]
        results["Key Phrases"] = Counter(key_phrases).most_common(10)

    if "entities" in options:
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]
        results["Named Entities"] = named_entities

    return results


# Initialize qdrant_client
qdrant_url = os.environ.get("QDRANT_URL")
api_key = os.environ.get("QDRANT_API_KEY")
qdrant_client = QdrantClient(url=qdrant_url, api_key=api_key)

# Initialize embeddings (replace with your actual embeddings initialization)
embedding = NVIDIAEmbeddings(
    nvidia_api_key=os.environ.get("NVIDIA_API_KEY"),
    model="snowflake/arctic-embed-l"
)

class ImprovedQdrantRetriever(BaseRetriever):
    def __init__(
            self,
            qdrant_client: QdrantClient,
            collection_name: str,
            embeddings,
            k: int = 3
    ):
        super().__init__()
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.k = k

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            query_embedding = self.embeddings.embed_query(query)

            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=self.k,
                with_payload=True
            )

            documents = [
                Document(
                    page_content=result.payload.get("text", ""),
                    metadata={"score": result.score}
                )
                for result in search_result
            ]

            return documents

        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            return []



def process_document_for_rag(content: str, collection_name: str = "current_document", embeddings=embedding):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_text(content)
        embeddings_list = embeddings.embed_documents(chunks)

        try:
            qdrant_client.delete_collection(collection_name)
        except Exception as delete_error:
            logger.warning(f"Collection deletion warning: {delete_error}")

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=len(embeddings_list[0]),
                distance=models.Distance.COSINE
            )
        )

        qdrant_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[
                models.PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={"text": chunk}
                ) for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings_list))
            ]
        )

        return True

    except Exception as e:
        logger.error(f"RAG document processing error: {str(e)}")
        return False

def get_conversational_chain(collection_name: str = "current_document", llm=None):
    try:
        if llm is None:
            llm = GoogleGenerativeAI(model='gemini-pro', google_api_key=os.environ.get("GOOGLE_API_KEY"))

        retriever = ImprovedQdrantRetriever(
            qdrant_client=qdrant_client,  # Use qdrant_client here
            collection_name=collection_name,
            embeddings=embedding
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )

        return chain

    except Exception as e:
        logger.error(f"Conversational chain creation error: {str(e)}")
        raise

def compose_email(prompt: str) -> str:
    try:
        llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.7)
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        logger.error(f"Email composition error: {str(e)}")
        return "An error occurred while composing the email."